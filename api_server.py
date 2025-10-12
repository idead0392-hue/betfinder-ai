# api_server.py - refactor to use SportbexProvider
# api_server.py - refactor to use SportbexProvider with comprehensive monitoring
from flask import Flask, request, jsonify, g
import time
import logging
import traceback
import sqlite3
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Any, Dict, Optional
from threading import Lock
import functools
import pandas as pd

# Import our provider system
try:
    from api_providers import SportbexProvider, SportType, create_sportbex_provider
    PROVIDER_AVAILABLE = True
except ImportError:
    PROVIDER_AVAILABLE = False
    SportbexProvider = None
    SportType = None

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Monitoring and metrics storage
class MetricsCollector:
    """Thread-safe metrics collection for API monitoring."""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.lock = Lock()
        self.request_times = deque(maxlen=max_history)
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'last_request': None
        })
        self.provider_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'last_success': None,
            'last_error': None
        })
        self.start_time = datetime.utcnow()
    
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record request metrics."""
        with self.lock:
            self.request_times.append({
                'timestamp': datetime.utcnow(),
                'duration': duration,
                'endpoint': endpoint,
                'success': success
            })
            
            stats = self.endpoint_stats[endpoint]
            stats['count'] += 1
            stats['total_time'] += duration
            stats['last_request'] = datetime.utcnow()
            
            if not success:
                stats['errors'] += 1
    
    def record_provider_call(self, provider_name: str, operation: str, duration: float, success: bool = True, error: str = None):
        """Record provider-specific metrics."""
        with self.lock:
            key = f"{provider_name}.{operation}"
            stats = self.provider_stats[key]
            stats['count'] += 1
            stats['total_time'] += duration
            
            if success:
                stats['last_success'] = datetime.utcnow()
            else:
                stats['errors'] += 1
                stats['last_error'] = {
                    'timestamp': datetime.utcnow(),
                    'error': error
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.lock:
            now = datetime.utcnow()
            uptime_seconds = (now - self.start_time).total_seconds()
            
            # Calculate recent performance (last 5 minutes)
            recent_cutoff = now - timedelta(minutes=5)
            recent_requests = [r for r in self.request_times if r['timestamp'] > recent_cutoff]
            
            total_requests = len(self.request_times)
            recent_count = len(recent_requests)
            
            stats = {
                'uptime_seconds': uptime_seconds,
                'total_requests': total_requests,
                'recent_requests_5min': recent_count,
                'requests_per_second': recent_count / 300 if recent_count > 0 else 0,
                'average_response_time': sum(r['duration'] for r in recent_requests) / recent_count if recent_count > 0 else 0,
                'error_rate': sum(1 for r in recent_requests if not r['success']) / recent_count if recent_count > 0 else 0,
                'endpoints': dict(self.endpoint_stats),
                'providers': dict(self.provider_stats)
            }
            
            return stats

# Global metrics collector
metrics = MetricsCollector()

# Provider initialization with monitoring
provider = None
provider_status = {
    'initialized': False,
    'last_check': None,
    'error': None,
    'provider_name': 'sportbex'
}

def initialize_provider():
    """Initialize provider with error handling and monitoring."""
    global provider, provider_status
    
    if not PROVIDER_AVAILABLE:
        provider_status.update({
            'initialized': False,
            'error': 'SportbexProvider module not available',
            'last_check': datetime.utcnow()
        })
        return False
    
    try:
        start_time = time.time()
        provider = create_sportbex_provider()
        duration = time.time() - start_time
        
        # Test provider connectivity
        health_response = provider.health_check()
        
        if health_response and health_response.success:
            provider_status.update({
                'initialized': True,
                'error': None,
                'last_check': datetime.utcnow(),
                'initialization_time': duration
            })
            metrics.record_provider_call('sportbex', 'initialization', duration, True)
            logger.info(f"SportbexProvider initialized successfully in {duration:.2f}s")
            return True
        else:
            error_msg = health_response.error_message if health_response else "Unknown health check failure"
            provider_status.update({
                'initialized': False,
                'error': f"Provider health check failed: {error_msg}",
                'last_check': datetime.utcnow()
            })
            metrics.record_provider_call('sportbex', 'initialization', duration, False, error_msg)
            logger.error(f"SportbexProvider health check failed: {error_msg}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = str(e)
        provider_status.update({
            'initialized': False,
            'error': error_msg,
            'last_check': datetime.utcnow()
        })
        metrics.record_provider_call('sportbex', 'initialization', duration, False, error_msg)
        logger.error(f"Failed to initialize SportbexProvider: {error_msg}")
        return False

# Initialize provider on startup
initialize_provider()

# Request monitoring decorator
def monitor_request(endpoint_name: str):
    """Decorator to monitor request performance and errors."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger.error(f"Error in {endpoint_name}: {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_request(endpoint_name, duration, success)
        return wrapper
    return decorator

# Helper for consistent error responses
def make_error(message: str, status: int = 500, detail: Dict[str, Any] = None):
    payload = {"error": message, "timestamp": datetime.utcnow().isoformat()}
    if detail:
        payload["detail"] = detail
    return jsonify(payload), status

def provider_call_wrapper(operation: str, func, *args, **kwargs):
    """Wrapper for provider calls with monitoring."""
    if not provider_status['initialized']:
        return None, "Provider not initialized"
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        metrics.record_provider_call('sportbex', operation, duration, True)
        return result, None
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        metrics.record_provider_call('sportbex', operation, duration, False, error_msg)
        return None, error_msg

# ===== HEALTH CHECK ENDPOINTS =====

@app.route('/health', methods=['GET'])
@monitor_request('health')
def health_simple():
    """Simple health check for load balancers."""
    return jsonify({"status": "ok"}), 200

@app.route('/health/detailed', methods=['GET'])
@monitor_request('health_detailed')
def health_detailed():
    """Detailed health check with provider status."""
    health_data = {
        "status": "ok" if provider_status['initialized'] else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "provider": {
            "name": provider_status['provider_name'],
            "initialized": provider_status['initialized'],
            "last_check": provider_status['last_check'].isoformat() if provider_status['last_check'] else None,
            "error": provider_status['error']
        },
        "uptime_seconds": (datetime.utcnow() - metrics.start_time).total_seconds()
    }
    
    status_code = 200 if provider_status['initialized'] else 503
    return jsonify(health_data), status_code

@app.route('/health/provider', methods=['GET'])
@monitor_request('health_provider')
def health_provider():
    """Provider-specific health check with real API test."""
    if not provider_status['initialized']:
        return jsonify({
            "status": "error",
            "provider": provider_status['provider_name'],
            "error": provider_status['error'],
            "timestamp": datetime.utcnow().isoformat()
        }), 503
    
    # Perform real health check
    health_response, error = provider_call_wrapper('health_check', provider.health_check)
    
    if error:
        return jsonify({
            "status": "error",
            "provider": provider_status['provider_name'],
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }), 503
    
    if health_response and health_response.success:
        return jsonify({
            "status": "ok",
            "provider": provider_status['provider_name'],
            "response_time_ms": health_response.response_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    else:
        return jsonify({
            "status": "error",
            "provider": provider_status['provider_name'],
            "error": health_response.error_message if health_response else "Unknown error",
            "timestamp": datetime.utcnow().isoformat()
        }), 503

@app.route('/health/database', methods=['GET'])
@monitor_request('health_database')
def health_database():
    """Database connectivity check (if database is used)."""
    # Check if bet_db.py database exists and is accessible
    try:
        db_path = 'bets.db'  # Assuming default database name
        if os.path.exists(db_path):
            # Try to connect and run a simple query
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            
            return jsonify({
                "status": "ok",
                "database": "sqlite",
                "path": db_path,
                "timestamp": datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                "status": "warning",
                "database": "sqlite",
                "message": "Database file does not exist",
                "path": db_path,
                "timestamp": datetime.utcnow().isoformat()
            }), 200
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "database": "sqlite",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 503

@app.route('/metrics', methods=['GET'])
@monitor_request('metrics')
def get_metrics():
    """Comprehensive metrics endpoint for monitoring systems."""
    stats = metrics.get_stats()
    
    # Add provider status to metrics
    stats['provider_status'] = provider_status.copy()
    if stats['provider_status']['last_check']:
        stats['provider_status']['last_check'] = stats['provider_status']['last_check'].isoformat()
    
    # Convert datetime objects to ISO strings
    for endpoint, data in stats['endpoints'].items():
        if data['last_request']:
            data['last_request'] = data['last_request'].isoformat()
    
    for provider_op, data in stats['providers'].items():
        if data['last_success']:
            data['last_success'] = data['last_success'].isoformat()
        if data['last_error'] and isinstance(data['last_error'], dict):
            if 'timestamp' in data['last_error']:
                data['last_error']['timestamp'] = data['last_error']['timestamp'].isoformat()
    
    return jsonify(stats), 200

# Legacy health endpoint for backward compatibility
@app.route('/api/health', methods=['GET'])
@monitor_request('api_health_legacy')
def health_legacy():
    """Legacy health endpoint."""
    return jsonify({
        "status": "ok" if provider_status['initialized'] else "degraded",
        "provider": provider_status['provider_name']
    }), 200 if provider_status['initialized'] else 503

# Odds endpoints
@app.route('/api/odds', methods=['GET'])
@monitor_request('api_odds')
def get_odds():
    try:
        sport = request.args.get('sport')
        market = request.args.get('market')
        league = request.args.get('league')
        
        res, error = provider_call_wrapper('get_odds', provider.get_odds, sport=sport, market=market, league=league)
        if error:
            return make_error("Failed to fetch odds", 502, {"provider_error": error})
        
        return jsonify(res)
    except Exception as e:
        logger.error(f"Error in get_odds: {str(e)}\n{traceback.format_exc()}")
        return make_error("Failed to fetch odds", 502, {"exception": str(e)})

# Props endpoints
@app.route('/api/props', methods=['GET'])
@monitor_request('api_props')
def get_props():
    try:
        sport = request.args.get('sport')
        league = request.args.get('league')
        player = request.args.get('player')
        market = request.args.get('market')
        
        res, error = provider_call_wrapper('get_props', provider.get_props, sport=sport, league=league, player=player, market=market)
        if error:
            return make_error("Failed to fetch props", 502, {"provider_error": error})
        
        return jsonify(res)
    except Exception as e:
        logger.error(f"Error in get_props: {str(e)}\n{traceback.format_exc()}")
        return make_error("Failed to fetch props", 502, {"exception": str(e)})

# Competitions / schedule endpoints
@app.route('/api/competitions', methods=['GET'])
@monitor_request('api_competitions')
def get_competitions():
    try:
        sport = request.args.get('sport')
        league = request.args.get('league')
        
        res, error = provider_call_wrapper('get_competitions', provider.get_competitions, sport=sport, league=league)
        if error:
            return make_error("Failed to fetch competitions", 502, {"provider_error": error})
        
        return jsonify(res)
    except Exception as e:
        logger.error(f"Error in get_competitions: {str(e)}\n{traceback.format_exc()}")
        return make_error("Failed to fetch competitions", 502, {"exception": str(e)})

# Backward-compat legacy endpoints mapping
@app.route('/api/lines', methods=['GET'])
@monitor_request('api_lines_legacy')
def get_lines_legacy():
    try:
        # Map to provider odds
        sport = request.args.get('sport')
        market = request.args.get('market')
        league = request.args.get('league')
        
        res, error = provider_call_wrapper('get_odds', provider.get_odds, sport=sport, market=market, league=league)
        if error:
            return make_error("Failed to fetch lines", 502, {"provider_error": error})
        
        return jsonify(res)
    except Exception as e:
        logger.error(f"Error in get_lines_legacy: {str(e)}\n{traceback.format_exc()}")
        return make_error("Failed to fetch lines", 502, {"exception": str(e)})

@app.route('/api/player_props', methods=['GET'])
@monitor_request('api_player_props_legacy')
def get_player_props_legacy():
    try:
        sport = request.args.get('sport')
        league = request.args.get('league')
        player = request.args.get('player')
        market = request.args.get('market')
        
        res, error = provider_call_wrapper('get_props', provider.get_props, sport=sport, league=league, player=player, market=market)
        if error:
            return make_error("Failed to fetch player props", 502, {"provider_error": error})
        
        return jsonify(res)
    except Exception as e:
        logger.error(f"Error in get_player_props_legacy: {str(e)}\n{traceback.format_exc()}")
        return make_error("Failed to fetch player props", 502, {"exception": str(e)})

# Documentation route to aid migration
@app.route('/api/meta/provider', methods=['GET'])
@monitor_request('api_meta_provider')
def provider_meta():
    try:
        return jsonify({
            "active_provider": provider_status['provider_name'],
            "initialized": provider_status['initialized'],
            "last_check": provider_status['last_check'].isoformat() if provider_status['last_check'] else None,
            "error": provider_status['error'],
            "supported_sports": [sport.value for sport in SportType] if SportType else [],
            "capabilities": provider.capabilities() if provider and hasattr(provider, 'capabilities') else {}
        })
    except Exception as e:
        logger.error(f"Error in provider_meta: {str(e)}\n{traceback.format_exc()}")
        return make_error("Failed to fetch provider metadata", 500, {"exception": str(e)})

# =====================
# PrizePicks CSV bridge
# =====================

def _parse_event_start(game_date: str, game_time_et: str) -> str:
    try:
        if not (game_date and game_time_et):
            return ''
        t = str(game_time_et).replace('ET', '').strip()
        # Try 12-hour first then 24-hour
        try:
            dt_local = datetime.strptime(f"{game_date} {t}", "%Y-%m-%d %I:%M %p")
        except Exception:
            dt_local = datetime.strptime(f"{game_date} {t}", "%Y-%m-%d %H:%M")
        # Treat provided time as America/New_York
        from zoneinfo import ZoneInfo
        dt_et = dt_local.replace(tzinfo=ZoneInfo('America/New_York'))
        return dt_et.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
    except Exception:
        return ''

@app.route('/api/prizepicks/props', methods=['GET'])
@monitor_request('api_prizepicks_props')
def prizepicks_props():
    """Serve live props from PrizePicks CSV with event date/time and freshness.

    Query params:
      - sport: optional logical sport key (basketball, football, hockey, etc.) to filter
      - limit: optional int cap on results
    """
    try:
        csv_path = os.environ.get('PRIZEPICKS_CSV', 'prizepicks_props.csv')
        if not os.path.exists(csv_path):
            return make_error("PrizePicks CSV not found", 404, {"path": csv_path})

        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        name_c = cols.get('name') or cols.get('player')
        line_c = cols.get('points') or cols.get('line')
        prop_c = cols.get('prop')
        league_c = cols.get('league')
        team_c = cols.get('team')
        matchup_c = cols.get('matchup') or cols.get('game')
        gdate_c = cols.get('game_date')
        gtime_c = cols.get('game_time')
        lupd_c = cols.get('last_updated')

        def to_float(x):
            try:
                s = str(x).strip().replace('+','')
                return float(s)
            except Exception:
                return None

        sport_filter = (request.args.get('sport') or '').strip().lower()
        limit = int(request.args.get('limit') or 0)

        items = []
        for _, row in df.iterrows():
            player = str(row.get(name_c, '')).strip() if name_c else ''
            prop_type = str(row.get(prop_c, '')).strip()
            team = str(row.get(team_c, '')).strip() if team_c else ''
            league = str(row.get(league_c, '')).strip() if league_c else ''
            matchup = str(row.get(matchup_c, '')).strip() if matchup_c else ''
            date = str(row.get(gdate_c, '')).strip() if gdate_c else ''
            time_et = str(row.get(gtime_c, '')).strip() if gtime_c else ''
            last_updated = str(row.get(lupd_c, '')).strip() if lupd_c else ''
            line = to_float(row.get(line_c)) if line_c else None

            # Optional sport filter - approximate using league
            if sport_filter:
                league_lc = league.lower()
                mapping = {
                    'basketball': ['nba','wnba','cbb'],
                    'football': ['nfl','nfl1h','nfl1q','nfl2h'],
                    'college_football': ['cfb','ncaa'],
                    'baseball': ['mlb','mlblive'],
                    'hockey': ['nhl','nhl1p'],
                    'soccer': ['soccer']
                }
                acceptable = mapping.get(sport_filter, [])
                if acceptable and not any(a in league_lc for a in acceptable):
                    continue

            event_start_time = _parse_event_start(date, time_et)
            # Filter out expired past events when we have a time
            if event_start_time:
                try:
                    dt = datetime.fromisoformat(event_start_time.replace('Z','+00:00'))
                    if dt < datetime.now(timezone.utc):
                        continue
                except Exception:
                    pass

            item = {
                "player": player,
                "team": team,
                "matchup": matchup,
                "prop": prop_type,
                "line": line,
                "league": league,
                "date": date,
                "time": time_et,
                "event_start_time": event_start_time,
                "last_updated": last_updated
            }
            items.append(item)

            if limit and len(items) >= limit:
                break

        # Compute overall last_updated for payload transparency
        overall_updated = None
        for it in items:
            lu = it.get('last_updated')
            if lu:
                try:
                    dt = datetime.fromisoformat(lu.replace('Z','+00:00'))
                    if (overall_updated is None) or (dt > overall_updated):
                        overall_updated = dt
                except Exception:
                    pass
        if overall_updated is None:
            try:
                overall_updated = datetime.fromtimestamp(os.path.getmtime(csv_path), timezone.utc)
            except Exception:
                pass

        return jsonify({
            "count": len(items),
            "last_updated": overall_updated.isoformat().replace('+00:00','Z') if overall_updated else None,
            "props": items
        })
    except Exception as e:
        logger.error(f"Error in prizepicks_props: {str(e)}\n{traceback.format_exc()}")
        return make_error("Failed to load PrizePicks props", 500, {"exception": str(e)})

# Error handler for unhandled exceptions
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return make_error("Internal server error", 500)

@app.errorhandler(404)
def not_found(error):
    return make_error("Endpoint not found", 404)

if __name__ == '__main__':
    logger.info("Starting BetFinder AI API Server with comprehensive monitoring")
    logger.info(f"Provider available: {PROVIDER_AVAILABLE}")
    logger.info(f"Provider initialized: {provider_status['initialized']}")
    app.run(host='0.0.0.0', port=5001)
