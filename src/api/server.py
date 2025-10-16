# src/api/server.py
from flask import Flask, request, jsonify
import time
import logging
import traceback
import sqlite3
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Any, Dict
from threading import Lock
import functools
import pandas as pd

# Import our provider system
try:
    from .providers import MockSportsDataProvider, SportType, create_data_provider
    PROVIDER_AVAILABLE = True
except ImportError:
    PROVIDER_AVAILABLE = False
    MockSportsDataProvider = None
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
    
    def __init__(self):
        self.lock = Lock()
        self.request_times = deque(maxlen=1000)  # Last 1000 requests
        self.error_counts = defaultdict(int)
        self.endpoint_metrics = defaultdict(lambda: {'count': 0, 'total_time': 0})
        self.provider_metrics = defaultdict(lambda: {'calls': 0, 'errors': 0, 'total_time': 0})
        
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record a request metric."""
        with self.lock:
            self.request_times.append(time.time())
            self.endpoint_metrics[endpoint]['count'] += 1
            self.endpoint_metrics[endpoint]['total_time'] += duration
            if not success:
                self.error_counts[endpoint] += 1
                
    def record_provider_call(self, method: str, duration: float, success: bool = True):
        """Record a provider call metric."""
        with self.lock:
            self.provider_metrics[method]['calls'] += 1
            self.provider_metrics[method]['total_time'] += duration
            if not success:
                self.provider_metrics[method]['errors'] += 1
                
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics statistics."""
        with self.lock:
            now = time.time()
            recent_requests = sum(1 for t in self.request_times if now - t < 60)
            
            return {
                'requests_per_minute': recent_requests,
                'total_errors': sum(self.error_counts.values()),
                'endpoint_stats': dict(self.endpoint_metrics),
                'provider_stats': dict(self.provider_metrics)
            }

metrics = MetricsCollector()

# Global provider instance and status
provider = None
provider_status = {
    'initialized': False,
    'last_error': None,
    'last_check': None,
    'provider_type': 'mock-data'
}

def initialize_provider():
    """Initialize the data provider."""
    global provider, provider_status
    
    if not PROVIDER_AVAILABLE:
        logger.error("Provider module not available")
        provider_status['last_error'] = "Provider module not available"
        return False
        
    try:
        logger.info("Initializing MockSportsDataProvider...")
        provider = create_data_provider()
        provider_status['initialized'] = True
        provider_status['last_error'] = None
        provider_status['last_check'] = datetime.now(timezone.utc).isoformat()
        logger.info("MockSportsDataProvider initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MockSportsDataProvider: {e}")
        logger.error(traceback.format_exc())
        provider_status['last_error'] = str(e)
        provider_status['last_check'] = datetime.now(timezone.utc).isoformat()
        return False

def provider_call_wrapper(method_name: str):
    """Decorator to wrap provider calls with error handling and metrics."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not provider_status['initialized']:
                logger.warning(f"Provider call '{method_name}' attempted before initialization")
                return None
                
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_provider_call(method_name, duration, success=True)
                logger.debug(f"Provider call '{method_name}' completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_provider_call(method_name, duration, success=False)
                logger.error(f"Provider call '{method_name}' failed after {duration:.3f}s: {e}")
                logger.error(traceback.format_exc())
                return None
        return wrapper
    return decorator

@provider_call_wrapper('get_sports')
def get_sports():
    """Get available sports from the mock data provider."""
    return provider.get_sports()

@provider_call_wrapper('get_leagues')
def get_leagues(sport_type: SportType):
    """Get leagues for a sport from the mock data provider."""
    return provider.get_leagues(sport_type)

@provider_call_wrapper('get_events')
def get_events(sport_type: SportType, league_id: str = None):
    """Get events from the mock data provider."""
    return provider.get_events(sport_type, league_id)

@provider_call_wrapper('get_odds')
def get_odds(event_id: str):
    """Get odds for an event from the mock data provider."""
    return provider.get_odds(event_id)

# Initialize provider on startup
initialize_provider()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with provider status."""
    stats = metrics.get_stats()
    
    health_info = {
        'status': 'healthy' if provider_status['initialized'] else 'degraded',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'provider': {
            'type': 'mock-data',
            'initialized': provider_status['initialized'],
            'last_check': provider_status['last_check'],
            'last_error': provider_status['last_error']
        },
        'metrics': {
            'requests_per_minute': stats['requests_per_minute'],
            'total_errors': stats['total_errors']
        }
    }
    
    status_code = 200 if provider_status['initialized'] else 503
    return jsonify(health_info), status_code

# Metrics endpoint
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get detailed metrics."""
    stats = metrics.get_stats()
    return jsonify({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': stats,
        'provider_status': provider_status
    })

# Sports endpoints
@app.route('/api/sports', methods=['GET'])
def api_sports():
    """Get list of available sports."""
    start_time = time.time()
    
    try:
        sports = get_sports()
        if sports is None:
            metrics.record_request('/api/sports', time.time() - start_time, success=False)
            return jsonify({'error': 'Failed to fetch sports from mock data provider'}), 503
            
        metrics.record_request('/api/sports', time.time() - start_time, success=True)
        return jsonify(sports)
    except Exception as e:
        logger.error(f"Error in /api/sports: {e}")
        logger.error(traceback.format_exc())
        metrics.record_request('/api/sports', time.time() - start_time, success=False)
        return jsonify({'error': str(e)}), 500

@app.route('/api/leagues/<sport>', methods=['GET'])
def api_leagues(sport):
    """Get leagues for a specific sport."""
    start_time = time.time()
    
    try:
        # Convert sport string to SportType enum
        try:
            sport_type = SportType[sport.upper()]
        except KeyError:
            metrics.record_request(f'/api/leagues/{sport}', time.time() - start_time, success=False)
            return jsonify({'error': f'Invalid sport: {sport}'}), 400
            
        leagues = get_leagues(sport_type)
        if leagues is None:
            metrics.record_request(f'/api/leagues/{sport}', time.time() - start_time, success=False)
            return jsonify({'error': f'Failed to fetch leagues from mock data provider'}), 503
            
        metrics.record_request(f'/api/leagues/{sport}', time.time() - start_time, success=True)
        return jsonify(leagues)
    except Exception as e:
        logger.error(f"Error in /api/leagues/{sport}: {e}")
        logger.error(traceback.format_exc())
        metrics.record_request(f'/api/leagues/{sport}', time.time() - start_time, success=False)
        return jsonify({'error': str(e)}), 500

@app.route('/api/events/<sport>', methods=['GET'])
def api_events(sport):
    """Get events for a specific sport."""
    start_time = time.time()
    
    try:
        # Convert sport string to SportType enum
        try:
            sport_type = SportType[sport.upper()]
        except KeyError:
            metrics.record_request(f'/api/events/{sport}', time.time() - start_time, success=False)
            return jsonify({'error': f'Invalid sport: {sport}'}), 400
            
        league_id = request.args.get('league_id')
        events = get_events(sport_type, league_id)
        
        if events is None:
            metrics.record_request(f'/api/events/{sport}', time.time() - start_time, success=False)
            return jsonify({'error': f'Failed to fetch events from mock data provider'}), 503
            
        metrics.record_request(f'/api/events/{sport}', time.time() - start_time, success=True)
        return jsonify(events)
    except Exception as e:
        logger.error(f"Error in /api/events/{sport}: {e}")
        logger.error(traceback.format_exc())
        metrics.record_request(f'/api/events/{sport}', time.time() - start_time, success=False)
        return jsonify({'error': str(e)}), 500

@app.route('/api/odds/<event_id>', methods=['GET'])
def api_odds(event_id):
    """Get odds for a specific event."""
    start_time = time.time()
    
    try:
        odds = get_odds(event_id)
        if odds is None:
            metrics.record_request(f'/api/odds/{event_id}', time.time() - start_time, success=False)
            return jsonify({'error': f'Failed to fetch odds from mock data provider'}), 503
            
        metrics.record_request(f'/api/odds/{event_id}', time.time() - start_time, success=True)
        return jsonify(odds)
    except Exception as e:
        logger.error(f"Error in /api/odds/{event_id}: {e}")
        logger.error(traceback.format_exc())
        metrics.record_request(f'/api/odds/{event_id}', time.time() - start_time, success=False)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
