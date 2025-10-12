"""
Agent Monitor - Comprehensive monitoring and analytics system for BetFinder AI agents
Tracks performance metrics, pick quality, outcomes, and provides real-time insights
"""

import json
import time
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

from agent_logger import get_logger, AgentLogEntry, AgentEvent, LogLevel

class MetricType(Enum):
    """Types of metrics to track"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    OUTCOME = "outcome"
    FINANCIAL = "financial"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"

@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent"""
    agent_name: str
    sport: str
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    
    # Pick generation metrics
    picks_generated: int = 0
    avg_confidence: float = 0.0
    confidence_distribution: Dict[str, int] = None
    
    # Outcome metrics
    picks_resolved: int = 0
    picks_won: int = 0
    picks_lost: int = 0
    picks_pushed: int = 0
    win_rate: float = 0.0
    
    # Financial metrics
    total_profit: float = 0.0
    total_volume: float = 0.0
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    error_counts: Dict[str, int] = None
    
    # Time-based metrics
    uptime_percentage: float = 100.0
    last_active: Optional[str] = None
    
    # Calculated metrics
    requests_per_hour: float = 0.0
    picks_per_hour: float = 0.0
    
    def __post_init__(self):
        if self.confidence_distribution is None:
            self.confidence_distribution = {}
        if self.error_counts is None:
            self.error_counts = {}

@dataclass
class PickAnalytics:
    """Analytics for individual picks"""
    pick_id: str
    agent_name: str
    sport: str
    
    # Pick details
    pick_type: str
    player_name: str
    stat_category: str
    line_value: float
    prediction: str  # 'over' or 'under'
    confidence: float
    
    # Market data
    odds: Optional[float] = None
    expected_value: Optional[float] = None
    kelly_size: Optional[float] = None
    
    # Outcome data
    outcome: Optional[str] = None  # 'win', 'loss', 'push', 'pending'
    actual_value: Optional[float] = None
    profit_loss: Optional[float] = None
    
    # Timing
    generated_at: str = ""
    resolved_at: Optional[str] = None
    
    # Context
    game_context: Optional[Dict[str, Any]] = None
    weather_context: Optional[Dict[str, Any]] = None
    injury_context: Optional[Dict[str, Any]] = None
    
    # Quality scores
    accuracy_score: Optional[float] = None
    value_score: Optional[float] = None
    difficulty_score: Optional[float] = None

class AgentMonitor:
    """Comprehensive monitoring system for agent performance and analytics"""
    
    def __init__(self, data_dir: str = "monitoring_data"):
        """Initialize the agent monitor"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = get_logger()
        
        # In-memory caches for real-time monitoring
        self.performance_cache: Dict[str, PerformanceMetrics] = {}
        self.pick_analytics_cache: List[PickAnalytics] = []
        self.real_time_metrics: Dict[str, Any] = {}
        
        # Historical data storage
        self.metrics_file = self.data_dir / "performance_metrics.json"
        self.picks_file = self.data_dir / "pick_analytics.json"
        
        # Load existing data
        self._load_historical_data()
        
        # Initialize real-time tracking
        self._initialize_real_time_metrics()
    
    def _load_historical_data(self):
        """Load historical performance data"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    for key, metrics_data in data.items():
                        self.performance_cache[key] = PerformanceMetrics(**metrics_data)
            
            if self.picks_file.exists():
                with open(self.picks_file, 'r') as f:
                    data = json.load(f)
                    self.pick_analytics_cache = [PickAnalytics(**pick_data) for pick_data in data]
        
        except Exception as e:
            self.logger.logger.error(f"Failed to load historical data: {e}")
    
    def _save_performance_data(self):
        """Save performance data to disk"""
        try:
            data = {}
            for key, metrics in self.performance_cache.items():
                data[key] = asdict(metrics)
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.logger.error(f"Failed to save performance data: {e}")
    
    def _save_pick_analytics(self):
        """Save pick analytics to disk"""
        try:
            data = [asdict(pick) for pick in self.pick_analytics_cache]
            
            with open(self.picks_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.logger.error(f"Failed to save pick analytics: {e}")
    
    def _initialize_real_time_metrics(self):
        """Initialize real-time metrics tracking"""
        self.real_time_metrics = {
            'current_hour_requests': 0,
            'current_hour_picks': 0,
            'active_sessions': 0,
            'system_health': 'healthy',
            'total_agents': len(self.performance_cache),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def get_agent_key(self, agent_name: str, sport: str) -> str:
        """Generate key for agent metrics"""
        return f"{agent_name}:{sport}"
    
    def get_or_create_metrics(self, agent_name: str, sport: str) -> PerformanceMetrics:
        """Get or create performance metrics for an agent"""
        key = self.get_agent_key(agent_name, sport)
        
        if key not in self.performance_cache:
            self.performance_cache[key] = PerformanceMetrics(
                agent_name=agent_name,
                sport=sport
            )
        
        return self.performance_cache[key]
    
    def record_request(self, agent_name: str, sport: str, response_time_ms: float, 
                      success: bool = True, error_type: Optional[str] = None):
        """Record a request and its performance metrics"""
        metrics = self.get_or_create_metrics(agent_name, sport)
        
        # Update request counts
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
            if error_type:
                if error_type not in metrics.error_counts:
                    metrics.error_counts[error_type] = 0
                metrics.error_counts[error_type] += 1
        
        # Update response time metrics
        if success:
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = response_time_ms
            else:
                # Running average
                total_success = metrics.successful_requests
                metrics.avg_response_time = (
                    (metrics.avg_response_time * (total_success - 1)) + response_time_ms
                ) / total_success
            
            metrics.max_response_time = max(metrics.max_response_time, response_time_ms)
            metrics.min_response_time = min(metrics.min_response_time, response_time_ms)
        
        # Update derived metrics
        metrics.error_rate = metrics.failed_requests / metrics.total_requests if metrics.total_requests > 0 else 0
        metrics.last_active = datetime.now(timezone.utc).isoformat()
        
        # Update real-time metrics
        self.real_time_metrics['current_hour_requests'] += 1
        self.real_time_metrics['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Save periodically
        if metrics.total_requests % 10 == 0:
            self._save_performance_data()
    
    def record_pick_generated(self, agent_name: str, sport: str, pick_data: Dict[str, Any],
                             confidence: float, pick_id: str):
        """Record when a pick is generated"""
        metrics = self.get_or_create_metrics(agent_name, sport)
        
        # Update pick generation metrics
        metrics.picks_generated += 1
        
        # Update confidence metrics
        if metrics.avg_confidence == 0:
            metrics.avg_confidence = confidence
        else:
            metrics.avg_confidence = (
                (metrics.avg_confidence * (metrics.picks_generated - 1)) + confidence
            ) / metrics.picks_generated
        
        # Update confidence distribution
        confidence_bucket = self._get_confidence_bucket(confidence)
        if confidence_bucket not in metrics.confidence_distribution:
            metrics.confidence_distribution[confidence_bucket] = 0
        metrics.confidence_distribution[confidence_bucket] += 1
        
        # Create pick analytics entry
        pick_analytics = PickAnalytics(
            pick_id=pick_id,
            agent_name=agent_name,
            sport=sport,
            pick_type=pick_data.get('type', 'unknown'),
            player_name=pick_data.get('player', 'unknown'),
            stat_category=pick_data.get('stat', 'unknown'),
            line_value=pick_data.get('line', 0.0),
            prediction=pick_data.get('prediction', 'unknown'),
            confidence=confidence,
            odds=pick_data.get('odds'),
            expected_value=pick_data.get('expected_value'),
            kelly_size=pick_data.get('kelly_size'),
            generated_at=datetime.now(timezone.utc).isoformat(),
            game_context=pick_data.get('game_context'),
            weather_context=pick_data.get('weather_context'),
            injury_context=pick_data.get('injury_context')
        )
        
        self.pick_analytics_cache.append(pick_analytics)
        
        # Update real-time metrics
        self.real_time_metrics['current_hour_picks'] += 1
        
        # Calculate hourly rates
        self._update_hourly_rates(metrics)
        
        # Save periodically
        if len(self.pick_analytics_cache) % 10 == 0:
            self._save_pick_analytics()
    
    def record_pick_outcome(self, pick_id: str, outcome: str, actual_value: float,
                           profit_loss: float):
        """Record the outcome of a pick"""
        # Find the pick in analytics cache
        pick_analytics = None
        for pick in self.pick_analytics_cache:
            if pick.pick_id == pick_id:
                pick_analytics = pick
                break
        
        if not pick_analytics:
            self.logger.logger.warning(f"Pick analytics not found for pick_id: {pick_id}")
            return
        
        # Update pick analytics
        pick_analytics.outcome = outcome
        pick_analytics.actual_value = actual_value
        pick_analytics.profit_loss = profit_loss
        pick_analytics.resolved_at = datetime.now(timezone.utc).isoformat()
        
        # Calculate quality scores
        pick_analytics.accuracy_score = self._calculate_accuracy_score(pick_analytics)
        pick_analytics.value_score = self._calculate_value_score(pick_analytics)
        pick_analytics.difficulty_score = self._calculate_difficulty_score(pick_analytics)
        
        # Update agent metrics
        metrics = self.get_or_create_metrics(pick_analytics.agent_name, pick_analytics.sport)
        
        metrics.picks_resolved += 1
        
        if outcome == 'win':
            metrics.picks_won += 1
        elif outcome == 'loss':
            metrics.picks_lost += 1
        elif outcome == 'push':
            metrics.picks_pushed += 1
        
        # Update financial metrics
        metrics.total_profit += profit_loss
        if pick_analytics.kelly_size:
            metrics.total_volume += pick_analytics.kelly_size
        
        # Update win rate
        if metrics.picks_resolved > 0:
            metrics.win_rate = metrics.picks_won / metrics.picks_resolved
        
        # Update ROI
        if metrics.total_volume > 0:
            metrics.roi = metrics.total_profit / metrics.total_volume
        
        # Calculate advanced financial metrics
        self._update_financial_metrics(metrics, pick_analytics.agent_name, pick_analytics.sport)
        
        # Save data
        self._save_pick_analytics()
        self._save_performance_data()
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for distribution tracking"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.7:
            return "medium_high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.5:
            return "medium_low"
        else:
            return "low"
    
    def _update_hourly_rates(self, metrics: PerformanceMetrics):
        """Update hourly rate calculations"""
        # This is a simplified calculation - in practice you'd want to track time windows
        # For now, estimate based on recent activity
        now = datetime.now(timezone.utc)
        if metrics.last_active:
            last_active = datetime.fromisoformat(metrics.last_active.replace('Z', '+00:00'))
            hours_active = max(1, (now - last_active).total_seconds() / 3600)
            
            metrics.requests_per_hour = metrics.total_requests / hours_active
            metrics.picks_per_hour = metrics.picks_generated / hours_active
    
    def _calculate_accuracy_score(self, pick: PickAnalytics) -> float:
        """Calculate accuracy score for a pick"""
        if pick.outcome == 'push':
            return 0.5  # Neutral for pushes
        elif pick.outcome == 'win':
            # Score based on how close the prediction was
            if pick.actual_value is not None and pick.line_value is not None:
                margin = abs(pick.actual_value - pick.line_value)
                # Higher score for larger margins of victory
                return min(1.0, 0.5 + (margin / (2 * pick.line_value)) if pick.line_value > 0 else 0.5)
            return 1.0
        else:  # loss
            # Score based on how close we came
            if pick.actual_value is not None and pick.line_value is not None:
                margin = abs(pick.actual_value - pick.line_value)
                # Give some credit for close losses
                return max(0.0, 0.3 - (margin / (2 * pick.line_value)) if pick.line_value > 0 else 0.0)
            return 0.0
    
    def _calculate_value_score(self, pick: PickAnalytics) -> float:
        """Calculate value score based on expected value and outcome"""
        if pick.expected_value is None:
            return 0.5  # Neutral if no EV calculated
        
        # Base score on expected value
        ev_score = min(1.0, max(0.0, pick.expected_value + 0.5))  # Normalize around 0 EV
        
        # Adjust based on outcome
        if pick.outcome == 'win':
            return min(1.0, ev_score * 1.2)  # Boost for wins
        elif pick.outcome == 'loss':
            return max(0.0, ev_score * 0.8)  # Reduce for losses
        else:
            return ev_score
    
    def _calculate_difficulty_score(self, pick: PickAnalytics) -> float:
        """Calculate difficulty score for a pick"""
        # Base difficulty on confidence (lower confidence = higher difficulty)
        base_difficulty = 1.0 - pick.confidence
        
        # Adjust for pick type (some stats are harder to predict)
        type_modifiers = {
            'points': 1.0,
            'rebounds': 1.1,
            'assists': 1.2,
            'steals': 1.5,
            'blocks': 1.5,
            'turnovers': 1.3,
            'threes': 1.4
        }
        
        type_modifier = type_modifiers.get(pick.stat_category.lower(), 1.0)
        
        return min(1.0, base_difficulty * type_modifier)
    
    def _update_financial_metrics(self, metrics: PerformanceMetrics, agent_name: str, sport: str):
        """Update advanced financial metrics"""
        # Get recent picks for this agent
        agent_picks = [
            pick for pick in self.pick_analytics_cache
            if pick.agent_name == agent_name and pick.sport == sport
            and pick.outcome in ['win', 'loss'] and pick.profit_loss is not None
        ]
        
        if len(agent_picks) < 10:
            return  # Need enough data for meaningful metrics
        
        # Calculate Sharpe ratio
        returns = [pick.profit_loss for pick in agent_picks[-50:]]  # Last 50 picks
        if len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            if std_return > 0:
                metrics.sharpe_ratio = avg_return / std_return
        
        # Calculate max drawdown
        cumulative_returns = []
        cumulative = 0
        for ret in returns:
            cumulative += ret
            cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            peak = cumulative_returns[0]
            max_dd = 0
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                drawdown = peak - value
                if drawdown > max_dd:
                    max_dd = drawdown
            metrics.max_drawdown = max_dd
    
    def get_agent_performance(self, agent_name: str = None, sport: str = None) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for agents"""
        if agent_name and sport:
            key = self.get_agent_key(agent_name, sport)
            return {key: self.performance_cache.get(key)} if key in self.performance_cache else {}
        
        # Filter by criteria
        filtered_metrics = {}
        for key, metrics in self.performance_cache.items():
            if agent_name and metrics.agent_name != agent_name:
                continue
            if sport and metrics.sport != sport:
                continue
            filtered_metrics[key] = metrics
        
        return filtered_metrics
    
    def get_pick_analytics(self, 
                          agent_name: str = None,
                          sport: str = None,
                          days: int = 7,
                          outcome: str = None) -> List[PickAnalytics]:
        """Get pick analytics with filtering"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        filtered_picks = []
        for pick in self.pick_analytics_cache:
            # Date filter
            generated_time = datetime.fromisoformat(pick.generated_at.replace('Z', '+00:00'))
            if generated_time < cutoff_date:
                continue
            
            # Agent filter
            if agent_name and pick.agent_name != agent_name:
                continue
            
            # Sport filter
            if sport and pick.sport != sport:
                continue
            
            # Outcome filter
            if outcome and pick.outcome != outcome:
                continue
            
            filtered_picks.append(pick)
        
        return filtered_picks
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        total_agents = len(self.performance_cache)
        healthy_agents = 0
        error_agents = 0
        
        for metrics in self.performance_cache.values():
            if metrics.error_rate < 0.1:  # Less than 10% error rate
                healthy_agents += 1
            elif metrics.error_rate > 0.3:  # More than 30% error rate
                error_agents += 1
        
        system_health = "healthy"
        if error_agents > total_agents * 0.3:
            system_health = "critical"
        elif error_agents > total_agents * 0.1:
            system_health = "warning"
        
        return {
            'status': system_health,
            'total_agents': total_agents,
            'healthy_agents': healthy_agents,
            'warning_agents': total_agents - healthy_agents - error_agents,
            'error_agents': error_agents,
            'uptime_percentage': sum(m.uptime_percentage for m in self.performance_cache.values()) / total_agents if total_agents > 0 else 100,
            'avg_response_time': sum(m.avg_response_time for m in self.performance_cache.values()) / total_agents if total_agents > 0 else 0,
            'total_requests_hour': self.real_time_metrics.get('current_hour_requests', 0),
            'total_picks_hour': self.real_time_metrics.get('current_hour_picks', 0),
            'last_updated': self.real_time_metrics.get('last_updated')
        }
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'report_period': f"{days} days",
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'system_health': self.get_system_health(),
            'agent_performance': {},
            'pick_analytics': {},
            'financial_summary': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Agent performance summary
        for key, metrics in self.performance_cache.items():
            agent_name, sport = key.split(':', 1)
            
            report['agent_performance'][key] = {
                'requests': {
                    'total': metrics.total_requests,
                    'success_rate': (metrics.successful_requests / metrics.total_requests) if metrics.total_requests > 0 else 0,
                    'avg_response_time': metrics.avg_response_time,
                    'requests_per_hour': metrics.requests_per_hour
                },
                'picks': {
                    'generated': metrics.picks_generated,
                    'resolved': metrics.picks_resolved,
                    'win_rate': metrics.win_rate,
                    'avg_confidence': metrics.avg_confidence,
                    'picks_per_hour': metrics.picks_per_hour
                },
                'financial': {
                    'total_profit': metrics.total_profit,
                    'roi': metrics.roi,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown
                }
            }
        
        # Pick analytics summary
        recent_picks = self.get_pick_analytics(days=days)
        total_picks = len(recent_picks)
        won_picks = len([p for p in recent_picks if p.outcome == 'win'])
        
        report['pick_analytics'] = {
            'total_picks': total_picks,
            'win_rate': won_picks / total_picks if total_picks > 0 else 0,
            'avg_confidence': sum(p.confidence for p in recent_picks) / total_picks if total_picks > 0 else 0,
            'avg_accuracy_score': sum(p.accuracy_score or 0 for p in recent_picks) / total_picks if total_picks > 0 else 0,
            'avg_value_score': sum(p.value_score or 0 for p in recent_picks) / total_picks if total_picks > 0 else 0,
            'sport_breakdown': self._get_sport_breakdown(recent_picks)
        }
        
        # Financial summary
        total_profit = sum(p.profit_loss or 0 for p in recent_picks)
        total_volume = sum(p.kelly_size or 0 for p in recent_picks)
        
        report['financial_summary'] = {
            'total_profit': total_profit,
            'total_volume': total_volume,
            'roi': total_profit / total_volume if total_volume > 0 else 0,
            'profit_by_sport': self._get_profit_by_sport(recent_picks)
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(recent_picks)
        
        return report
    
    def _get_sport_breakdown(self, picks: List[PickAnalytics]) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by sport"""
        sport_stats = {}
        
        for pick in picks:
            if pick.sport not in sport_stats:
                sport_stats[pick.sport] = {
                    'total': 0,
                    'won': 0,
                    'profit': 0.0
                }
            
            sport_stats[pick.sport]['total'] += 1
            if pick.outcome == 'win':
                sport_stats[pick.sport]['won'] += 1
            if pick.profit_loss:
                sport_stats[pick.sport]['profit'] += pick.profit_loss
        
        # Calculate win rates
        for sport, stats in sport_stats.items():
            stats['win_rate'] = stats['won'] / stats['total'] if stats['total'] > 0 else 0
        
        return sport_stats
    
    def _get_profit_by_sport(self, picks: List[PickAnalytics]) -> Dict[str, float]:
        """Get profit breakdown by sport"""
        profit_by_sport = {}
        
        for pick in picks:
            if pick.sport not in profit_by_sport:
                profit_by_sport[pick.sport] = 0.0
            
            if pick.profit_loss:
                profit_by_sport[pick.sport] += pick.profit_loss
        
        return profit_by_sport
    
    def _generate_recommendations(self, picks: List[PickAnalytics]) -> List[str]:
        """Generate actionable recommendations based on performance"""
        recommendations = []
        
        # Analyze performance by sport
        sport_breakdown = self._get_sport_breakdown(picks)
        
        for sport, stats in sport_breakdown.items():
            if stats['total'] >= 10:  # Enough data for meaningful analysis
                if stats['win_rate'] < 0.45:
                    recommendations.append(
                        f"Consider reviewing {sport} agent strategy - win rate is {stats['win_rate']:.1%}"
                    )
                elif stats['win_rate'] > 0.65:
                    recommendations.append(
                        f"{sport} agent performing excellently - consider increasing bet sizes"
                    )
        
        # Analyze confidence vs outcomes
        high_conf_picks = [p for p in picks if p.confidence > 0.8]
        if len(high_conf_picks) >= 5:
            high_conf_win_rate = len([p for p in high_conf_picks if p.outcome == 'win']) / len(high_conf_picks)
            if high_conf_win_rate < 0.6:
                recommendations.append(
                    "High confidence picks underperforming - review confidence calibration"
                )
        
        # Analyze error rates
        for key, metrics in self.performance_cache.items():
            if metrics.error_rate > 0.2:
                recommendations.append(
                    f"High error rate for {key} - investigate API reliability"
                )
        
        return recommendations
    
    def export_analytics(self, output_file: str, format: str = 'json') -> str:
        """Export analytics data to file"""
        try:
            report = self.generate_performance_report(days=30)  # 30-day report
            
            output_path = Path(output_file)
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Create DataFrames for different aspects
                picks_df = pd.DataFrame([asdict(pick) for pick in self.pick_analytics_cache])
                metrics_df = pd.DataFrame([asdict(metrics) for metrics in self.performance_cache.values()])
                
                # Save to CSV
                base_path = output_path.stem
                picks_df.to_csv(f"{base_path}_picks.csv", index=False)
                metrics_df.to_csv(f"{base_path}_metrics.csv", index=False)
                
                # Save summary report as JSON
                with open(f"{base_path}_report.json", 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            return str(output_path.absolute())
        
        except Exception as e:
            self.logger.logger.error(f"Failed to export analytics: {e}")
            raise

# Global monitor instance
_global_monitor: Optional[AgentMonitor] = None

def get_monitor() -> AgentMonitor:
    """Get the global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AgentMonitor()
    return _global_monitor

def init_monitor(data_dir: str = "monitoring_data") -> AgentMonitor:
    """Initialize the global monitor with custom settings"""
    global _global_monitor
    _global_monitor = AgentMonitor(data_dir=data_dir)
    return _global_monitor