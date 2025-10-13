"""
Agent Analytics Tracker - Advanced analytics for pick quality and ML feedback
Tracks detailed performance metrics, identifies patterns, and provides insights for prompt optimization
"""

import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import statistics

from agent_logger import get_logger
from agent_monitor import get_monitor, PickAnalytics

class AnalysisType(Enum):
    """Types of analytics to perform"""
    PATTERN_DETECTION = "pattern_detection"
    QUALITY_ASSESSMENT = "quality_assessment"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    BIAS_DETECTION = "bias_detection"
    PREDICTION_ACCURACY = "prediction_accuracy"
    MARKET_TIMING = "market_timing"
    CONTEXTUAL_ANALYSIS = "contextual_analysis"

@dataclass
class PatternInsight:
    """Insights from pattern analysis"""
    pattern_type: str
    description: str
    confidence: float
    affected_picks: List[str]
    recommendation: str
    impact_score: float
    
@dataclass
class QualityMetrics:
    """Detailed quality metrics for an agent"""
    agent_name: str
    sport: str
    
    # Accuracy metrics
    overall_accuracy: float = 0.0
    confidence_calibration: float = 0.0
    prediction_bias: float = 0.0
    
    # Consistency metrics
    performance_volatility: float = 0.0
    streak_analysis: Dict[str, int] = None
    
    # Contextual performance
    performance_by_context: Dict[str, float] = None
    performance_by_difficulty: Dict[str, float] = None
    
    # Market timing
    early_pick_accuracy: float = 0.0
    late_pick_accuracy: float = 0.0
    
    # Value generation
    value_creation_rate: float = 0.0
    false_confidence_rate: float = 0.0
    
    # Learning indicators
    improvement_trend: float = 0.0
    adaptation_speed: float = 0.0
    
    def __post_init__(self):
        if self.streak_analysis is None:
            self.streak_analysis = {}
        if self.performance_by_context is None:
            self.performance_by_context = {}
        if self.performance_by_difficulty is None:
            self.performance_by_difficulty = {}

@dataclass
class PromptAnalysis:
    """Analysis for prompt optimization"""
    agent_name: str
    sport: str
    
    # Current prompt performance
    current_prompt_version: str = "1.0"
    prompt_performance_score: float = 0.0
    
    # Areas for improvement
    weak_areas: List[str] = None
    strong_areas: List[str] = None
    
    # Specific recommendations
    recommendations: List[str] = None
    priority_fixes: List[str] = None
    
    # A/B testing results
    ab_test_results: Dict[str, float] = None
    
    # Contextual improvements
    context_specific_prompts: Dict[str, str] = None
    
    def __post_init__(self):
        if self.weak_areas is None:
            self.weak_areas = []
        if self.strong_areas is None:
            self.strong_areas = []
        if self.recommendations is None:
            self.recommendations = []
        if self.priority_fixes is None:
            self.priority_fixes = []
        if self.ab_test_results is None:
            self.ab_test_results = {}
        if self.context_specific_prompts is None:
            self.context_specific_prompts = {}

class AgentAnalyticsTracker:
    """Advanced analytics tracker for agent performance and optimization"""
    
    def __init__(self, data_dir: str = "analytics_data", analysis_window: int = 30):
        """Initialize the analytics tracker"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.analysis_window = analysis_window  # Days to analyze
        self.logger = get_logger()
        self.monitor = get_monitor()
        
        # Analytics storage
        self.quality_metrics: Dict[str, QualityMetrics] = {}
        self.pattern_insights: List[PatternInsight] = []
        self.prompt_analyses: Dict[str, PromptAnalysis] = {}
        
        # Analysis caches
        self.rolling_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.contextual_cache: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
        
        # Load existing analytics
        self._load_analytics_data()
        
        # Initialize analysis components
        self._initialize_analyzers()
    
    def _load_analytics_data(self):
        """Load existing analytics data"""
        try:
            quality_file = self.data_dir / "quality_metrics.json"
            if quality_file.exists():
                with open(quality_file, 'r') as f:
                    data = json.load(f)
                    for key, metrics_data in data.items():
                        self.quality_metrics[key] = QualityMetrics(**metrics_data)
            
            insights_file = self.data_dir / "pattern_insights.json"
            if insights_file.exists():
                with open(insights_file, 'r') as f:
                    data = json.load(f)
                    self.pattern_insights = [PatternInsight(**insight) for insight in data]
            
            prompts_file = self.data_dir / "prompt_analyses.json"
            if prompts_file.exists():
                with open(prompts_file, 'r') as f:
                    data = json.load(f)
                    for key, analysis_data in data.items():
                        self.prompt_analyses[key] = PromptAnalysis(**analysis_data)
        
        except Exception as e:
            self.logger.logger.error(f"Failed to load analytics data: {e}")
    
    def _save_analytics_data(self):
        """Save analytics data to disk"""
        try:
            # Save quality metrics
            quality_data = {}
            for key, metrics in self.quality_metrics.items():
                quality_data[key] = asdict(metrics)
            
            with open(self.data_dir / "quality_metrics.json", 'w') as f:
                json.dump(quality_data, f, indent=2, default=str)
            
            # Save pattern insights
            insights_data = [asdict(insight) for insight in self.pattern_insights]
            with open(self.data_dir / "pattern_insights.json", 'w') as f:
                json.dump(insights_data, f, indent=2, default=str)
            
            # Save prompt analyses
            prompts_data = {}
            for key, analysis in self.prompt_analyses.items():
                prompts_data[key] = asdict(analysis)
            
            with open(self.data_dir / "prompt_analyses.json", 'w') as f:
                json.dump(prompts_data, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.logger.error(f"Failed to save analytics data: {e}")
    
    def _initialize_analyzers(self):
        """Initialize analysis components"""
        self.pattern_detectors = {
            'win_streak': self._detect_win_streaks,
            'loss_streak': self._detect_loss_streaks,
            'confidence_drift': self._detect_confidence_drift,
            'performance_degradation': self._detect_performance_degradation,
            'context_bias': self._detect_context_bias,
            'timing_patterns': self._detect_timing_patterns,
            'value_patterns': self._detect_value_patterns
        }
    
    def get_agent_key(self, agent_name: str, sport: str) -> str:
        """Generate key for agent analytics"""
        return f"{agent_name}:{sport}"
    
    def analyze_agent_quality(self, agent_name: str, sport: str, 
                            days: int = None) -> QualityMetrics:
        """Perform comprehensive quality analysis for an agent"""
        if days is None:
            days = self.analysis_window
        
        key = self.get_agent_key(agent_name, sport)
        
        # Get recent picks for analysis
        picks = self.monitor.get_pick_analytics(
            agent_name=agent_name,
            sport=sport,
            days=days
        )
        
        if not picks:
            return QualityMetrics(agent_name=agent_name, sport=sport)
        
        # Calculate quality metrics
        quality_metrics = QualityMetrics(agent_name=agent_name, sport=sport)
        
        # Overall accuracy
        resolved_picks = [p for p in picks if p.outcome in ['win', 'loss']]
        if resolved_picks:
            wins = len([p for p in resolved_picks if p.outcome == 'win'])
            quality_metrics.overall_accuracy = wins / len(resolved_picks)
        
        # Confidence calibration
        quality_metrics.confidence_calibration = self._calculate_confidence_calibration(picks)
        
        # Prediction bias
        quality_metrics.prediction_bias = self._calculate_prediction_bias(picks)
        
        # Performance volatility
        quality_metrics.performance_volatility = self._calculate_volatility(picks)
        
        # Streak analysis
        quality_metrics.streak_analysis = self._analyze_streaks(picks)
        
        # Contextual performance
        quality_metrics.performance_by_context = self._analyze_contextual_performance(picks)
        quality_metrics.performance_by_difficulty = self._analyze_difficulty_performance(picks)
        
        # Market timing
        early_picks, late_picks = self._split_by_timing(picks)
        quality_metrics.early_pick_accuracy = self._calculate_accuracy(early_picks)
        quality_metrics.late_pick_accuracy = self._calculate_accuracy(late_picks)
        
        # Value metrics
        quality_metrics.value_creation_rate = self._calculate_value_creation_rate(picks)
        quality_metrics.false_confidence_rate = self._calculate_false_confidence_rate(picks)
        
        # Learning indicators
        quality_metrics.improvement_trend = self._calculate_improvement_trend(picks)
        quality_metrics.adaptation_speed = self._calculate_adaptation_speed(picks)
        
        # Store and save
        self.quality_metrics[key] = quality_metrics
        self._save_analytics_data()
        
        return quality_metrics
    
    def _calculate_confidence_calibration(self, picks: List[PickAnalytics]) -> float:
        """Calculate how well confidence scores match actual outcomes"""
        if not picks:
            return 0.0
        
        # Group picks by confidence buckets
        buckets = defaultdict(list)
        for pick in picks:
            if pick.outcome in ['win', 'loss']:
                bucket = int(pick.confidence * 10) / 10  # 0.1 buckets
                buckets[bucket].append(pick.outcome == 'win')
        
        # Calculate calibration error
        calibration_error = 0.0
        total_picks = 0
        
        for confidence, outcomes in buckets.items():
            if len(outcomes) >= 3:  # Need minimum picks for reliable measurement
                actual_rate = sum(outcomes) / len(outcomes)
                expected_rate = confidence
                error = abs(actual_rate - expected_rate)
                calibration_error += error * len(outcomes)
                total_picks += len(outcomes)
        
        if total_picks > 0:
            avg_error = calibration_error / total_picks
            return 1.0 - avg_error  # Convert to calibration score
        
        return 0.5  # Neutral if not enough data
    
    def _calculate_prediction_bias(self, picks: List[PickAnalytics]) -> float:
        """Calculate bias in predictions (tendency to pick over/under)"""
        if not picks:
            return 0.0
        
        over_predictions = len([p for p in picks if p.prediction.lower() == 'over'])
        under_predictions = len([p for p in picks if p.prediction.lower() == 'under'])
        total = over_predictions + under_predictions
        
        if total == 0:
            return 0.0
        
        # Calculate bias (-1 = all under, +1 = all over, 0 = balanced)
        bias = (over_predictions - under_predictions) / total
        return bias
    
    def _calculate_volatility(self, picks: List[PickAnalytics]) -> float:
        """Calculate performance volatility over time"""
        if len(picks) < 10:
            return 0.0
        
        # Sort picks by time
        sorted_picks = sorted(picks, key=lambda p: p.generated_at)
        
        # Calculate rolling win rates
        window_size = max(5, len(picks) // 10)
        win_rates = []
        
        for i in range(len(sorted_picks) - window_size + 1):
            window = sorted_picks[i:i + window_size]
            resolved = [p for p in window if p.outcome in ['win', 'loss']]
            if resolved:
                wins = len([p for p in resolved if p.outcome == 'win'])
                win_rates.append(wins / len(resolved))
        
        if len(win_rates) > 1:
            return statistics.stdev(win_rates)
        
        return 0.0
    
    def _analyze_streaks(self, picks: List[PickAnalytics]) -> Dict[str, int]:
        """Analyze winning and losing streaks"""
        if not picks:
            return {'longest_win_streak': 0, 'longest_loss_streak': 0, 'current_streak': 0}
        
        # Sort picks by time
        sorted_picks = sorted(picks, key=lambda p: p.generated_at)
        resolved_picks = [p for p in sorted_picks if p.outcome in ['win', 'loss']]
        
        if not resolved_picks:
            return {'longest_win_streak': 0, 'longest_loss_streak': 0, 'current_streak': 0}
        
        longest_win = 0
        longest_loss = 0
        current_win = 0
        current_loss = 0
        
        for pick in resolved_picks:
            if pick.outcome == 'win':
                current_win += 1
                current_loss = 0
                longest_win = max(longest_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                longest_loss = max(longest_loss, current_loss)
        
        # Current streak
        last_pick = resolved_picks[-1]
        current_streak = current_win if last_pick.outcome == 'win' else -current_loss
        
        return {
            'longest_win_streak': longest_win,
            'longest_loss_streak': longest_loss,
            'current_streak': current_streak
        }
    
    def _analyze_contextual_performance(self, picks: List[PickAnalytics]) -> Dict[str, float]:
        """Analyze performance in different contexts"""
        context_performance = {}
        
        # Group by various contexts
        contexts = {
            'high_confidence': [p for p in picks if p.confidence > 0.8],
            'medium_confidence': [p for p in picks if 0.6 <= p.confidence <= 0.8],
            'low_confidence': [p for p in picks if p.confidence < 0.6],
            'points_picks': [p for p in picks if 'point' in p.stat_category.lower()],
            'rebounds_picks': [p for p in picks if 'rebound' in p.stat_category.lower()],
            'assists_picks': [p for p in picks if 'assist' in p.stat_category.lower()],
        }
        
        for context_name, context_picks in contexts.items():
            context_performance[context_name] = self._calculate_accuracy(context_picks)
        
        return context_performance
    
    def _analyze_difficulty_performance(self, picks: List[PickAnalytics]) -> Dict[str, float]:
        """Analyze performance by pick difficulty"""
        if not picks:
            return {}
        
        # Calculate difficulty scores for all picks
        difficulties = []
        for pick in picks:
            if hasattr(pick, 'difficulty_score') and pick.difficulty_score is not None:
                difficulties.append(pick.difficulty_score)
        
        if not difficulties:
            return {}
        
        # Create difficulty buckets
        difficulty_buckets = {
            'easy': [p for p in picks if hasattr(p, 'difficulty_score') and 
                    p.difficulty_score is not None and p.difficulty_score < 0.3],
            'medium': [p for p in picks if hasattr(p, 'difficulty_score') and 
                      p.difficulty_score is not None and 0.3 <= p.difficulty_score <= 0.7],
            'hard': [p for p in picks if hasattr(p, 'difficulty_score') and 
                    p.difficulty_score is not None and p.difficulty_score > 0.7],
        }
        
        return {
            bucket: self._calculate_accuracy(picks_list)
            for bucket, picks_list in difficulty_buckets.items()
        }
    
    def _split_by_timing(self, picks: List[PickAnalytics]) -> Tuple[List[PickAnalytics], List[PickAnalytics]]:
        """Split picks by timing (early vs late)"""
        if not picks:
            return [], []
        
        # Sort by generation time
        sorted_picks = sorted(picks, key=lambda p: p.generated_at)
        midpoint = len(sorted_picks) // 2
        
        early_picks = sorted_picks[:midpoint]
        late_picks = sorted_picks[midpoint:]
        
        return early_picks, late_picks
    
    def _calculate_accuracy(self, picks: List[PickAnalytics]) -> float:
        """Calculate accuracy for a list of picks"""
        if not picks:
            return 0.0
        
        resolved_picks = [p for p in picks if p.outcome in ['win', 'loss']]
        if not resolved_picks:
            return 0.0
        
        wins = len([p for p in resolved_picks if p.outcome == 'win'])
        return wins / len(resolved_picks)
    
    def _calculate_value_creation_rate(self, picks: List[PickAnalytics]) -> float:
        """Calculate rate of positive expected value picks"""
        if not picks:
            return 0.0
        
        ev_picks = [p for p in picks if p.expected_value is not None]
        if not ev_picks:
            return 0.0
        
        positive_ev = len([p for p in ev_picks if p.expected_value > 0])
        return positive_ev / len(ev_picks)
    
    def _calculate_false_confidence_rate(self, picks: List[PickAnalytics]) -> float:
        """Calculate rate of high confidence picks that lose"""
        high_conf_picks = [p for p in picks if p.confidence > 0.8 and p.outcome in ['win', 'loss']]
        if not high_conf_picks:
            return 0.0
        
        losses = len([p for p in high_conf_picks if p.outcome == 'loss'])
        return losses / len(high_conf_picks)
    
    def _calculate_improvement_trend(self, picks: List[PickAnalytics]) -> float:
        """Calculate improvement trend over time"""
        if len(picks) < 20:
            return 0.0
        
        # Sort picks by time
        sorted_picks = sorted(picks, key=lambda p: p.generated_at)
        resolved_picks = [p for p in sorted_picks if p.outcome in ['win', 'loss']]
        
        if len(resolved_picks) < 20:
            return 0.0
        
        # Split into early and late periods
        midpoint = len(resolved_picks) // 2
        early_picks = resolved_picks[:midpoint]
        late_picks = resolved_picks[midpoint:]
        
        early_accuracy = self._calculate_accuracy(early_picks)
        late_accuracy = self._calculate_accuracy(late_picks)
        
        return late_accuracy - early_accuracy
    
    def _calculate_adaptation_speed(self, picks: List[PickAnalytics]) -> float:
        """Calculate how quickly agent adapts to new contexts"""
        # This is a simplified metric - in practice, you'd want more sophisticated analysis
        # For now, calculate variance in performance across different time windows
        
        if len(picks) < 30:
            return 0.0
        
        sorted_picks = sorted(picks, key=lambda p: p.generated_at)
        resolved_picks = [p for p in sorted_picks if p.outcome in ['win', 'loss']]
        
        if len(resolved_picks) < 30:
            return 0.0
        
        # Calculate performance in rolling windows
        window_size = 10
        performances = []
        
        for i in range(len(resolved_picks) - window_size + 1):
            window = resolved_picks[i:i + window_size]
            accuracy = self._calculate_accuracy(window)
            performances.append(accuracy)
        
        if len(performances) > 1:
            # Lower variance suggests more consistent adaptation
            variance = statistics.variance(performances)
            return 1.0 - min(1.0, variance * 4)  # Scale to 0-1
        
        return 0.5
    
    def detect_patterns(self, agent_name: str = None, sport: str = None) -> List[PatternInsight]:
        """Detect patterns in agent performance"""
        new_insights = []
        
        # Get agents to analyze
        if agent_name and sport:
            agents_to_analyze = [(agent_name, sport)]
        else:
            agents_to_analyze = []
            for key in self.quality_metrics.keys():
                name, sport_key = key.split(':', 1)
                if (not agent_name or name == agent_name) and (not sport or sport_key == sport):
                    agents_to_analyze.append((name, sport_key))
        
        # Run pattern detection for each agent
        for name, sport_key in agents_to_analyze:
            picks = self.monitor.get_pick_analytics(
                agent_name=name,
                sport=sport_key,
                days=self.analysis_window
            )
            
            if len(picks) < 10:
                continue
            
            # Run all pattern detectors
            for detector_name, detector_func in self.pattern_detectors.items():
                try:
                    insight = detector_func(name, sport_key, picks)
                    if insight:
                        new_insights.append(insight)
                except Exception as e:
                    self.logger.logger.error(f"Pattern detection failed for {detector_name}: {e}")
        
        # Add new insights to collection
        self.pattern_insights.extend(new_insights)
        
        # Keep only recent insights (last 1000)
        if len(self.pattern_insights) > 1000:
            self.pattern_insights = self.pattern_insights[-500:]
        
        self._save_analytics_data()
        
        return new_insights
    
    def _detect_win_streaks(self, agent_name: str, sport: str, picks: List[PickAnalytics]) -> Optional[PatternInsight]:
        """Detect notable winning streaks"""
        quality_metrics = self.quality_metrics.get(f"{agent_name}:{sport}")
        if not quality_metrics:
            return None
        
        longest_streak = quality_metrics.streak_analysis.get('longest_win_streak', 0)
        
        if longest_streak >= 7:  # Notable streak
            streak_picks = []
            # Find the picks in the streak (simplified)
            sorted_picks = sorted([p for p in picks if p.outcome == 'win'], key=lambda p: p.generated_at)
            if len(sorted_picks) >= longest_streak:
                streak_picks = [p.pick_id for p in sorted_picks[-longest_streak:]]
            
            return PatternInsight(
                pattern_type="win_streak",
                description=f"Agent achieved {longest_streak}-game winning streak",
                confidence=0.9,
                affected_picks=streak_picks,
                recommendation="Analyze successful patterns and replicate approach",
                impact_score=longest_streak / 10.0
            )
        
        return None
    
    def _detect_loss_streaks(self, agent_name: str, sport: str, picks: List[PickAnalytics]) -> Optional[PatternInsight]:
        """Detect concerning losing streaks"""
        quality_metrics = self.quality_metrics.get(f"{agent_name}:{sport}")
        if not quality_metrics:
            return None
        
        longest_streak = quality_metrics.streak_analysis.get('longest_loss_streak', 0)
        
        if longest_streak >= 5:  # Concerning streak
            return PatternInsight(
                pattern_type="loss_streak",
                description=f"Agent experienced {longest_streak}-game losing streak",
                confidence=0.85,
                affected_picks=[],
                recommendation="Review strategy and consider prompt adjustments",
                impact_score=-longest_streak / 5.0
            )
        
        return None
    
    def _detect_confidence_drift(self, agent_name: str, sport: str, picks: List[PickAnalytics]) -> Optional[PatternInsight]:
        """Detect drift in confidence calibration"""
        if len(picks) < 20:
            return None
        
        # Split picks into early and late periods
        sorted_picks = sorted(picks, key=lambda p: p.generated_at)
        midpoint = len(sorted_picks) // 2
        early_picks = sorted_picks[:midpoint]
        late_picks = sorted_picks[midpoint:]
        
        early_calibration = self._calculate_confidence_calibration(early_picks)
        late_calibration = self._calculate_confidence_calibration(late_picks)
        
        drift = abs(early_calibration - late_calibration)
        
        if drift > 0.2:  # Significant drift
            return PatternInsight(
                pattern_type="confidence_drift",
                description=f"Confidence calibration drifted by {drift:.2f}",
                confidence=0.75,
                affected_picks=[p.pick_id for p in late_picks],
                recommendation="Recalibrate confidence scoring mechanism",
                impact_score=-drift
            )
        
        return None
    
    def _detect_performance_degradation(self, agent_name: str, sport: str, picks: List[PickAnalytics]) -> Optional[PatternInsight]:
        """Detect performance degradation over time"""
        quality_metrics = self.quality_metrics.get(f"{agent_name}:{sport}")
        if not quality_metrics:
            return None
        
        improvement_trend = quality_metrics.improvement_trend
        
        if improvement_trend < -0.15:  # Significant degradation
            return PatternInsight(
                pattern_type="performance_degradation",
                description=f"Performance declined by {abs(improvement_trend):.1%}",
                confidence=0.8,
                affected_picks=[],
                recommendation="Investigate recent changes and consider model refresh",
                impact_score=improvement_trend
            )
        
        return None
    
    def _detect_context_bias(self, agent_name: str, sport: str, picks: List[PickAnalytics]) -> Optional[PatternInsight]:
        """Detect bias in specific contexts"""
        quality_metrics = self.quality_metrics.get(f"{agent_name}:{sport}")
        if not quality_metrics:
            return None
        
        context_performance = quality_metrics.performance_by_context
        
        # Check for significant performance gaps
        high_conf = context_performance.get('high_confidence', 0.5)
        low_conf = context_performance.get('low_confidence', 0.5)
        
        if high_conf < low_conf - 0.2:  # High confidence worse than low confidence
            return PatternInsight(
                pattern_type="context_bias",
                description="High confidence picks underperforming low confidence picks",
                confidence=0.7,
                affected_picks=[],
                recommendation="Review confidence scoring and decision criteria",
                impact_score=-(high_conf - low_conf)
            )
        
        return None
    
    def _detect_timing_patterns(self, agent_name: str, sport: str, picks: List[PickAnalytics]) -> Optional[PatternInsight]:
        """Detect patterns related to timing of picks"""
        quality_metrics = self.quality_metrics.get(f"{agent_name}:{sport}")
        if not quality_metrics:
            return None
        
        early_accuracy = quality_metrics.early_pick_accuracy
        late_accuracy = quality_metrics.late_pick_accuracy
        
        timing_diff = abs(early_accuracy - late_accuracy)
        
        if timing_diff > 0.2:  # Significant timing effect
            better_timing = "early" if early_accuracy > late_accuracy else "late"
            
            return PatternInsight(
                pattern_type="timing_patterns",
                description=f"Performance {timing_diff:.1%} better in {better_timing} period",
                confidence=0.6,
                affected_picks=[],
                recommendation=f"Focus more resources on {better_timing} period analysis",
                impact_score=timing_diff if better_timing == "late" else -timing_diff
            )
        
        return None
    
    def _detect_value_patterns(self, agent_name: str, sport: str, picks: List[PickAnalytics]) -> Optional[PatternInsight]:
        """Detect patterns in value generation"""
        quality_metrics = self.quality_metrics.get(f"{agent_name}:{sport}")
        if not quality_metrics:
            return None
        
        value_creation_rate = quality_metrics.value_creation_rate
        false_confidence_rate = quality_metrics.false_confidence_rate
        
        if value_creation_rate < 0.3:  # Low value creation
            return PatternInsight(
                pattern_type="value_patterns",
                description=f"Low value creation rate: {value_creation_rate:.1%}",
                confidence=0.7,
                affected_picks=[],
                recommendation="Improve expected value calculations and market analysis",
                impact_score=-value_creation_rate
            )
        
        if false_confidence_rate > 0.4:  # High false confidence
            return PatternInsight(
                pattern_type="value_patterns",
                description=f"High false confidence rate: {false_confidence_rate:.1%}",
                confidence=0.75,
                affected_picks=[],
                recommendation="Recalibrate confidence thresholds and risk assessment",
                impact_score=-false_confidence_rate
            )
        
        return None
    
    def analyze_prompt_optimization(self, agent_name: str, sport: str) -> PromptAnalysis:
        """Analyze current prompt performance and suggest optimizations"""
        key = self.get_agent_key(agent_name, sport)
        
        # Get current quality metrics
        quality_metrics = self.quality_metrics.get(key)
        if not quality_metrics:
            quality_metrics = self.analyze_agent_quality(agent_name, sport)
        
        # Get recent patterns
        recent_patterns = [
            insight for insight in self.pattern_insights[-50:]
            if insight.pattern_type in ['pattern_detection', 'quality_assessment']
        ]
        
        # Create prompt analysis
        prompt_analysis = PromptAnalysis(agent_name=agent_name, sport=sport)
        
        # Calculate prompt performance score
        prompt_analysis.prompt_performance_score = self._calculate_prompt_score(quality_metrics)
        
        # Identify weak areas
        prompt_analysis.weak_areas = self._identify_weak_areas(quality_metrics, recent_patterns)
        
        # Identify strong areas
        prompt_analysis.strong_areas = self._identify_strong_areas(quality_metrics)
        
        # Generate recommendations
        prompt_analysis.recommendations = self._generate_prompt_recommendations(
            quality_metrics, recent_patterns
        )
        
        # Identify priority fixes
        prompt_analysis.priority_fixes = self._identify_priority_fixes(
            quality_metrics, recent_patterns
        )
        
        # Store and save
        self.prompt_analyses[key] = prompt_analysis
        self._save_analytics_data()
        
        return prompt_analysis
    
    def _calculate_prompt_score(self, quality_metrics: QualityMetrics) -> float:
        """Calculate overall prompt performance score"""
        scores = []
        
        # Accuracy component (40%)
        scores.append(quality_metrics.overall_accuracy * 0.4)
        
        # Calibration component (20%)
        scores.append(quality_metrics.confidence_calibration * 0.2)
        
        # Consistency component (20%)
        volatility_score = max(0, 1.0 - quality_metrics.performance_volatility * 2)
        scores.append(volatility_score * 0.2)
        
        # Value creation component (20%)
        scores.append(quality_metrics.value_creation_rate * 0.2)
        
        return sum(scores)
    
    def _identify_weak_areas(self, quality_metrics: QualityMetrics, 
                           patterns: List[PatternInsight]) -> List[str]:
        """Identify areas needing improvement"""
        weak_areas = []
        
        if quality_metrics.overall_accuracy < 0.5:
            weak_areas.append("Overall prediction accuracy")
        
        if quality_metrics.confidence_calibration < 0.6:
            weak_areas.append("Confidence calibration")
        
        if quality_metrics.performance_volatility > 0.3:
            weak_areas.append("Consistency and stability")
        
        if quality_metrics.false_confidence_rate > 0.4:
            weak_areas.append("Overconfidence in predictions")
        
        if abs(quality_metrics.prediction_bias) > 0.3:
            weak_areas.append("Prediction bias (over/under balance)")
        
        if quality_metrics.value_creation_rate < 0.3:
            weak_areas.append("Expected value identification")
        
        # Add pattern-based weak areas
        for pattern in patterns:
            if pattern.impact_score < -0.2:
                weak_areas.append(f"Pattern issue: {pattern.description}")
        
        return weak_areas
    
    def _identify_strong_areas(self, quality_metrics: QualityMetrics) -> List[str]:
        """Identify strong performing areas"""
        strong_areas = []
        
        if quality_metrics.overall_accuracy > 0.6:
            strong_areas.append("Strong prediction accuracy")
        
        if quality_metrics.confidence_calibration > 0.8:
            strong_areas.append("Well-calibrated confidence scoring")
        
        if quality_metrics.performance_volatility < 0.1:
            strong_areas.append("Consistent performance")
        
        if quality_metrics.improvement_trend > 0.1:
            strong_areas.append("Positive learning trend")
        
        if quality_metrics.adaptation_speed > 0.7:
            strong_areas.append("Fast adaptation to new contexts")
        
        return strong_areas
    
    def _generate_prompt_recommendations(self, quality_metrics: QualityMetrics,
                                       patterns: List[PatternInsight]) -> List[str]:
        """Generate specific prompt optimization recommendations"""
        recommendations = []
        
        if quality_metrics.overall_accuracy < 0.5:
            recommendations.append(
                "Add more specific criteria for evaluating player performance trends"
            )
            recommendations.append(
                "Include recent form analysis and injury status checks"
            )
        
        if quality_metrics.confidence_calibration < 0.6:
            recommendations.append(
                "Add confidence calibration instructions with specific thresholds"
            )
            recommendations.append(
                "Include examples of high vs low confidence scenarios"
            )
        
        if abs(quality_metrics.prediction_bias) > 0.3:
            bias_direction = "over" if quality_metrics.prediction_bias > 0 else "under"
            recommendations.append(
                f"Address {bias_direction} prediction bias with balanced analysis instructions"
            )
        
        if quality_metrics.false_confidence_rate > 0.4:
            recommendations.append(
                "Add conservative confidence scoring for uncertain situations"
            )
            recommendations.append(
                "Include risk factors that should lower confidence scores"
            )
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.recommendation and pattern.impact_score < -0.1:
                recommendations.append(f"Pattern fix: {pattern.recommendation}")
        
        return recommendations
    
    def _identify_priority_fixes(self, quality_metrics: QualityMetrics,
                               patterns: List[PatternInsight]) -> List[str]:
        """Identify the highest priority fixes needed"""
        priority_fixes = []
        
        # Critical accuracy issues
        if quality_metrics.overall_accuracy < 0.45:
            priority_fixes.append("CRITICAL: Improve basic prediction accuracy")
        
        # Severe calibration issues
        if quality_metrics.confidence_calibration < 0.4:
            priority_fixes.append("HIGH: Fix confidence calibration system")
        
        # Dangerous overconfidence
        if quality_metrics.false_confidence_rate > 0.5:
            priority_fixes.append("HIGH: Reduce overconfident predictions")
        
        # Critical patterns
        critical_patterns = [p for p in patterns if p.impact_score < -0.3]
        for pattern in critical_patterns:
            priority_fixes.append(f"URGENT: {pattern.description}")
        
        return priority_fixes
    
    def get_ml_feedback_data(self, agent_name: str = None, sport: str = None,
                           days: int = 30) -> Dict[str, Any]:
        """Get structured data for ML model feedback and training"""
        feedback_data = {
            'performance_metrics': {},
            'quality_scores': {},
            'pattern_insights': [],
            'feature_importance': {},
            'training_suggestions': []
        }
        
        # Get relevant quality metrics
        for key, metrics in self.quality_metrics.items():
            name, sport_key = key.split(':', 1)
            if (not agent_name or name == agent_name) and (not sport or sport_key == sport):
                feedback_data['performance_metrics'][key] = asdict(metrics)
        
        # Get recent picks for feature analysis
        picks = self.monitor.get_pick_analytics(
            agent_name=agent_name,
            sport=sport,
            days=days
        )
        
        if picks:
            # Calculate feature importance
            feedback_data['feature_importance'] = self._calculate_feature_importance(picks)
            
            # Generate training suggestions
            feedback_data['training_suggestions'] = self._generate_training_suggestions(picks)
        
        # Add relevant pattern insights
        feedback_data['pattern_insights'] = [
            asdict(insight) for insight in self.pattern_insights[-20:]
            if (not agent_name or agent_name in insight.description)
        ]
        
        return feedback_data
    
    def _calculate_feature_importance(self, picks: List[PickAnalytics]) -> Dict[str, float]:
        """Calculate which features are most important for prediction accuracy"""
        # This is a simplified analysis - in practice, you'd use more sophisticated methods
        feature_scores = defaultdict(list)
        
        for pick in picks:
            if pick.outcome in ['win', 'loss']:
                outcome_score = 1.0 if pick.outcome == 'win' else 0.0
                
                # Confidence feature
                feature_scores['confidence'].append((pick.confidence, outcome_score))
                
                # Stat category feature
                feature_scores[f'stat_{pick.stat_category}'].append((1.0, outcome_score))
                
                # Expected value feature
                if pick.expected_value is not None:
                    feature_scores['expected_value'].append((pick.expected_value, outcome_score))
        
        # Calculate correlations (simplified)
        importance = {}
        for feature, data in feature_scores.items():
            if len(data) >= 5:
                feature_values = [d[0] for d in data]
                outcomes = [d[1] for d in data]
                
                # Simple correlation calculation
                if len(set(feature_values)) > 1:
                    correlation = abs(np.corrcoef(feature_values, outcomes)[0, 1])
                    importance[feature] = correlation if not np.isnan(correlation) else 0.0
                else:
                    importance[feature] = 0.0
        
        return importance
    
    def _generate_training_suggestions(self, picks: List[PickAnalytics]) -> List[str]:
        """Generate suggestions for ML model training"""
        suggestions = []
        
        # Analyze data distribution
        if len(picks) < 100:
            suggestions.append("Collect more training data for robust model training")
        
        # Check outcome balance
        resolved_picks = [p for p in picks if p.outcome in ['win', 'loss']]
        if resolved_picks:
            win_rate = len([p for p in resolved_picks if p.outcome == 'win']) / len(resolved_picks)
            if win_rate < 0.3 or win_rate > 0.7:
                suggestions.append("Address class imbalance in training data")
        
        # Check feature diversity
        stat_categories = set(p.stat_category for p in picks)
        if len(stat_categories) < 3:
            suggestions.append("Increase diversity of stat categories in training data")
        
        # Check confidence distribution
        high_conf_picks = len([p for p in picks if p.confidence > 0.8])
        if high_conf_picks < len(picks) * 0.1:
            suggestions.append("Include more high-confidence examples in training")
        
        return suggestions
    
    def export_analytics_report(self, output_file: str, 
                              agent_name: str = None, sport: str = None) -> str:
        """Export comprehensive analytics report"""
        try:
            report = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'analysis_window_days': self.analysis_window,
                'quality_metrics': {},
                'pattern_insights': [],
                'prompt_analyses': {},
                'ml_feedback': {},
                'summary': {}
            }
            
            # Filter by agent/sport if specified
            for key, metrics in self.quality_metrics.items():
                name, sport_key = key.split(':', 1)
                if (not agent_name or name == agent_name) and (not sport or sport_key == sport):
                    report['quality_metrics'][key] = asdict(metrics)
            
            # Add relevant pattern insights
            report['pattern_insights'] = [
                asdict(insight) for insight in self.pattern_insights
                if (not agent_name or agent_name in insight.description)
            ]
            
            # Add prompt analyses
            for key, analysis in self.prompt_analyses.items():
                name, sport_key = key.split(':', 1)
                if (not agent_name or name == agent_name) and (not sport or sport_key == sport):
                    report['prompt_analyses'][key] = asdict(analysis)
            
            # Add ML feedback data
            report['ml_feedback'] = self.get_ml_feedback_data(agent_name, sport)
            
            # Generate summary
            report['summary'] = self._generate_analytics_summary(
                report['quality_metrics'], report['pattern_insights']
            )
            
            # Save report
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return str(output_path.absolute())
        
        except Exception as e:
            self.logger.logger.error(f"Failed to export analytics report: {e}")
            raise
    
    def _generate_analytics_summary(self, quality_metrics: Dict[str, Any],
                                  pattern_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of analytics findings"""
        summary = {
            'total_agents_analyzed': len(quality_metrics),
            'overall_performance': 'good',
            'key_findings': [],
            'urgent_issues': [],
            'recommendations': []
        }
        
        if not quality_metrics:
            return summary
        
        # Calculate overall metrics
        accuracies = [m.get('overall_accuracy', 0) for m in quality_metrics.values()]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        calibrations = [m.get('confidence_calibration', 0) for m in quality_metrics.values()]
        avg_calibration = sum(calibrations) / len(calibrations) if calibrations else 0
        
        # Determine overall performance
        if avg_accuracy > 0.6 and avg_calibration > 0.7:
            summary['overall_performance'] = 'excellent'
        elif avg_accuracy > 0.5 and avg_calibration > 0.6:
            summary['overall_performance'] = 'good'
        elif avg_accuracy > 0.45 and avg_calibration > 0.5:
            summary['overall_performance'] = 'fair'
        else:
            summary['overall_performance'] = 'poor'
        
        # Key findings
        summary['key_findings'] = [
            f"Average accuracy across agents: {avg_accuracy:.1%}",
            f"Average confidence calibration: {avg_calibration:.1%}",
            f"Total pattern insights detected: {len(pattern_insights)}"
        ]
        
        # Urgent issues
        for insight in pattern_insights:
            if insight.get('impact_score', 0) < -0.3:
                summary['urgent_issues'].append(insight.get('description', ''))
        
        # High-level recommendations
        if avg_accuracy < 0.5:
            summary['recommendations'].append("Immediate focus on improving prediction accuracy")
        if avg_calibration < 0.6:
            summary['recommendations'].append("Recalibrate confidence scoring across all agents")
        
        return summary

# Global analytics tracker instance
_global_tracker: Optional[AgentAnalyticsTracker] = None

def get_analytics_tracker() -> AgentAnalyticsTracker:
    """Get the global analytics tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = AgentAnalyticsTracker()
    return _global_tracker

def init_analytics_tracker(data_dir: str = "analytics_data",
                         analysis_window: int = 30) -> AgentAnalyticsTracker:
    """Initialize the global analytics tracker with custom settings"""
    global _global_tracker
    _global_tracker = AgentAnalyticsTracker(data_dir=data_dir, analysis_window=analysis_window)
    return _global_tracker