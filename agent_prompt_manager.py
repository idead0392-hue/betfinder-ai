"""
Agent Prompt Manager - Dynamic system prompts with versioning, A/B testing, and iterative improvement
Manages prompt templates, tracks performance, and optimizes agent instructions
"""

import json
import random
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import uuid

from agent_logger import get_logger, LogLevel, AgentEvent
from agent_monitor import get_monitor
from agent_analytics_tracker import get_analytics_tracker

class PromptType(Enum):
    """Types of prompts"""
    SYSTEM = "system"
    INSTRUCTION = "instruction"
    CONTEXT = "context"
    EXAMPLE = "example"
    CONSTRAINT = "constraint"
    OUTPUT_FORMAT = "output_format"

class TestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

@dataclass
class PromptTemplate:
    """A versioned prompt template"""
    template_id: str
    name: str
    prompt_type: PromptType
    content: str
    version: str
    sport: str
    agent_name: str
    
    # Metadata
    created_at: str = ""
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Performance tracking
    usage_count: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_response_time: float = 0.0
    
    # Version control
    parent_version: Optional[str] = None
    changes_from_parent: str = ""
    
    # Status
    is_active: bool = False
    is_deprecated: bool = False
    deprecation_reason: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.template_id:
            self.template_id = str(uuid.uuid4())

@dataclass
class ABTest:
    """A/B test configuration and results"""
    test_id: str
    name: str
    description: str
    sport: str
    agent_name: str
    
    # Test configuration
    control_template_id: str
    variant_template_ids: List[str]
    traffic_split: Dict[str, float]  # template_id -> percentage
    
    # Test criteria
    success_metric: str = "win_rate"  # win_rate, confidence_calibration, response_time
    minimum_sample_size: int = 50
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.05
    
    # Test status
    status: TestStatus = TestStatus.DRAFT
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Results
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    winner: Optional[str] = None
    statistical_significance: bool = False
    
    # Tracking
    created_at: str = ""
    created_by: str = "system"
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.test_id:
            self.test_id = str(uuid.uuid4())

@dataclass
class PromptPerformance:
    """Performance metrics for a prompt template"""
    template_id: str
    measurement_period: str  # ISO date range
    
    # Usage metrics
    total_uses: int = 0
    unique_sessions: int = 0
    
    # Performance metrics
    win_rate: float = 0.0
    confidence_calibration: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    
    # Quality metrics
    avg_confidence: float = 0.0
    prediction_accuracy: float = 0.0
    value_creation_rate: float = 0.0
    
    # Context-specific performance
    performance_by_context: Dict[str, float] = field(default_factory=dict)
    
    # Comparison metrics
    vs_baseline: Dict[str, float] = field(default_factory=dict)
    improvement_over_previous: Dict[str, float] = field(default_factory=dict)

class PromptManager:
    """Comprehensive prompt management system"""
    
    def __init__(self, data_dir: str = "prompt_data"):
        """Initialize the prompt manager"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = get_logger()
        self.monitor = get_monitor()
        self.analytics_tracker = get_analytics_tracker()
        
        # Storage
        self.templates: Dict[str, PromptTemplate] = {}
        self.ab_tests: Dict[str, ABTest] = {}
        self.performance_history: Dict[str, List[PromptPerformance]] = {}
        
        # Active assignments
        self.active_assignments: Dict[str, str] = {}  # agent_key -> template_id
        self.ab_test_assignments: Dict[str, str] = {}  # session_id -> template_id
        
        # File paths
        self.templates_file = self.data_dir / "prompt_templates.json"
        self.ab_tests_file = self.data_dir / "ab_tests.json"
        self.performance_file = self.data_dir / "performance_history.json"
        self.assignments_file = self.data_dir / "assignments.json"
        
        # Load existing data
        self._load_data()
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def _load_data(self):
        """Load existing prompt data"""
        try:
            if self.templates_file.exists():
                with open(self.templates_file, 'r') as f:
                    data = json.load(f)
                    for template_id, template_data in data.items():
                        self.templates[template_id] = PromptTemplate(**template_data)
            
            if self.ab_tests_file.exists():
                with open(self.ab_tests_file, 'r') as f:
                    data = json.load(f)
                    for test_id, test_data in data.items():
                        self.ab_tests[test_id] = ABTest(**test_data)
            
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    for template_id, perf_list in data.items():
                        self.performance_history[template_id] = [
                            PromptPerformance(**perf_data) for perf_data in perf_list
                        ]
            
            if self.assignments_file.exists():
                with open(self.assignments_file, 'r') as f:
                    data = json.load(f)
                    self.active_assignments = data.get('active_assignments', {})
                    self.ab_test_assignments = data.get('ab_test_assignments', {})
        
        except Exception as e:
            self.logger.logger.error(f"Failed to load prompt data: {e}")
    
    def _save_data(self):
        """Save prompt data to disk"""
        try:
            # Save templates
            templates_data = {}
            for template_id, template in self.templates.items():
                templates_data[template_id] = asdict(template)
            
            with open(self.templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2, default=str)
            
            # Save A/B tests
            tests_data = {}
            for test_id, test in self.ab_tests.items():
                tests_data[test_id] = asdict(test)
            
            with open(self.ab_tests_file, 'w') as f:
                json.dump(tests_data, f, indent=2, default=str)
            
            # Save performance history
            performance_data = {}
            for template_id, perf_list in self.performance_history.items():
                performance_data[template_id] = [asdict(perf) for perf in perf_list]
            
            with open(self.performance_file, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            # Save assignments
            assignments_data = {
                'active_assignments': self.active_assignments,
                'ab_test_assignments': self.ab_test_assignments
            }
            
            with open(self.assignments_file, 'w') as f:
                json.dump(assignments_data, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.logger.error(f"Failed to save prompt data: {e}")
    
    def _initialize_default_templates(self):
        """Initialize default prompt templates if none exist"""
        if self.templates:
            return  # Already have templates
        
        # Create default system prompts for each sport
        sports_prompts = {
            "basketball": {
                "system": """You are an expert NBA basketball analyst specializing in player prop betting. 
                Analyze player performance data, recent trends, matchup factors, and injury reports to make accurate predictions.
                
                Key factors to consider:
                - Recent game performance (last 5-10 games)
                - Season averages and trends
                - Opponent defensive rankings
                - Home/away performance splits
                - Rest days and back-to-back games
                - Injury status and load management
                - Historical head-to-head performance
                
                Always provide a confidence score (0.0-1.0) based on the strength of your analysis.
                Be conservative with high confidence scores - reserve 0.9+ for very clear situations.""",
                
                "output_format": """Provide your analysis in this format:
                {
                    "prediction": "over" or "under",
                    "confidence": 0.0-1.0,
                    "reasoning": "Clear explanation of your analysis",
                    "key_factors": ["factor1", "factor2", "factor3"],
                    "risk_assessment": "low/medium/high",
                    "expected_value": numerical_ev_if_calculable
                }"""
            },
            
            "football": {
                "system": """You are an expert NFL football analyst specializing in player prop betting.
                Analyze player performance, team dynamics, weather conditions, and game script to make accurate predictions.
                
                Key factors to consider:
                - Recent performance trends
                - Target share and usage patterns
                - Opponent defensive rankings vs position
                - Weather conditions for outdoor games
                - Game script and pace projections
                - Injury reports and snap counts
                - Home field advantage
                - Divisional matchup history
                
                Account for the volatility of football stats and be appropriately cautious with confidence levels.""",
                
                "output_format": """Provide your analysis in this format:
                {
                    "prediction": "over" or "under",
                    "confidence": 0.0-1.0,
                    "reasoning": "Clear explanation of your analysis",
                    "key_factors": ["factor1", "factor2", "factor3"],
                    "risk_assessment": "low/medium/high",
                    "expected_value": numerical_ev_if_calculable
                }"""
            }
        }
        
        # Create templates for each sport
        for sport, prompts in sports_prompts.items():
            for prompt_type, content in prompts.items():
                template = PromptTemplate(
                    template_id=str(uuid.uuid4()),
                    name=f"{sport}_{prompt_type}_v1.0",
                    prompt_type=PromptType(prompt_type),
                    content=content,
                    version="1.0",
                    sport=sport,
                    agent_name=f"{sport}_agent",
                    description=f"Default {prompt_type} prompt for {sport}",
                    is_active=True
                )
                
                self.templates[template.template_id] = template
        
        self._save_data()
    
    def get_agent_key(self, agent_name: str, sport: str) -> str:
        """Generate key for agent"""
        return f"{agent_name}:{sport}"
    
    def create_template(self, name: str, prompt_type: PromptType, content: str,
                       sport: str, agent_name: str, description: str = "",
                       tags: List[str] = None, parent_version: str = None) -> str:
        """Create a new prompt template"""
        
        # Generate version number
        existing_versions = [
            t.version for t in self.templates.values()
            if t.name.startswith(name.split('_v')[0]) and t.sport == sport
        ]
        
        if existing_versions:
            # Extract version numbers and increment
            version_nums = []
            for v in existing_versions:
                try:
                    major, minor = v.split('.')
                    version_nums.append((int(major), int(minor)))
                except:
                    continue
            
            if version_nums:
                max_major, max_minor = max(version_nums)
                new_version = f"{max_major}.{max_minor + 1}"
            else:
                new_version = "1.0"
        else:
            new_version = "1.0"
        
        # Create template
        template = PromptTemplate(
            template_id=str(uuid.uuid4()),
            name=f"{name}_v{new_version}",
            prompt_type=prompt_type,
            content=content,
            version=new_version,
            sport=sport,
            agent_name=agent_name,
            description=description,
            tags=tags or [],
            parent_version=parent_version
        )
        
        # Store template
        self.templates[template.template_id] = template
        self._save_data()
        
        # Log creation
        self.logger.log_event(
            event_type=AgentEvent.PERFORMANCE_METRIC,
            level=LogLevel.INFO,
            agent_name=agent_name,
            sport=sport,
            session_id="prompt_manager",
            metadata={
                'action': 'template_created',
                'template_id': template.template_id,
                'version': new_version,
                'prompt_type': prompt_type.value
            }
        )
        
        return template.template_id
    
    def get_active_prompt(self, agent_name: str, sport: str, 
                         session_id: str = None) -> Optional[PromptTemplate]:
        """Get the active prompt for an agent (considering A/B tests)"""
        agent_key = self.get_agent_key(agent_name, sport)
        
        # Check if agent is in an A/B test
        if session_id:
            template_id = self._get_ab_test_assignment(agent_name, sport, session_id)
            if template_id:
                return self.templates.get(template_id)
        
        # Get regular active assignment
        template_id = self.active_assignments.get(agent_key)
        if template_id and template_id in self.templates:
            return self.templates[template_id]
        
        # Find default active template
        active_templates = [
            t for t in self.templates.values()
            if t.agent_name == agent_name and t.sport == sport and t.is_active
        ]
        
        if active_templates:
            # Return the latest version
            return max(active_templates, key=lambda t: t.version)
        
        return None
    
    def assign_template(self, agent_name: str, sport: str, template_id: str):
        """Assign a template to an agent"""
        agent_key = self.get_agent_key(agent_name, sport)
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        if template.agent_name != agent_name or template.sport != sport:
            raise ValueError("Template agent/sport mismatch")
        
        # Deactivate old template
        old_template_id = self.active_assignments.get(agent_key)
        if old_template_id and old_template_id in self.templates:
            self.templates[old_template_id].is_active = False
        
        # Activate new template
        template.is_active = True
        self.active_assignments[agent_key] = template_id
        
        self._save_data()
        
        # Log assignment
        self.logger.log_event(
            event_type=AgentEvent.PERFORMANCE_METRIC,
            level=LogLevel.INFO,
            agent_name=agent_name,
            sport=sport,
            session_id="prompt_manager",
            metadata={
                'action': 'template_assigned',
                'template_id': template_id,
                'previous_template': old_template_id
            }
        )
    
    def create_ab_test(self, name: str, description: str, agent_name: str, sport: str,
                      control_template_id: str, variant_template_ids: List[str],
                      traffic_split: Dict[str, float] = None,
                      success_metric: str = "win_rate",
                      minimum_sample_size: int = 50) -> str:
        """Create an A/B test"""
        
        # Validate templates
        all_template_ids = [control_template_id] + variant_template_ids
        for template_id in all_template_ids:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
        
        # Default traffic split
        if traffic_split is None:
            split_size = 1.0 / len(all_template_ids)
            traffic_split = {tid: split_size for tid in all_template_ids}
        
        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 0.01:
            raise ValueError("Traffic split must sum to 1.0")
        
        # Create test
        test = ABTest(
            test_id=str(uuid.uuid4()),
            name=name,
            description=description,
            sport=sport,
            agent_name=agent_name,
            control_template_id=control_template_id,
            variant_template_ids=variant_template_ids,
            traffic_split=traffic_split,
            success_metric=success_metric,
            minimum_sample_size=minimum_sample_size
        )
        
        # Store test
        self.ab_tests[test.test_id] = test
        self._save_data()
        
        # Log creation
        self.logger.log_event(
            event_type=AgentEvent.PERFORMANCE_METRIC,
            level=LogLevel.INFO,
            agent_name=agent_name,
            sport=sport,
            session_id="prompt_manager",
            metadata={
                'action': 'ab_test_created',
                'test_id': test.test_id,
                'templates': all_template_ids,
                'success_metric': success_metric
            }
        )
        
        return test.test_id
    
    def start_ab_test(self, test_id: str):
        """Start an A/B test"""
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test = self.ab_tests[test_id]
        test.status = TestStatus.ACTIVE
        test.start_date = datetime.now(timezone.utc).isoformat()
        
        self._save_data()
        
        # Log start
        self.logger.log_event(
            event_type=AgentEvent.PERFORMANCE_METRIC,
            level=LogLevel.INFO,
            agent_name=test.agent_name,
            sport=test.sport,
            session_id="prompt_manager",
            metadata={
                'action': 'ab_test_started',
                'test_id': test_id
            }
        )
    
    def _get_ab_test_assignment(self, agent_name: str, sport: str, session_id: str) -> Optional[str]:
        """Get A/B test assignment for a session"""
        # Find active tests for this agent
        active_tests = [
            test for test in self.ab_tests.values()
            if (test.status == TestStatus.ACTIVE and 
                test.agent_name == agent_name and 
                test.sport == sport)
        ]
        
        if not active_tests:
            return None
        
        # Use the first active test (in practice, limit to one test per agent)
        test = active_tests[0]
        
        # Check if session already has assignment
        if session_id in self.ab_test_assignments:
            assigned_template = self.ab_test_assignments[session_id]
            # Verify assignment is still valid for this test
            all_templates = [test.control_template_id] + test.variant_template_ids
            if assigned_template in all_templates:
                return assigned_template
        
        # Assign template based on traffic split
        template_id = self._assign_ab_template(test, session_id)
        self.ab_test_assignments[session_id] = template_id
        self._save_data()
        
        return template_id
    
    def _assign_ab_template(self, test: ABTest, session_id: str) -> str:
        """Assign a template for A/B test based on traffic split"""
        # Use session_id hash for consistent assignment
        session_hash = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        random.seed(session_hash)
        
        # Weighted random selection
        templates = list(test.traffic_split.keys())
        weights = list(test.traffic_split.values())
        
        rand_val = random.random()
        cumulative = 0.0
        
        for template_id, weight in zip(templates, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return template_id
        
        # Fallback to control
        return test.control_template_id
    
    def record_prompt_usage(self, template_id: str, session_id: str,
                           success: bool = True, confidence: float = None,
                           response_time_ms: float = None, context: Dict[str, Any] = None):
        """Record usage of a prompt template"""
        if template_id not in self.templates:
            return
        
        template = self.templates[template_id]
        
        # Update template usage stats
        template.usage_count += 1
        
        if success:
            # Update running averages
            if confidence is not None:
                if template.avg_confidence == 0:
                    template.avg_confidence = confidence
                else:
                    template.avg_confidence = (
                        (template.avg_confidence * (template.usage_count - 1)) + confidence
                    ) / template.usage_count
            
            if response_time_ms is not None:
                if template.avg_response_time == 0:
                    template.avg_response_time = response_time_ms
                else:
                    template.avg_response_time = (
                        (template.avg_response_time * (template.usage_count - 1)) + response_time_ms
                    ) / template.usage_count
        
        # Update A/B test results if applicable
        self._update_ab_test_results(template_id, session_id, success, confidence, context)
        
        self._save_data()
    
    def _update_ab_test_results(self, template_id: str, session_id: str,
                              success: bool, confidence: float = None,
                              context: Dict[str, Any] = None):
        """Update A/B test results"""
        # Find tests that include this template
        relevant_tests = [
            test for test in self.ab_tests.values()
            if (test.status == TestStatus.ACTIVE and 
                template_id in ([test.control_template_id] + test.variant_template_ids))
        ]
        
        for test in relevant_tests:
            if template_id not in test.results:
                test.results[template_id] = {
                    'uses': 0,
                    'successes': 0,
                    'success_rate': 0.0,
                    'total_confidence': 0.0,
                    'avg_confidence': 0.0
                }
            
            results = test.results[template_id]
            results['uses'] += 1
            
            if success:
                results['successes'] += 1
            
            results['success_rate'] = results['successes'] / results['uses']
            
            if confidence is not None:
                results['total_confidence'] += confidence
                results['avg_confidence'] = results['total_confidence'] / results['uses']
            
            # Check if test is ready for evaluation
            self._evaluate_ab_test(test)
    
    def _evaluate_ab_test(self, test: ABTest):
        """Evaluate if A/B test has reached statistical significance"""
        if test.status != TestStatus.ACTIVE:
            return
        
        # Check minimum sample size
        total_samples = sum(
            test.results.get(tid, {}).get('uses', 0)
            for tid in [test.control_template_id] + test.variant_template_ids
        )
        
        if total_samples < test.minimum_sample_size:
            return
        
        # Simple statistical test (in practice, use more sophisticated methods)
        control_results = test.results.get(test.control_template_id, {})
        control_rate = control_results.get('success_rate', 0.0)
        control_samples = control_results.get('uses', 0)
        
        if control_samples == 0:
            return
        
        # Find best performing variant
        best_variant = None
        best_rate = control_rate
        significant_difference = False
        
        for variant_id in test.variant_template_ids:
            variant_results = test.results.get(variant_id, {})
            variant_rate = variant_results.get('success_rate', 0.0)
            variant_samples = variant_results.get('uses', 0)
            
            if variant_samples == 0:
                continue
            
            # Simple significance test (replace with proper statistical test)
            rate_difference = abs(variant_rate - control_rate)
            
            if (rate_difference > test.effect_size_threshold and 
                variant_samples >= 30 and control_samples >= 30):
                
                if variant_rate > best_rate:
                    best_variant = variant_id
                    best_rate = variant_rate
                    significant_difference = True
        
        # Complete test if significant results found
        if significant_difference:
            test.status = TestStatus.COMPLETED
            test.end_date = datetime.now(timezone.utc).isoformat()
            test.winner = best_variant
            test.statistical_significance = True
            
            # Log completion
            self.logger.log_event(
                event_type=AgentEvent.PERFORMANCE_METRIC,
                level=LogLevel.INFO,
                agent_name=test.agent_name,
                sport=test.sport,
                session_id="prompt_manager",
                metadata={
                    'action': 'ab_test_completed',
                    'test_id': test.test_id,
                    'winner': best_variant,
                    'improvement': best_rate - control_rate
                }
            )
    
    def get_performance_report(self, template_id: str = None,
                             agent_name: str = None, sport: str = None,
                             days: int = 7) -> Dict[str, Any]:
        """Get performance report for templates"""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Filter templates
        templates_to_analyze = []
        for tid, template in self.templates.items():
            if template_id and tid != template_id:
                continue
            if agent_name and template.agent_name != agent_name:
                continue
            if sport and template.sport != sport:
                continue
            templates_to_analyze.append(template)
        
        report = {
            'period': f"{days} days",
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'templates': {},
            'ab_tests': {},
            'summary': {}
        }
        
        # Analyze each template
        total_usage = 0
        total_success_rate = 0.0
        template_count = 0
        
        for template in templates_to_analyze:
            # Get recent performance data
            performance = self._calculate_template_performance(template, start_date, end_date)
            
            report['templates'][template.template_id] = {
                'name': template.name,
                'version': template.version,
                'sport': template.sport,
                'agent_name': template.agent_name,
                'is_active': template.is_active,
                'performance': asdict(performance)
            }
            
            if performance.total_uses > 0:
                total_usage += performance.total_uses
                total_success_rate += performance.win_rate
                template_count += 1
        
        # Analyze active A/B tests
        for test_id, test in self.ab_tests.items():
            if test.status == TestStatus.ACTIVE:
                if agent_name and test.agent_name != agent_name:
                    continue
                if sport and test.sport != sport:
                    continue
                
                report['ab_tests'][test_id] = {
                    'name': test.name,
                    'status': test.status.value,
                    'results': test.results,
                    'statistical_significance': test.statistical_significance
                }
        
        # Generate summary
        report['summary'] = {
            'total_templates': len(templates_to_analyze),
            'active_templates': len([t for t in templates_to_analyze if t.is_active]),
            'total_usage': total_usage,
            'avg_success_rate': total_success_rate / template_count if template_count > 0 else 0,
            'active_ab_tests': len([t for t in self.ab_tests.values() if t.status == TestStatus.ACTIVE])
        }
        
        return report
    
    def _calculate_template_performance(self, template: PromptTemplate,
                                      start_date: datetime, end_date: datetime) -> PromptPerformance:
        """Calculate performance metrics for a template in a time period"""
        # This would integrate with the analytics tracker to get real performance data
        # For now, using template's running averages
        
        performance = PromptPerformance(
            template_id=template.template_id,
            measurement_period=f"{start_date.isoformat()} to {end_date.isoformat()}",
            total_uses=template.usage_count,
            win_rate=template.success_rate,
            avg_confidence=template.avg_confidence,
            avg_response_time=template.avg_response_time
        )
        
        # Get more detailed metrics from analytics tracker if available
        try:
            picks = self.monitor.get_pick_analytics(
                agent_name=template.agent_name,
                sport=template.sport,
                days=(end_date - start_date).days
            )
            
            if picks:
                resolved_picks = [p for p in picks if p.outcome in ['win', 'loss']]
                if resolved_picks:
                    wins = len([p for p in resolved_picks if p.outcome == 'win'])
                    performance.win_rate = wins / len(resolved_picks)
                    performance.prediction_accuracy = performance.win_rate
                
                # Calculate other metrics
                if picks:
                    confidences = [p.confidence for p in picks if p.confidence is not None]
                    if confidences:
                        performance.avg_confidence = sum(confidences) / len(confidences)
                    
                    ev_picks = [p for p in picks if p.expected_value is not None and p.expected_value > 0]
                    performance.value_creation_rate = len(ev_picks) / len(picks) if picks else 0
        
        except Exception as e:
            self.logger.logger.error(f"Failed to calculate detailed performance: {e}")
        
        return performance
    
    def optimize_prompts(self, agent_name: str = None, sport: str = None) -> List[str]:
        """Automatically suggest prompt optimizations"""
        recommendations = []
        
        # Get analytics for optimization suggestions
        try:
            # Get quality metrics
            quality_metrics = {}
            for key, metrics in self.analytics_tracker.quality_metrics.items():
                name, sport_key = key.split(':', 1)
                if (not agent_name or name == agent_name) and (not sport or sport_key == sport):
                    quality_metrics[key] = metrics
            
            # Analyze each agent's prompts
            for agent_key, metrics in quality_metrics.items():
                name, sport_key = agent_key.split(':', 1)
                
                # Get current active template
                current_template = self.get_active_prompt(name, sport_key)
                if not current_template:
                    continue
                
                # Generate specific recommendations
                agent_recommendations = self._generate_optimization_recommendations(
                    current_template, metrics
                )
                
                recommendations.extend([
                    f"[{name}:{sport_key}] {rec}" for rec in agent_recommendations
                ])
        
        except Exception as e:
            self.logger.logger.error(f"Failed to generate optimization recommendations: {e}")
        
        return recommendations
    
    def _generate_optimization_recommendations(self, template: PromptTemplate,
                                             quality_metrics) -> List[str]:
        """Generate specific optimization recommendations for a template"""
        recommendations = []
        
        # Accuracy-based recommendations
        if quality_metrics.overall_accuracy < 0.5:
            recommendations.append(
                "Consider adding more specific analysis criteria for better accuracy"
            )
        
        # Calibration-based recommendations
        if quality_metrics.confidence_calibration < 0.6:
            recommendations.append(
                "Add confidence calibration guidelines to improve score reliability"
            )
        
        # Bias-based recommendations
        if abs(quality_metrics.prediction_bias) > 0.3:
            bias_direction = "over" if quality_metrics.prediction_bias > 0 else "under"
            recommendations.append(
                f"Address {bias_direction} prediction bias in prompt instructions"
            )
        
        # Volatility-based recommendations
        if quality_metrics.performance_volatility > 0.3:
            recommendations.append(
                "Add consistency guidelines to reduce performance volatility"
            )
        
        # Context-specific recommendations
        context_performance = quality_metrics.performance_by_context
        if context_performance:
            worst_context = min(context_performance.items(), key=lambda x: x[1])
            if worst_context[1] < 0.4:
                recommendations.append(
                    f"Improve analysis for {worst_context[0]} situations"
                )
        
        return recommendations
    
    def export_prompt_data(self, output_file: str, include_performance: bool = True) -> str:
        """Export all prompt data"""
        try:
            export_data = {
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'templates': {},
                'ab_tests': {},
                'active_assignments': self.active_assignments,
                'performance_summary': {}
            }
            
            # Export templates
            for template_id, template in self.templates.items():
                export_data['templates'][template_id] = asdict(template)
            
            # Export A/B tests
            for test_id, test in self.ab_tests.items():
                export_data['ab_tests'][test_id] = asdict(test)
            
            # Export performance data if requested
            if include_performance:
                export_data['performance_history'] = {}
                for template_id, perf_list in self.performance_history.items():
                    export_data['performance_history'][template_id] = [
                        asdict(perf) for perf in perf_list
                    ]
            
            # Generate summary
            export_data['performance_summary'] = self.get_performance_report()
            
            # Save to file
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return str(output_path.absolute())
        
        except Exception as e:
            self.logger.logger.error(f"Failed to export prompt data: {e}")
            raise

# Global prompt manager instance
_global_prompt_manager: Optional[PromptManager] = None

def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance"""
    global _global_prompt_manager
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager()
    return _global_prompt_manager

def init_prompt_manager(data_dir: str = "prompt_data") -> PromptManager:
    """Initialize the global prompt manager with custom settings"""
    global _global_prompt_manager
    _global_prompt_manager = PromptManager(data_dir=data_dir)
    return _global_prompt_manager