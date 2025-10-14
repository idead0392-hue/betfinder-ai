"""
Integration layer between OpenAI Agent Router and existing BetFinder AI Sport Agents
Provides seamless routing to OpenAI Assistants while maintaining compatibility
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import existing modules
from openai_agent_router import OpenAIAgentRouter, PropData, StatsData, ContextData
from sport_data_formatters import format_props_for_sport, extract_prop_insights
from sport_agents import SportAgent, BasketballAgent, FootballAgent, CollegeFootballAgent

# Import comprehensive logging and monitoring
from agent_logger import AgentLogger
from agent_error_handler import AgentErrorHandler, with_error_handling
from agent_monitor import AgentMonitor
from agent_analytics_tracker import AgentAnalyticsTracker

logger = logging.getLogger(__name__)

class EnhancedSportAgent:
    """
    Enhanced Sport Agent that can route to either OpenAI Assistants or local agents
    """
    
    def __init__(self, sport: str, use_openai: bool = True, openai_api_key: str = None):
        """
        Initialize enhanced sport agent
        
        Args:
            sport: Sport name (e.g., 'NBA', 'NFL', 'Soccer')
            use_openai: Whether to use OpenAI Assistants (True) or local agents (False)
            openai_api_key: OpenAI API key (optional, can use env var)
        """
        self.sport = sport.lower()
        self.use_openai = use_openai
        self.openai_enabled = False
        
        # Initialize comprehensive logging and monitoring
        self.agent_logger = AgentLogger(
            log_dir=f"logs/integration/{self.sport}"
        )
        
        # Initialize error handling
        self.error_handler = AgentErrorHandler()
        
        # Initialize monitoring
        self.monitor = AgentMonitor()
        
        # Initialize analytics tracker
        self.analytics = AgentAnalyticsTracker(
            data_dir=f"analytics_data/{self.sport}"
        )
        
        # Initialize OpenAI router if enabled
        if use_openai:
            try:
                self.openai_router = OpenAIAgentRouter(
                    api_key=openai_api_key or os.getenv('OPENAI_API_KEY'),
                    enable_logging=True
                )
                self.openai_enabled = True
                
                # Log successful OpenAI initialization
                self.agent_logger.log_agent_request(
                    agent_name=f"{self.sport}_enhanced_agent",
                    sport=self.sport,
                    request_data={
                        "action": "openai_initialization",
                        "status": "success"
                    }
                )
                
                logger.info(f"OpenAI routing enabled for {sport}")
            except Exception as e:
                # Log OpenAI initialization failure
                self.agent_logger.log_agent_response(
                    agent_name=f"{self.sport}_enhanced_agent",
                    sport=self.sport,
                    response_data={"error": str(e)},
                    success=False,
                    error_message=f"OpenAI initialization failed: {e}"
                )
                
                logger.warning(f"OpenAI routing failed for {sport}: {e}")
                self.openai_enabled = False
        
        # Initialize fallback local agent
        self.local_agent = self._create_local_agent(sport)
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.openai_usage_count = 0
        self.local_usage_count = 0
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.openai_usage_count = 0
        self.local_usage_count = 0
        
        # Initialize fallback local agent
        self.local_agent = self._create_local_agent(sport)
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.openai_usage_count = 0
        self.local_usage_count = 0
    
    def _create_local_agent(self, sport: str) -> SportAgent:
        """Create appropriate local sport agent as fallback"""
        sport_lower = sport.lower()
        
        if sport_lower in ['basketball', 'nba']:
            return BasketballAgent()
        elif sport_lower in ['football', 'nfl']:
            return FootballAgent()
        elif sport_lower in ['college_football', 'cfb']:
            return CollegeFootballAgent()
        else:
            # Default to base SportAgent for unsupported sports
            # Create a concrete implementation since SportAgent is abstract
            class GenericSportAgent(SportAgent):
                def __init__(self, sport_name: str):
                    super().__init__(sport_name)
                
                def analyze_props(self, props_data):
                    # Simple default implementation
                    return []
            
            return GenericSportAgent(sport)
    
    @with_error_handling()
    def analyze_props(self, 
                     props_data: List[Dict],
                     include_stats: bool = True,
                     include_context: bool = True,
                     force_local: bool = False) -> Dict[str, Any]:
        """
        Analyze props using either OpenAI Assistants or local agents
        
        Args:
            props_data: List of prop data dictionaries
            include_stats: Whether to include statistical analysis
            include_context: Whether to include contextual information
            force_local: Force use of local agent (bypass OpenAI)
        
        Returns:
            Analysis results with picks and recommendations
        """
        
        # Start operation tracking
        self.request_count += 1
        start_time = datetime.now()
        
        # Log analysis request
        self.agent_logger.log_agent_request(
            agent_name=f"{self.sport}_enhanced_agent",
            sport=self.sport,
            request_data={
                "action": "analyze_props",
                "props_count": len(props_data),
                "include_stats": include_stats,
                "include_context": include_context,
                "force_local": force_local,
                "request_number": self.request_count
            }
        )
        
        try:
            # Decide which analysis method to use
            use_openai_for_request = (
                self.openai_enabled and 
                not force_local and 
                len(props_data) > 0
            )
            
            if use_openai_for_request:
                result = self._analyze_with_openai(
                    props_data, include_stats, include_context
                )
                self.openai_usage_count += 1
                analysis_method = "openai"
            else:
                result = self._analyze_with_local_agent(
                    props_data, include_stats, include_context
                )
                self.local_usage_count += 1
                analysis_method = "local"
            
            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Add metadata
            result['analysis_metadata'] = {
                'sport': self.sport,
                'method': analysis_method,
                'timestamp': start_time.isoformat(),
                'duration_ms': duration_ms,
                'props_analyzed': len(props_data),
                'request_id': f"{self.sport}_{self.request_count}_{int(start_time.timestamp())}"
            }
            
            # Log successful response
            self.agent_logger.log_agent_response(
                agent_name=f"{self.sport}_enhanced_agent",
                sport=self.sport,
                response_data={
                    "method": analysis_method,
                    "duration_ms": duration_ms,
                    "picks_count": len(result.get('picks', [])),
                    "success": True
                },
                success=True
            )
            
            self.success_count += 1
            return result
            
        except Exception as e:
            error_msg = f"Error in prop analysis for {self.sport}: {e}"
            logger.error(error_msg)
            
            # Fallback to local agent if OpenAI fails
            if use_openai_for_request:
                logger.info(f"Falling back to local agent for {self.sport}")
                try:
                    result = self._analyze_with_local_agent(
                        props_data, include_stats, include_context
                    )
                    
                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                    
                    result['analysis_metadata'] = {
                        'sport': self.sport,
                        'method': 'local_fallback',
                        'timestamp': start_time.isoformat(),
                        'duration_ms': duration_ms,
                        'props_analyzed': len(props_data),
                        'fallback_reason': str(e)
                    }
                    
                    # Log fallback success
                    self.agent_logger.log_agent_response(
                        agent_name=f"{self.sport}_enhanced_agent",
                        sport=self.sport,
                        response_data={
                            "method": "local_fallback",
                            "duration_ms": duration_ms,
                            "fallback_reason": str(e),
                            "success": True
                        },
                        success=True
                    )
                    
                    self.success_count += 1
                    self.local_usage_count += 1
                    return result
                    
                except Exception as fallback_error:
                    error_msg = f"Both OpenAI and local agent failed for {self.sport}: {fallback_error}"
                    
                    # Log complete failure
                    self.agent_logger.log_agent_response(
                        agent_name=f"{self.sport}_enhanced_agent",
                        sport=self.sport,
                        response_data={
                            "openai_error": str(e),
                            "local_error": str(fallback_error)
                        },
                        success=False,
                        error_message=error_msg
                    )
                    
                    logger.error(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'picks': [],
                        'analysis_metadata': {
                            'sport': self.sport,
                            'method': 'failed',
                            'timestamp': start_time.isoformat(),
                            'openai_error': str(e),
                            'local_error': str(fallback_error)
                        }
                    }
            else:
                # Log direct failure
                self.agent_logger.log_agent_response(
                    agent_name=f"{self.sport}_enhanced_agent",
                    sport=self.sport,
                    response_data={"error": str(e)},
                    success=False,
                    error_message=error_msg
                )
                
                return {
                    'success': False,
                    'error': error_msg,
                    'picks': [],
                    'analysis_metadata': {
                        'sport': self.sport,
                        'method': 'failed',
                        'timestamp': start_time.isoformat(),
                        'error': str(e)
                    }
                }
    
    def _analyze_with_openai(self, 
                            props_data: List[Dict],
                            include_stats: bool,
                            include_context: bool) -> Dict[str, Any]:
        """Analyze props using OpenAI Assistants"""
        
        # Format data for OpenAI
        formatted_data = format_props_for_sport(
            self.sport, props_data, 
            stats_data={} if include_stats else None,
            context_data={} if include_context else None
        )
        
        # Convert to PropData objects
        prop_objects = []
        for prop in formatted_data['props']:
            prop_objects.append(PropData(
                player=prop.get('player', ''),
                prop=prop.get('prop', ''),
                line=prop.get('line', 0),
                odds=prop.get('odds', -110),
                opponent=prop.get('opponent', ''),
                sport=self.sport,
                matchup=prop.get('matchup', '')
            ))
        
        # Prepare stats and context
        stats = StatsData(**formatted_data.get('stats', {})) if include_stats else None
        context = ContextData(**formatted_data.get('context', {})) if include_context else None
        
        # Route to OpenAI Agent
        openai_result = self.openai_router.route_to_sport_agent(
            sport=self.sport,
            props=prop_objects,
            stats=stats,
            context=context
        )
        
        if not openai_result.get('success', False):
            raise Exception(f"OpenAI analysis failed: {openai_result.get('error', 'Unknown error')}")
        
        # Parse OpenAI response and convert to standard format
        analysis_text = openai_result.get('analysis', '')
        picks = self._parse_openai_analysis(analysis_text, props_data)
        
        return {
            'success': True,
            'picks': picks,
            'total_picks': len(picks),
            'openai_response': openai_result,
            'formatted_data': formatted_data,
            'insights': extract_prop_insights(formatted_data['props'])
        }
    
    def _analyze_with_local_agent(self, 
                                 props_data: List[Dict],
                                 include_stats: bool,
                                 include_context: bool) -> Dict[str, Any]:
        """Analyze props using local sport agent"""
        
        # Use existing local agent
        picks = self.local_agent.make_picks(
            props_data=props_data,
            log_to_ledger=False  # We'll handle logging separately
        )
        
        return {
            'success': True,
            'picks': picks,
            'total_picks': len(picks),
            'local_agent': type(self.local_agent).__name__,
            'method': 'local_analysis'
        }
    
    def _parse_openai_analysis(self, analysis_text: str, original_props: List[Dict]) -> List[Dict]:
        """Parse OpenAI analysis text into structured picks"""
        picks = []
        
        try:
            # Try to parse as JSON first
            if analysis_text.strip().startswith('{') or analysis_text.strip().startswith('['):
                parsed = json.loads(analysis_text)
                if isinstance(parsed, dict) and 'picks' in parsed:
                    return parsed['picks']
                elif isinstance(parsed, list):
                    return parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback: generate picks from original props with default values
        for i, prop in enumerate(original_props):
            pick = {
                'player_name': prop.get('Name', prop.get('player', f'Player_{i}')),
                'stat_type': prop.get('Prop', prop.get('prop', 'Unknown')),
                'line': prop.get('Points', prop.get('line', 0)),
                'over_under': 'Over',  # Default
                'confidence': 75,  # Default confidence
                'expected_value': 5.0,  # Default EV
                'reasoning': f"OpenAI analysis: {analysis_text[:100]}..." if analysis_text else "Default analysis",
                'odds': prop.get('odds', -110),
                'matchup': prop.get('matchup', 'TBD'),
                'prizepicks_classification': 'DECENT',
                'analysis_method': 'openai_parsed'
            }
            picks.append(pick)
        
        return picks
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent"""
        total_requests = self.request_count
        success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'sport': self.sport,
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'success_rate': round(success_rate, 2),
            'openai_usage': self.openai_usage_count,
            'local_usage': self.local_usage_count,
            'openai_enabled': self.openai_enabled,
            'openai_percentage': round((self.openai_usage_count / total_requests * 100), 2) if total_requests > 0 else 0
        }

class AgentManager:
    """Manages multiple enhanced sport agents"""
    
    def __init__(self, use_openai: bool = True, openai_api_key: str = None):
        """
        Initialize agent manager
        
        Args:
            use_openai: Whether to enable OpenAI routing globally
            openai_api_key: OpenAI API key
        """
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self.agents: Dict[str, EnhancedSportAgent] = {}
        
        # Initialize agents for supported sports
        self.supported_sports = [
            'basketball', 'football', 'college_football', 'tennis',
            'baseball', 'hockey', 'soccer', 'csgo', 'league_of_legends',
            'dota2', 'valorant', 'overwatch', 'golf'
        ]
        
        for sport in self.supported_sports:
            self.agents[sport] = EnhancedSportAgent(
                sport=sport,
                use_openai=use_openai,
                openai_api_key=openai_api_key
            )
        
        logger.info(f"Agent Manager initialized with {len(self.agents)} agents")
    
    def get_agent(self, sport: str) -> Optional[EnhancedSportAgent]:
        """Get agent for specific sport"""
        sport_key = sport.lower().strip()
        return self.agents.get(sport_key)
    
    def analyze_props(self, 
                     sport: str,
                     props_data: List[Dict],
                     **kwargs) -> Dict[str, Any]:
        """Analyze props for specific sport"""
        agent = self.get_agent(sport)
        if not agent:
            return {
                'success': False,
                'error': f"No agent available for sport: {sport}",
                'supported_sports': self.supported_sports
            }
        
        return agent.analyze_props(props_data, **kwargs)
    
    def get_all_performance_stats(self) -> Dict[str, Any]:
        """Get performance stats for all agents"""
        stats = {}
        total_requests = 0
        total_successes = 0
        total_openai_usage = 0
        
        for sport, agent in self.agents.items():
            agent_stats = agent.get_performance_stats()
            stats[sport] = agent_stats
            total_requests += agent_stats['total_requests']
            total_successes += agent_stats['successful_requests']
            total_openai_usage += agent_stats['openai_usage']
        
        # Calculate overall statistics
        overall_success_rate = (total_successes / total_requests * 100) if total_requests > 0 else 0
        overall_openai_percentage = (total_openai_usage / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'individual_agents': stats,
            'overall_stats': {
                'total_requests': total_requests,
                'total_successes': total_successes,
                'overall_success_rate': round(overall_success_rate, 2),
                'total_openai_usage': total_openai_usage,
                'overall_openai_percentage': round(overall_openai_percentage, 2),
                'active_agents': len([a for a in self.agents.values() if a.request_count > 0])
            }
        }
    
    def switch_to_local_mode(self):
        """Switch all agents to local mode (disable OpenAI)"""
        for agent in self.agents.values():
            agent.use_openai = False
            agent.openai_enabled = False
        logger.info("All agents switched to local mode")
    
    def switch_to_openai_mode(self):
        """Switch all agents to OpenAI mode (if available)"""
        for agent in self.agents.values():
            if hasattr(agent, 'openai_router') and agent.openai_router:
                agent.use_openai = True
                agent.openai_enabled = True
        logger.info("All agents switched to OpenAI mode (where available)")

# Factory function for easy integration
def create_agent_manager(use_openai: bool = True, openai_api_key: str = None) -> AgentManager:
    """Create and return an Agent Manager instance"""
    return AgentManager(use_openai=use_openai, openai_api_key=openai_api_key)

# Example usage and testing
if __name__ == "__main__":
    # Test the integration
    manager = create_agent_manager(use_openai=False)  # Start with local mode for testing
    
    # Sample prop data
    sample_props = [
        {
            'Name': 'LeBron James',
            'Prop': 'Over 24.5 Points',
            'Points': 24.5,
            'team': 'LAL',
            'opponent': 'GSW'
        }
    ]
    
    # Test analysis
    result = manager.analyze_props('basketball', sample_props)
    print("Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Check performance stats
    stats = manager.get_all_performance_stats()
    print("\nPerformance Stats:")
    print(json.dumps(stats, indent=2))