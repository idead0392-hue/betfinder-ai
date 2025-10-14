"""
OpenAI Agent Router for BetFinder AI
Routes prop analysis requests to specialized OpenAI Agents via Assistants API
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI

# Import configuration manager
from agent_config import get_config_manager

# Import comprehensive logging and monitoring
from agent_logger import AgentLogger
from agent_error_handler import AgentErrorHandler, with_error_handling
from agent_monitor import AgentMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PropData:
    """Structured prop betting data"""
    player: str
    prop: str
    line: float
    odds: int
    opponent: str
    sport: str
    matchup: str
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class StatsData:
    """Player/team statistical data"""
    season_avg: Optional[float] = None
    last_5_avg: Optional[float] = None
    last_10_avg: Optional[float] = None
    home_avg: Optional[float] = None
    away_avg: Optional[float] = None
    vs_opponent_avg: Optional[float] = None
    trend: Optional[str] = None
    percentile: Optional[float] = None
    
@dataclass
class ContextData:
    """Additional context for analysis"""
    injuries: List[str] = None
    weather: Optional[str] = None
    news: List[str] = None
    matchup_notes: Optional[str] = None
    meta_info: Optional[str] = None
    
    def __post_init__(self):
        if self.injuries is None:
            self.injuries = []
        if self.news is None:
            self.news = []

class OpenAIAgentRouter:
    """Routes prop analysis requests to specialized OpenAI Agents"""
    
    # Sport to Agent mapping - easily extensible
    SPORT_AGENT_MAP = {
        "basketball": "NBA_Analysis_Agent",
        "nba": "NBA_Analysis_Agent",
        "football": "NFL_Analysis_Agent", 
        "nfl": "NFL_Analysis_Agent",
        "college_football": "CFB_Analysis_Agent",
        "cfb": "CFB_Analysis_Agent",
        "soccer": "Soccer_Analysis_Agent",
        "hockey": "NHL_Analysis_Agent",
        "nhl": "NHL_Analysis_Agent",
        "baseball": "MLB_Analysis_Agent",
        "mlb": "MLB_Analysis_Agent",
        "csgo": "CSGO_Analysis_Agent",
        "cs": "CSGO_Analysis_Agent",
        "league_of_legends": "LoL_Analysis_Agent",
        "lol": "LoL_Analysis_Agent",
        "dota2": "Dota2_Analysis_Agent",
        "dota": "Dota2_Analysis_Agent",
        "valorant": "Valorant_Analysis_Agent",
        "val": "Valorant_Analysis_Agent",
        "overwatch": "Overwatch_Analysis_Agent",
        "ow": "Overwatch_Analysis_Agent",
        "tennis": "Tennis_Analysis_Agent",
        "golf": "Golf_Analysis_Agent"
    }
    
    # Agent Assistant IDs (would be stored in config/env in production)
    AGENT_ASSISTANT_IDS = {
        "NBA_Analysis_Agent": "asst_nba_analysis_001",
        "NFL_Analysis_Agent": "asst_nfl_analysis_001", 
        "CFB_Analysis_Agent": "asst_cfb_analysis_001",
        "Soccer_Analysis_Agent": "asst_soccer_analysis_001",
        "NHL_Analysis_Agent": "asst_nhl_analysis_001",
        "MLB_Analysis_Agent": "asst_mlb_analysis_001",
        "CSGO_Analysis_Agent": "asst_csgo_analysis_001",
        "LoL_Analysis_Agent": "asst_lol_analysis_001",
        "Dota2_Analysis_Agent": "asst_dota2_analysis_001",
        "Valorant_Analysis_Agent": "asst_valorant_analysis_001",
        "Overwatch_Analysis_Agent": "asst_overwatch_analysis_001",
        "Tennis_Analysis_Agent": "asst_tennis_analysis_001",
        "Golf_Analysis_Agent": "asst_golf_analysis_001"
    }
    
    def __init__(self, api_key: str = None, enable_logging: bool = True):
        """Initialize the router with OpenAI API key"""
        # Load configuration
        self.config_manager = get_config_manager()
        
        # Initialize comprehensive logging and monitoring
        self.agent_logger = AgentLogger(
            log_dir="logs/agent_router"
        )
        
        # Initialize error handling
        self.error_handler = AgentErrorHandler()
        
        # Initialize monitoring
        self.monitor = AgentMonitor()
        
        # Set API key from parameter, config, or environment
        self.api_key = (api_key or 
                       self.config_manager.router_config.openai_api_key or 
                       os.getenv('OPENAI_API_KEY'))
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.client = OpenAI(api_key=self.api_key)
        self.enable_logging = enable_logging
        
        # Load dynamic mappings from config
        self.SPORT_AGENT_MAP = self.config_manager.get_sport_agent_mapping()
        self.AGENT_ASSISTANT_IDS = self.config_manager.get_assistant_id_mapping()
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.openai_usage_count = 0
        self.local_usage_count = 0
        
        # Log initialization
        self.agent_logger.log_agent_request(
            agent_name="agent_router",
            sport="system",
            request_data={
                "action": "initialization",
                "total_agents": len(self.AGENT_ASSISTANT_IDS),
                "supported_sports": len(self.SPORT_AGENT_MAP)
            }
        )
        
        if enable_logging:
            logger.info("OpenAI Agent Router initialized successfully")
            logger.info(f"Loaded {len(self.AGENT_ASSISTANT_IDS)} agent configurations")
    
    def detect_sport(self, prop_data: Union[PropData, dict, str]) -> str:
        """Detect sport from prop data or context"""
        with self.monitor.track_operation("sport_detection"):
            if isinstance(prop_data, PropData):
                detected_sport = prop_data.sport.lower()
            elif isinstance(prop_data, dict):
                detected_sport = prop_data.get('sport', '').lower()
            elif isinstance(prop_data, str):
                # Simple keyword detection
                sport_keywords = {
                    'basketball': ['nba', 'basketball', 'points', 'rebounds', 'assists'],
                    'football': ['nfl', 'football', 'touchdown', 'yards', 'passing'],
                    'soccer': ['soccer', 'football', 'goals', 'shots', 'cards'],
                    'baseball': ['mlb', 'baseball', 'hits', 'runs', 'strikeouts'],
                    'hockey': ['nhl', 'hockey', 'goals', 'assists', 'saves'],
                    'csgo': ['csgo', 'counter-strike', 'kills', 'maps', 'rounds'],
                    'tennis': ['tennis', 'sets', 'games', 'aces']
                }
                
                text_lower = prop_data.lower()
                detected_sport = 'unknown'
                for sport, keywords in sport_keywords.items():
                    if any(keyword in text_lower for keyword in keywords):
                        detected_sport = sport
                        break
            else:
                detected_sport = 'unknown'
            
            # Log sport detection
            self.agent_logger.log_agent_request(
                agent_name="sport_detector",
                sport=detected_sport,
                request_data={
                    "input_type": type(prop_data).__name__,
                    "detected_sport": detected_sport
                }
            )
            
            return detected_sport
    
    def get_agent_for_sport(self, sport: str) -> Optional[str]:
        """Get the appropriate agent name for a sport"""
        sport_key = sport.lower().strip()
        agent_name = self.SPORT_AGENT_MAP.get(sport_key)
        
        if not agent_name:
            logger.warning(f"No agent found for sport: {sport}")
            return None
            
        return agent_name
    
    def format_agent_input(self, 
                          props: List[PropData], 
                          stats: Optional[StatsData] = None,
                          context: Optional[ContextData] = None) -> Dict[str, Any]:
        """Format input data for OpenAI Agent"""
        
        formatted_input = {
            "analysis_type": "prop_betting_analysis",
            "timestamp": datetime.now().isoformat(),
            "props": []
        }
        
        # Format props data
        for prop in props:
            prop_dict = asdict(prop) if isinstance(prop, PropData) else prop
            formatted_input["props"].append(prop_dict)
        
        # Add stats if provided
        if stats:
            formatted_input["stats"] = asdict(stats) if isinstance(stats, StatsData) else stats
        
        # Add context if provided  
        if context:
            formatted_input["context"] = asdict(context) if isinstance(context, ContextData) else context
        
        return formatted_input
    
    def create_analysis_prompt(self, agent_input: Dict[str, Any], sport: str) -> str:
        """Create a structured prompt for the agent"""
        
        prompt_parts = [
            f"# {sport.upper()} Prop Betting Analysis Request",
            "",
            "## Props to Analyze:",
        ]
        
        for i, prop in enumerate(agent_input.get("props", []), 1):
            prompt_parts.extend([
                f"{i}. **Player**: {prop.get('player')}",
                f"   **Prop**: {prop.get('prop')}",
                f"   **Line**: {prop.get('line')}",
                f"   **Odds**: {prop.get('odds')}",
                f"   **Opponent**: {prop.get('opponent')}",
                f"   **Matchup**: {prop.get('matchup')}",
                ""
            ])
        
        # Add stats section if available
        if "stats" in agent_input:
            prompt_parts.extend([
                "## Statistical Data:",
                json.dumps(agent_input["stats"], indent=2),
                ""
            ])
        
        # Add context section if available
        if "context" in agent_input:
            prompt_parts.extend([
                "## Additional Context:",
                json.dumps(agent_input["context"], indent=2),
                ""
            ])
        
        prompt_parts.extend([
            "## Analysis Request:",
            "Please provide a comprehensive analysis including:",
            "1. Value assessment (positive/negative expected value)",
            "2. Statistical trends and patterns",
            "3. Situational factors and context",
            "4. Confidence level (1-100)",
            "5. Recommended action (bet/pass/watch)",
            "6. Key reasoning behind recommendation",
            "",
            "Format the response as structured JSON with clear sections."
        ])
        
        return "\n".join(prompt_parts)
    
    @with_error_handling()
    async def call_openai_agent(self, 
                               agent_name: str, 
                               agent_input: Dict[str, Any],
                               sport: str) -> Dict[str, Any]:
        """Call OpenAI Assistant API for analysis"""
        
        try:
            assistant_id = self.AGENT_ASSISTANT_IDS.get(agent_name)
            if not assistant_id:
                raise ValueError(f"No Assistant ID configured for agent: {agent_name}")
            
            # Create analysis prompt
            prompt = self.create_analysis_prompt(agent_input, sport)
            
            # Log agent request
            self.agent_logger.log_agent_request(
                agent_name=agent_name,
                sport=sport,
                request_data={
                    "assistant_id": assistant_id,
                    "prompt_length": len(prompt),
                    "props_count": len(agent_input.get("props", [])),
                    "has_stats": "stats" in agent_input,
                    "has_context": "context" in agent_input
                }
            )
            
            if self.enable_logging:
                logger.info(f"Calling {agent_name} for {sport} analysis")
                logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Create thread
            thread = self.client.beta.threads.create()
            
            # Add message to thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            
            # Wait for completion (in production, implement polling)
            import time
            while run.status in ['queued', 'in_progress']:
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get response messages
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                # Get the assistant's response
                assistant_response = messages.data[0].content[0].text.value
                
                # Log successful response
                self.agent_logger.log_agent_response(
                    agent_name=agent_name,
                    sport=sport,
                    response_data={
                        "status": "completed",
                        "response_length": len(assistant_response),
                        "thread_id": thread.id,
                        "run_id": run.id
                    },
                    success=True
                )
                
                if self.enable_logging:
                    logger.info(f"Successfully received analysis from {agent_name}")
                
                return {
                    "success": True,
                    "agent": agent_name,
                    "sport": sport,
                    "analysis": assistant_response,
                    "thread_id": thread.id,
                    "run_id": run.id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_msg = f"Agent run failed with status: {run.status}"
                
                # Log failed response
                self.agent_logger.log_agent_response(
                    agent_name=agent_name,
                    sport=sport,
                    response_data={"status": run.status, "error": error_msg},
                    success=False,
                    error_message=error_msg
                )
                
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "agent": agent_name,
                    "sport": sport
                }
                
        except Exception as e:
            error_msg = f"Error calling OpenAI agent {agent_name}: {str(e)}"
            
            # Log error
            self.agent_logger.log_agent_response(
                agent_name=agent_name,
                sport=sport,
                response_data={"error": str(e)},
                success=False,
                error_message=error_msg
            )
            
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "agent": agent_name,
                "sport": sport
            }
    
    @with_error_handling()
    def route_to_sport_agent(self, 
                           sport: str,
                           props: List[Union[PropData, dict]],
                           stats: Optional[Union[StatsData, dict]] = None,
                           context: Optional[Union[ContextData, dict]] = None) -> Dict[str, Any]:
        """Main routing function - routes props to appropriate sport agent"""
        
        try:
            # Normalize sport name and get agent
            sport_normalized = sport.lower().strip()
            agent_name = self.get_agent_for_sport(sport_normalized)
            
            if not agent_name:
                error_result = {
                    "success": False,
                    "error": f"No agent available for sport: {sport}",
                    "available_sports": list(set(self.SPORT_AGENT_MAP.keys()))
                }
                
                # Log routing failure
                self.agent_logger.log_agent_request(
                    agent_name="router",
                    sport=sport,
                    request_data={
                        "action": "routing_failed",
                        "reason": "no_agent_available",
                        "requested_sport": sport
                    }
                )
                
                return error_result
            
            # Convert props to PropData objects if needed
            prop_objects = []
            for prop in props:
                if isinstance(prop, dict):
                    prop_objects.append(PropData(**prop))
                elif isinstance(prop, PropData):
                    prop_objects.append(prop)
                else:
                    logger.warning(f"Invalid prop data type: {type(prop)}")
                    continue
            
            if not prop_objects:
                error_result = {
                    "success": False,
                    "error": "No valid props provided for analysis"
                }
                
                # Log validation failure
                self.agent_logger.log_agent_request(
                    agent_name="router",
                    sport=sport,
                    request_data={
                        "action": "validation_failed",
                        "reason": "no_valid_props",
                        "original_props_count": len(props)
                    }
                )
                
                return error_result
            
            # Format input for agent
            agent_input = self.format_agent_input(prop_objects, stats, context)
            
            # Log successful routing request
            self.agent_logger.log_agent_request(
                agent_name=agent_name,
                sport=sport,
                request_data={
                    "action": "routing_to_agent",
                    "props_count": len(prop_objects),
                    "has_stats": bool(stats),
                    "has_context": bool(context)
                }
            )
            
            if self.enable_logging:
                logger.info(f"Routing {len(prop_objects)} props to {agent_name} for {sport}")
            
            # Call the agent (sync wrapper for async)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.call_openai_agent(agent_name, agent_input, sport)
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error in routing to sport agent: {str(e)}"
            
            # Log routing error
            self.agent_logger.log_agent_response(
                agent_name="router",
                sport=sport,
                response_data={"error": str(e)},
                success=False,
                error_message=error_msg
            )
            
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "sport": sport
            }
    
    def get_supported_sports(self) -> List[str]:
        """Get list of supported sports"""
        return sorted(list(set(self.SPORT_AGENT_MAP.keys())))
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all configured agents"""
        status = {
            "total_agents": len(self.AGENT_ASSISTANT_IDS),
            "supported_sports": self.get_supported_sports(),
            "agent_mapping": self.SPORT_AGENT_MAP,
            "assistant_ids_configured": len(self.AGENT_ASSISTANT_IDS),
            "api_key_configured": bool(self.api_key),
            "monitoring": {
                "monitor_initialized": self.monitor is not None,
                "data_directory": str(self.monitor.data_dir) if self.monitor else None
            },
            "logging": {
                "log_directory": str(self.agent_logger.log_dir),
                "log_level": self.agent_logger.log_level.value,
                "max_log_files": self.agent_logger.max_log_files
            }
        }
        return status

# Factory function for easy import
def create_agent_router(api_key: str = None, enable_logging: bool = True) -> OpenAIAgentRouter:
    """Create and return an OpenAI Agent Router instance"""
    return OpenAIAgentRouter(api_key=api_key, enable_logging=enable_logging)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    router = create_agent_router()
    
    # Print supported sports
    print("Supported Sports:")
    for sport in router.get_supported_sports():
        print(f"  - {sport}")
    
    # Print agent status
    print("\nAgent Status:")
    status = router.get_agent_status()
    for key, value in status.items():
        print(f"  {key}: {value}")