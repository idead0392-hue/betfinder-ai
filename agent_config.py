"""
Configuration manager for OpenAI Agent Router
Handles API keys, Assistant IDs, and routing settings
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for a specific OpenAI Assistant"""
    name: str
    assistant_id: str
    sport: str
    description: str
    enabled: bool = True
    max_tokens: int = 4000
    temperature: float = 0.3
    model: str = "gpt-4-1106-preview"

@dataclass
class RouterConfig:
    """Main router configuration"""
    openai_api_key: str = ""
    default_timeout: int = 30
    max_retries: int = 3
    enable_fallback: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"

class ConfigManager:
    """Manages configuration for OpenAI Agent Router"""
    
    def __init__(self, config_file: str = "agent_config.json"):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.router_config = RouterConfig()
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # Load configuration
        self.load_config()
        
        # Set up default agent configurations if not loaded
        if not self.agent_configs:
            self._setup_default_agents()
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load router config
                if 'router_config' in config_data:
                    router_data = config_data['router_config']
                    self.router_config = RouterConfig(**router_data)
                
                # Load agent configs
                if 'agent_configs' in config_data:
                    for agent_name, agent_data in config_data['agent_configs'].items():
                        self.agent_configs[agent_name] = AgentConfig(**agent_data)
                
                logger.info(f"Configuration loaded from {self.config_file}")
                return True
            else:
                logger.info(f"Config file {self.config_file} not found, using defaults")
                return False
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            config_data = {
                'router_config': asdict(self.router_config),
                'agent_configs': {
                    name: asdict(config) 
                    for name, config in self.agent_configs.items()
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def _setup_default_agents(self):
        """Set up default agent configurations"""
        default_agents = [
            {
                "name": "NBA_Analysis_Agent",
                "assistant_id": os.getenv("NBA_ASSISTANT_ID", "asst_nba_analysis_001"),
                "sport": "basketball",
                "description": "Specialized NBA prop betting analysis with advanced statistics"
            },
            {
                "name": "NFL_Analysis_Agent", 
                "assistant_id": os.getenv("NFL_ASSISTANT_ID", "asst_nfl_analysis_001"),
                "sport": "football",
                "description": "NFL prop analysis with weather, matchup, and situational factors"
            },
            {
                "name": "CFB_Analysis_Agent",
                "assistant_id": os.getenv("CFB_ASSISTANT_ID", "asst_cfb_analysis_001"),
                "sport": "college_football",
                "description": "College football prop analysis with recruiting and team dynamics"
            },
            {
                "name": "Soccer_Analysis_Agent",
                "assistant_id": os.getenv("SOCCER_ASSISTANT_ID", "asst_soccer_analysis_001"),
                "sport": "soccer",
                "description": "Soccer/Football prop analysis with international and league context"
            },
            {
                "name": "NHL_Analysis_Agent",
                "assistant_id": os.getenv("NHL_ASSISTANT_ID", "asst_nhl_analysis_001"),
                "sport": "hockey",
                "description": "NHL prop analysis with ice conditions and playoff implications"
            },
            {
                "name": "MLB_Analysis_Agent",
                "assistant_id": os.getenv("MLB_ASSISTANT_ID", "asst_mlb_analysis_001"),
                "sport": "baseball",
                "description": "MLB prop analysis with weather, ballpark factors, and pitcher matchups"
            },
            {
                "name": "CSGO_Analysis_Agent",
                "assistant_id": os.getenv("CSGO_ASSISTANT_ID", "asst_csgo_analysis_001"),
                "sport": "csgo",
                "description": "CS:GO esports analysis with map pools, team compositions, and meta"
            },
            {
                "name": "LoL_Analysis_Agent",
                "assistant_id": os.getenv("LOL_ASSISTANT_ID", "asst_lol_analysis_001"),
                "sport": "league_of_legends",
                "description": "League of Legends analysis with patch meta, draft priority, and team synergy"
            },
            {
                "name": "Dota2_Analysis_Agent",
                "assistant_id": os.getenv("DOTA2_ASSISTANT_ID", "asst_dota2_analysis_001"),
                "sport": "dota2",
                "description": "Dota 2 analysis with hero combinations, farm priority, and objective control"
            },
            {
                "name": "Valorant_Analysis_Agent",
                "assistant_id": os.getenv("VALORANT_ASSISTANT_ID", "asst_valorant_analysis_001"),
                "sport": "valorant",
                "description": "Valorant analysis with agent compositions, map control, and round economics"
            },
            {
                "name": "Overwatch_Analysis_Agent",
                "assistant_id": os.getenv("OVERWATCH_ASSISTANT_ID", "asst_overwatch_analysis_001"),
                "sport": "overwatch",
                "description": "Overwatch analysis with team compositions, map types, and ultimate timing"
            },
            {
                "name": "Tennis_Analysis_Agent",
                "assistant_id": os.getenv("TENNIS_ASSISTANT_ID", "asst_tennis_analysis_001"),
                "sport": "tennis",
                "description": "Tennis analysis with surface preferences, head-to-head, and physical condition"
            },
            {
                "name": "Golf_Analysis_Agent",
                "assistant_id": os.getenv("GOLF_ASSISTANT_ID", "asst_golf_analysis_001"),
                "sport": "golf",
                "description": "Golf analysis with course conditions, weather, and recent form"
            }
        ]
        
        for agent_data in default_agents:
            self.agent_configs[agent_data["name"]] = AgentConfig(**agent_data)
        
        # Set API key from environment
        self.router_config.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        logger.info(f"Set up {len(default_agents)} default agent configurations")
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for specific agent"""
        return self.agent_configs.get(agent_name)
    
    def get_assistant_id(self, agent_name: str) -> Optional[str]:
        """Get OpenAI Assistant ID for agent"""
        config = self.get_agent_config(agent_name)
        return config.assistant_id if config else None
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if agent is enabled"""
        config = self.get_agent_config(agent_name)
        return config.enabled if config else False
    
    def enable_agent(self, agent_name: str) -> bool:
        """Enable an agent"""
        if agent_name in self.agent_configs:
            self.agent_configs[agent_name].enabled = True
            return True
        return False
    
    def disable_agent(self, agent_name: str) -> bool:
        """Disable an agent"""
        if agent_name in self.agent_configs:
            self.agent_configs[agent_name].enabled = False
            return True
        return False
    
    def add_agent(self, config: AgentConfig) -> bool:
        """Add new agent configuration"""
        try:
            self.agent_configs[config.name] = config
            logger.info(f"Added agent configuration: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding agent config: {e}")
            return False
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove agent configuration"""
        if agent_name in self.agent_configs:
            del self.agent_configs[agent_name]
            logger.info(f"Removed agent configuration: {agent_name}")
            return True
        return False
    
    def update_assistant_id(self, agent_name: str, assistant_id: str) -> bool:
        """Update Assistant ID for agent"""
        if agent_name in self.agent_configs:
            self.agent_configs[agent_name].assistant_id = assistant_id
            logger.info(f"Updated Assistant ID for {agent_name}: {assistant_id}")
            return True
        return False
    
    def get_sport_agent_mapping(self) -> Dict[str, str]:
        """Get mapping of sports to agent names"""
        mapping = {}
        for agent_name, config in self.agent_configs.items():
            if config.enabled:
                sport = config.sport
                # Add common aliases
                mapping[sport] = agent_name
                
                # Add sport-specific aliases
                if sport == "basketball":
                    mapping["nba"] = agent_name
                elif sport == "football":
                    mapping["nfl"] = agent_name
                elif sport == "hockey":
                    mapping["nhl"] = agent_name
                elif sport == "baseball":
                    mapping["mlb"] = agent_name
                elif sport == "csgo":
                    mapping["cs"] = agent_name
                elif sport == "league_of_legends":
                    mapping["lol"] = agent_name
                elif sport == "dota2":
                    mapping["dota"] = agent_name
                elif sport == "valorant":
                    mapping["val"] = agent_name
                elif sport == "overwatch":
                    mapping["ow"] = agent_name
        
        return mapping
    
    def get_assistant_id_mapping(self) -> Dict[str, str]:
        """Get mapping of agent names to Assistant IDs"""
        return {
            name: config.assistant_id 
            for name, config in self.agent_configs.items() 
            if config.enabled and config.assistant_id
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "agent_count": len(self.agent_configs),
            "enabled_agents": len([c for c in self.agent_configs.values() if c.enabled])
        }
        
        # Check API key
        if not self.router_config.openai_api_key:
            validation_results["errors"].append("OpenAI API key not configured")
            validation_results["valid"] = False
        
        # Check each agent
        for name, config in self.agent_configs.items():
            if config.enabled:
                if not config.assistant_id:
                    validation_results["errors"].append(f"Agent {name} missing Assistant ID")
                    validation_results["valid"] = False
                elif config.assistant_id.startswith("asst_") and "001" in config.assistant_id:
                    validation_results["warnings"].append(f"Agent {name} using placeholder Assistant ID")
        
        return validation_results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        enabled_agents = [name for name, config in self.agent_configs.items() if config.enabled]
        disabled_agents = [name for name, config in self.agent_configs.items() if not config.enabled]
        
        return {
            "router_config": asdict(self.router_config),
            "total_agents": len(self.agent_configs),
            "enabled_agents": enabled_agents,
            "disabled_agents": disabled_agents,
            "sports_covered": list(set(config.sport for config in self.agent_configs.values() if config.enabled)),
            "config_file": self.config_file,
            "validation": self.validate_config()
        }

# Global config manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    return config_manager

def setup_agent_config(config_file: str = None) -> ConfigManager:
    """Set up agent configuration with custom file"""
    global config_manager
    if config_file:
        config_manager = ConfigManager(config_file)
    return config_manager

# Example usage and testing
if __name__ == "__main__":
    # Test configuration manager
    config_mgr = ConfigManager("test_config.json")
    
    print("Configuration Summary:")
    summary = config_mgr.get_config_summary()
    print(json.dumps(summary, indent=2))
    
    print("\nValidation Results:")
    validation = config_mgr.validate_config()
    print(json.dumps(validation, indent=2))
    
    # Save configuration
    config_mgr.save_config()
    print(f"\nConfiguration saved to {config_mgr.config_file}")