"""
BetFinder AI Sports Data MCP Server

A Model Context Protocol server that provides OpenAI Agent Builder with access to
real-time sports data, player props, and betting statistics for automated
analysis and decision making.

This server integrates with:
- PrizePicks data for player props and betting data
- Enhanced AI picks engine for recommendations
- Bankroll management system for bet sizing
- Risk assessment algorithms

Author: BetFinder AI Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List
from datetime import datetime

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our existing BetFinder AI components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize components with fallback handling
picks_engine_available = False
bankroll_manager_available = False

# Strict mode: when true, never emit mock data; return empty/error structures instead
PRIZEPICKS_ONLY = os.environ.get("PRIZEPICKS_ONLY", "true").lower() in ("1", "true", "yes")
picks_engine_available = False
bankroll_manager_available = False

try:
    from picks_engine import PicksEngine
    picks_engine_available = True
    logger.info("✅ PicksEngine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  PicksEngine not available: {e}")

try:
    from bankroll_manager import BankrollManager
    bankroll_manager_available = True
    logger.info("✅ BankrollManager imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  BankrollManager not available: {e}")

# Initialize MCP server
server = Server("betfinder-sports-data")

class BetFinderMCPServer:
    """BetFinder AI MCP Server for sports data integration"""
    
    def __init__(self):
        """Initialize the BetFinder MCP server"""
        self.picks_engine = None
        self.bankroll_manager = None
        self.prizepicks_provider = None
        try:
            # Initialize providers based on availability            
            if picks_engine_available:
                self.picks_engine = PicksEngine()
                logger.info("✅ PicksEngine initialized")
            if bankroll_manager_available:
                self.bankroll_manager = BankrollManager()
                logger.info("✅ BankrollManager initialized")
            # PrizePicksProvider integration
            try:
                from prizepicks_provider import PrizePicksProvider
                self.prizepicks_provider = PrizePicksProvider()
                logger.info("✅ PrizePicksProvider initialized")
            except Exception as e:
                logger.warning(f"⚠️  PrizePicksProvider not available: {e}")
            logger.info("✅ BetFinder AI MCP Server initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize all components: {e}")
            logger.warning(f"Exception type: {type(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            logger.info("Running with limited functionality")
    
    async def get_live_odds(self, sport: str = None, market: str = None) -> Dict[str, Any]:
        """Get live betting odds for specified sport/market using PrizePicksProvider"""
        try:
            if self.prizepicks_provider:
                resp = self.prizepicks_provider.get_props(sport=sport, market=market)
                if hasattr(resp, 'success') and resp.success:
                    # Normalize to MCP format
                    props = resp.data.get('data') if isinstance(resp.data, dict) and 'data' in resp.data else resp.data
                    return {
                        "status": "success",
                        "data": props,
                        "timestamp": datetime.now().isoformat(),
                        "source": "PrizePicksProvider",
                        "total_props": len(props) if props else 0
                    }
                else:
                    return {
                        "status": "error",
                        "message": getattr(resp, 'error_message', 'Unknown error'),
                        "timestamp": datetime.now().isoformat(),
                        "source": "PrizePicksProvider",
                        "total_props": 0
                    }
            else:
                return self._get_mock_odds_data(sport, market)
        except Exception as e:
            logger.error(f"Error getting live odds: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_mock_odds_data(self, sport: str = None, market: str = None) -> Dict[str, Any]:
        """Generate mock odds data for testing"""
        mock_games = {
            "basketball": [
                {
                    "event_id": "mock_nba_001",
                    "sport": "basketball",
                    "player": "LeBron James",
                    "prop_type": "Points",
                    "line": 25.5,
                    "over_odds": 1.90,
                    "under_odds": 1.90,
                    "market": market or "props"
                }
            ],
            "football": [
                {
                    "event_id": "mock_nfl_001", 
                    "sport": "football",
                    "player": "Tom Brady",
                    "prop_type": "Passing Yards",
                    "line": 275.5,
                    "over_odds": 1.90,
                    "under_odds": 1.90,
                    "market": market or "props"
                }
            ]
        }
        
        games = mock_games.get(sport, mock_games["basketball"])
        
        return {
            "status": "success",
            "data": games,
            "timestamp": datetime.now().isoformat(),
            "source": "Mock Data",
            "note": "PrizePicks data not available, using mock data"
        }

    async def get_ai_picks(self, sport: str = None, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Get AI-generated betting recommendations with confidence scores"""
        try:
            if self.picks_engine:
                # Use real picks engine if available
                picks = await asyncio.to_thread(
                    self.picks_engine.generate_picks,
                    sport=sport,
                    confidence_threshold=confidence_threshold
                )
                return {
                    "status": "success",
                    "picks": picks,
                    "timestamp": datetime.now().isoformat(),
                    "source": "AI Picks Engine"
                }
            else:
                # Fallback behavior when engine missing
                if PRIZEPICKS_ONLY:
                    return {
                        "status": "no-engine",
                        "picks": [],
                        "timestamp": datetime.now().isoformat(),
                        "source": "AI Picks Engine",
                        "note": "AI Picks Engine not available and strict mode is enabled; mock picks disabled"
                    }
                return self._get_mock_ai_picks(sport, confidence_threshold)
        except Exception as e:
            logger.error(f"Error generating AI picks: {e}")
            return {"status": "error", "message": str(e)}

    def _get_mock_ai_picks(self, sport: str = None, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Generate mock AI picks for testing"""
        mock_picks = [
            {
                "pick_id": "mock_pick_001",
                "sport": sport or "basketball",
                "player": "Stephen Curry",
                "prop_type": "3-Pointers Made",
                "line": 4.5,
                "prediction": "Over",
                "confidence": 0.85,
                "reasoning": "Curry averaging 5.2 threes per game in last 10 games"
            },
            {
                "pick_id": "mock_pick_002", 
                "sport": sport or "football",
                "player": "Travis Kelce",
                "prop_type": "Receiving Yards",
                "line": 75.5,
                "prediction": "Over",
                "confidence": 0.78,
                "reasoning": "Favorable matchup against weak secondary"
            }
        ]
        
        # Filter by confidence threshold
        filtered_picks = [pick for pick in mock_picks if pick["confidence"] >= confidence_threshold]
        
        return {
            "status": "success",
            "picks": filtered_picks,
            "timestamp": datetime.now().isoformat(),
            "source": "Mock AI Engine",
            "note": "AI Picks Engine not available, using mock data"
        }

    async def calculate_bet_size(self, odds: float, confidence: float, bankroll: float = None) -> Dict[str, Any]:
        """Calculate optimal bet size using Kelly Criterion and risk management"""
        try:
            if self.bankroll_manager:
                bet_size = await asyncio.to_thread(
                    self.bankroll_manager.calculate_kelly_bet_size,
                    odds=odds,
                    confidence=confidence,
                    bankroll=bankroll
                )
                return {
                    "status": "success",
                    "recommended_bet_size": bet_size,
                    "kelly_percentage": bet_size / (bankroll or 1000) if bankroll else None,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Bankroll Manager"
                }
            else:
                # Simplified Kelly Criterion calculation
                implied_probability = 1 / odds
                edge = confidence - implied_probability
                kelly_fraction = edge / (odds - 1) if odds > 1 else 0
                
                # Conservative sizing - max 5% of bankroll
                kelly_fraction = min(kelly_fraction, 0.05)
                kelly_fraction = max(kelly_fraction, 0)  # No negative bets
                
                bet_size = (bankroll or 1000) * kelly_fraction
                
                return {
                    "status": "success",
                    "recommended_bet_size": round(bet_size, 2),
                    "kelly_percentage": round(kelly_fraction * 100, 2),
                    "timestamp": datetime.now().isoformat(),
                    "source": "Simplified Kelly Calculator",
                    "note": "Bankroll Manager not available, using simplified calculation"
                }
        except Exception as e:
            logger.error(f"Error calculating bet size: {e}")
            return {"status": "error", "message": str(e)}

    async def analyze_value_bet(self, odds: float, true_probability: float) -> Dict[str, Any]:
        """Analyze if a bet offers positive expected value"""
        try:
            implied_probability = 1 / odds
            expected_value = (true_probability * (odds - 1)) - (1 - true_probability)
            value_percentage = ((true_probability * odds) - 1) * 100
            
            is_value_bet = expected_value > 0
            
            return {
                "status": "success",
                "is_value_bet": is_value_bet,
                "expected_value": round(expected_value, 4),
                "value_percentage": round(value_percentage, 2),
                "implied_probability": round(implied_probability, 4),
                "true_probability": true_probability,
                "odds": odds,
                "recommendation": "BET" if is_value_bet else "PASS",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing value bet: {e}")
            return {"status": "error", "message": str(e)}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive betting performance and bankroll metrics"""
        try:
            if self.bankroll_manager:
                metrics = await asyncio.to_thread(
                    self.bankroll_manager.get_performance_summary
                )
                return {
                    "status": "success",
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Bankroll Manager"
                }
            else:
                # In strict mode, do not emit mock performance metrics
                if PRIZEPICKS_ONLY:
                    return {
                        "status": "no-manager",
                        "metrics": {},
                        "timestamp": datetime.now().isoformat(),
                        "source": "Bankroll Manager",
                        "note": "Bankroll Manager not available and strict mode is enabled; mock metrics disabled"
                    }
                # Mock performance metrics (non-strict/dev mode)
                return {
                    "status": "success",
                    "metrics": {
                        "total_bets": 127,
                        "winning_bets": 68,
                        "losing_bets": 59,
                        "win_rate": 53.5,
                        "total_wagered": 12750.00,
                        "total_winnings": 13421.50,
                        "net_profit": 671.50,
                        "roi": 5.3,
                        "current_bankroll": 10671.50,
                        "largest_win": 385.00,
                        "largest_loss": -250.00,
                        "average_bet_size": 100.39,
                        "longest_winning_streak": 7,
                        "longest_losing_streak": 4
                    },
                    "timestamp": datetime.now().isoformat(),
                    "source": "Mock Performance Data",
                    "note": "Bankroll Manager not available, using mock data"
                }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"status": "error", "message": str(e)}


# Initialize the BetFinder server
betfinder_server = BetFinderMCPServer()

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools for the MCP client"""
    return [
        types.Tool(
            name="get_live_odds",
            description="Get real-time betting odds for sports events",
            inputSchema={
                "type": "object",
                "properties": {
                    "sport": {
                        "type": "string",
                        "description": "Sport type (basketball, football, tennis, etc.)",
                        "enum": ["basketball", "football", "tennis", "baseball", "soccer", "hockey"]
                    },
                    "market": {
                        "type": "string", 
                        "description": "Betting market type (moneyline, spread, totals, props)",
                        "enum": ["moneyline", "spread", "totals", "props"]
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_ai_picks",
            description="Get AI-generated betting recommendations with confidence scores",
            inputSchema={
                "type": "object",
                "properties": {
                    "sport": {
                        "type": "string",
                        "description": "Sport to analyze",
                        "enum": ["basketball", "football", "tennis", "baseball", "soccer", "hockey"]
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence level for picks (0.0 to 1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="calculate_bet_size",
            description="Calculate optimal bet size using Kelly Criterion and risk management",
            inputSchema={
                "type": "object",
                "properties": {
                    "odds": {
                        "type": "number",
                        "description": "Decimal odds for the bet",
                        "minimum": 1.01
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level in the bet (0.0 to 1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "bankroll": {
                        "type": "number",
                        "description": "Current bankroll amount (optional)",
                        "minimum": 0
                    }
                },
                "required": ["odds", "confidence"]
            }
        ),
        types.Tool(
            name="analyze_value_bet",
            description="Analyze if a bet offers positive expected value",
            inputSchema={
                "type": "object",
                "properties": {
                    "odds": {
                        "type": "number",
                        "description": "Decimal odds offered",
                        "minimum": 1.01
                    },
                    "true_probability": {
                        "type": "number",
                        "description": "Your estimated true probability of winning",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["odds", "true_probability"]
            }
        ),
        types.Tool(
            name="get_performance_metrics",
            description="Get comprehensive betting performance and bankroll metrics",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls from the MCP client"""
    
    try:
        if name == "get_live_odds":
            result = await betfinder_server.get_live_odds(
                sport=arguments.get("sport"),
                market=arguments.get("market")
            )
        elif name == "get_ai_picks":
            result = await betfinder_server.get_ai_picks(
                sport=arguments.get("sport"),
                confidence_threshold=arguments.get("confidence_threshold", 0.7)
            )
        elif name == "calculate_bet_size":
            result = await betfinder_server.calculate_bet_size(
                odds=arguments["odds"],
                confidence=arguments["confidence"],
                bankroll=arguments.get("bankroll")
            )
        elif name == "analyze_value_bet":
            result = await betfinder_server.analyze_value_bet(
                odds=arguments["odds"],
                true_probability=arguments["true_probability"]
            )
        elif name == "get_performance_metrics":
            result = await betfinder_server.get_performance_metrics()
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """List available resources for the MCP client"""
    return [
        types.Resource(
            uri="betfinder://sports/current-odds",
            name="Current Sports Odds",
            description="Real-time odds and props data",
            mimeType="application/json"
        ),
        types.Resource(
            uri="betfinder://ai/daily-picks", 
            name="Daily AI Picks",
            description="AI-generated betting recommendations",
            mimeType="application/json"
        ),
        types.Resource(
            uri="betfinder://bankroll/status",
            name="Bankroll Status",
            description="Current bankroll and performance metrics",
            mimeType="application/json"
        ),
        types.Resource(
            uri="betfinder://config/settings",
            name="Server Configuration",
            description="MCP server configuration and capabilities",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content based on URI"""
    
    try:
        if uri == "betfinder://sports/current-odds":
            odds_data = await betfinder_server.get_live_odds()
            return json.dumps(odds_data, indent=2, default=str)
            
        elif uri == "betfinder://ai/daily-picks":
            picks_data = await betfinder_server.get_ai_picks()
            return json.dumps(picks_data, indent=2, default=str)
            
        elif uri == "betfinder://bankroll/status":
            metrics_data = await betfinder_server.get_performance_metrics()
            return json.dumps(metrics_data, indent=2, default=str)
            
        elif uri == "betfinder://config/settings":
            config_data = {
                "server_name": "BetFinder AI MCP Server",
                "version": "1.0.0",
                "capabilities": {
                    "tools": ["get_live_odds", "get_ai_picks", "calculate_bet_size", "analyze_value_bet", "get_performance_metrics"],
                    "resources": ["current-odds", "daily-picks", "bankroll-status", "settings"],
                    "data_sources": ["PrizePicks Data", "Internal AI Engine"],
                    "sports_supported": ["basketball", "football", "tennis", "baseball", "soccer", "hockey"]
                },
                "status": "operational",
                "last_updated": datetime.now().isoformat()
            }
            return json.dumps(config_data, indent=2, default=str)
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
            
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        return json.dumps({"error": str(e)}, indent=2)

async def main():
    """Main entry point for the MCP server"""
    logger.info("🚀 Starting BetFinder AI MCP Server...")
    
    # Use stdio transport for VS Code and other MCP clients
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="betfinder-sports-data",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())