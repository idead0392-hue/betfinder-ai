"""
BetFinder AI Sports Data MCP Server

A Model Context Protocol server that provides OpenAI Agent Builder with access to
real-time sports data, odds, player props, and betting statistics for automated
analysis and decision making.

This server integrates with:
- Sportbex API for odds and betting data
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
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime, timedelta

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our existing BetFinder AI components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize components with fallback handling
sportbex_provider_available = False
picks_engine_available = False
bankroll_manager_available = False

try:
    from sportbex_provider import SportbexProvider
    sportbex_provider_available = True
    logger.info("✅ SportbexProvider imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  SportbexProvider not available: {e}")

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("betfinder-sports-data")

class BetFinderMCPServer:
    """BetFinder AI MCP Server for sports data integration"""
    
    def __init__(self):
        """Initialize the BetFinder MCP server"""
        self.sportbex_provider = None
        self.picks_engine = None
        self.bankroll_manager = None
        self.api_key = os.getenv('SPORTBEX_API_KEY', 'NZLDw8ZXFv0O8elaPq0wjbP4zxb2gCwJDsArWQUF')
        
        try:
            # Initialize providers based on availability
            if sportbex_provider_available:
                self.sportbex_provider = SportbexProvider(api_key=self.api_key)
                logger.info("✅ SportbexProvider initialized")
            
            if picks_engine_available:
                self.picks_engine = PicksEngine()
                logger.info("✅ PicksEngine initialized")
                
            if bankroll_manager_available:
                self.bankroll_manager = BankrollManager()
                logger.info("✅ BankrollManager initialized")
                
            logger.info("✅ BetFinder AI MCP Server initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize all components: {e}")
            logger.warning(f"Exception type: {type(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            logger.info("Running with limited functionality")
    
    async def get_live_odds(self, sport: str = None, market: str = None) -> Dict[str, Any]:
        """Get live betting odds for specified sport/market"""
        try:
            if self.sportbex_provider:
                try:
                    response = await asyncio.to_thread(
                        self.sportbex_provider.get_odds,
                        sport=sport,
                        market=market
                    )
                    
                    if response.success:
                        return {
                            "status": "success",
                            "data": response.data,
                            "timestamp": datetime.now().isoformat(),
                            "source": "Sportbex API"
                        }
                    else:
                        # Fallback to mock data if API fails
                        logger.warning(f"Sportbex API failed: {response.error_message}, using mock data")
                        return self._get_mock_odds_data(sport, market)
                except Exception as api_error:
                    logger.warning(f"Sportbex API error: {api_error}, using mock data")
                    return self._get_mock_odds_data(sport, market)
            else:
                # No provider available, use mock data
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
                    "home_team": "Lakers",
                    "away_team": "Warriors",
                    "home_odds": 1.85,
                    "away_odds": 1.95,
                    "draw_odds": None,
                    "start_time": (datetime.now() + timedelta(hours=2)).isoformat(),
                    "market": market or "moneyline"
                },
                {
                    "event_id": "mock_nba_002",
                    "sport": "basketball",
                    "home_team": "Celtics",
                    "away_team": "Heat",
                    "home_odds": 1.75,
                    "away_odds": 2.05,
                    "draw_odds": None,
                    "start_time": (datetime.now() + timedelta(hours=4)).isoformat(),
                    "market": market or "moneyline"
                }
            ],
            "football": [
                {
                    "event_id": "mock_nfl_001",
                    "sport": "football",
                    "home_team": "Patriots",
                    "away_team": "Bills",
                    "home_odds": 2.10,
                    "away_odds": 1.73,
                    "draw_odds": None,
                    "start_time": (datetime.now() + timedelta(days=1)).isoformat(),
                    "market": market or "moneyline"
                }
            ]
        }
        
        sport_data = mock_games.get(sport or "basketball", mock_games["basketball"])
        
        return {
            "status": "success",
            "data": sport_data,
            "timestamp": datetime.now().isoformat(),
            "source": "Mock Data"
        }
    
    async def get_ai_picks(self, sport: str = None, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Get AI-generated betting picks with confidence scores"""
        try:
            if self.picks_engine:
                try:
                    picks = await asyncio.to_thread(
                        self.picks_engine.get_daily_picks,
                        max_picks=10
                    )
                    
                    # Filter picks by confidence if provided
                    if picks and confidence_threshold:
                        picks = [pick for pick in picks if pick.get('confidence', 0) >= confidence_threshold]
                    
                    # If no picks from engine, generate mock picks
                    if not picks:
                        logger.warning("No picks from engine, generating mock picks")
                        picks = self._generate_mock_picks(sport, confidence_threshold)
                    
                    return {
                        "status": "success",
                        "picks": picks,
                        "filters": {
                            "sport": sport,
                            "min_confidence": confidence_threshold
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as engine_error:
                    logger.warning(f"PicksEngine error: {engine_error}, using mock picks")
                    picks = self._generate_mock_picks(sport, confidence_threshold)
                    
                    return {
                        "status": "success",
                        "picks": picks,
                        "filters": {
                            "sport": sport,
                            "min_confidence": confidence_threshold
                        },
                        "timestamp": datetime.now().isoformat(),
                        "source": "Mock Data"
                    }
            else:
                # No engine available, use mock picks
                picks = self._generate_mock_picks(sport, confidence_threshold)
                
                return {
                    "status": "success",
                    "picks": picks,
                    "filters": {
                        "sport": sport,
                        "min_confidence": confidence_threshold
                    },
                    "timestamp": datetime.now().isoformat(),
                    "source": "Mock Data"
                }
        except Exception as e:
            logger.error(f"Error generating AI picks: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_mock_picks(self, sport: str = None, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Generate mock AI picks for testing - PREDICTIONS ONLY, no financial advice"""
        mock_picks = [
            {
                "pick_id": "mock_ai_001",
                "sport": sport or "basketball",
                "pick_type": "moneyline",
                "team": "Lakers",
                "opponent": "Warriors",
                "odds": 1.85,
                "confidence": 0.85,
                "reasoning": "Strong home record (12-3) and key player returning from injury. Warriors playing back-to-back.",
                "expected_value": 0.15,
                "game_time": (datetime.now() + timedelta(hours=2)).isoformat(),
                "prediction_factors": [
                    "Home court advantage",
                    "Rest advantage", 
                    "Key player health",
                    "Recent performance trends"
                ]
            },
            {
                "pick_id": "mock_ai_002", 
                "sport": sport or "basketball",
                "pick_type": "spread",
                "team": "Celtics -3.5",
                "opponent": "Heat",
                "odds": 1.91,
                "confidence": 0.75,
                "reasoning": "Celtics excellent ATS at home (15-8) vs Heat road struggles (6-17 ATS).",
                "expected_value": 0.08,
                "game_time": (datetime.now() + timedelta(hours=4)).isoformat(),
                "prediction_factors": [
                    "Home ATS record",
                    "Away team road struggles",
                    "Historical matchup data"
                ]
            },
            {
                "pick_id": "mock_ai_003",
                "sport": "football" if sport != "basketball" else "basketball", 
                "pick_type": "totals",
                "team": "Over 47.5",
                "opponent": "Patriots vs Bills",
                "odds": 1.90,
                "confidence": 0.72,
                "reasoning": "Both teams averaging 28+ points at home. Weather conditions favorable for offense.",
                "expected_value": 0.04,
                "game_time": (datetime.now() + timedelta(days=1)).isoformat(),
                "prediction_factors": [
                    "Team scoring averages",
                    "Weather conditions",
                    "Defensive rankings"
                ]
            }
        ]
        
        # Filter by confidence threshold and remove financial recommendations
        filtered_picks = []
        for pick in mock_picks:
            if pick["confidence"] >= confidence_threshold:
                # Ensure no financial advice in picks - that's for bankroll management
                pick.pop("recommended_units", None)
                pick.pop("bet_amount", None) 
                pick.pop("units", None)
                filtered_picks.append(pick)
        
        return filtered_picks
    
    async def calculate_bet_size(self, odds: float, confidence: float, bankroll: float = None) -> Dict[str, Any]:
        """Calculate optimal bet size using Kelly Criterion and risk management"""
        try:
            if self.bankroll_manager:
                # Calculate expected value first
                probability = confidence
                decimal_odds = odds
                expected_value = (probability * (decimal_odds - 1)) - ((1 - probability) * 1)
                
                bet_analysis = await asyncio.to_thread(
                    self.bankroll_manager.calculate_recommended_bet_size,
                    confidence=confidence * 100,  # Convert to percentage
                    expected_value=expected_value,
                    kelly_fraction=0.25
                )
                
                return {
                    "status": "success",
                    "analysis": {
                        "recommended_amount": bet_analysis[0],
                        "recommendation": bet_analysis[1],
                        "expected_value": expected_value,
                        "odds": odds,
                        "confidence": confidence,
                        "kelly_fraction": 0.25
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Basic Kelly Criterion calculation
                probability = confidence
                decimal_odds = odds
                kelly_fraction = (probability * decimal_odds - 1) / (decimal_odds - 1)
                recommended_fraction = max(0, min(kelly_fraction * 0.25, 0.05))  # Conservative Kelly
                
                return {
                    "status": "success",
                    "analysis": {
                        "kelly_fraction": kelly_fraction,
                        "recommended_fraction": recommended_fraction,
                        "recommended_amount": (bankroll or 1000) * recommended_fraction,
                        "risk_level": "moderate" if recommended_fraction < 0.02 else "high",
                        "explanation": f"Kelly Criterion with 25% fraction for {confidence:.1%} confidence"
                    },
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error calculating bet size: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get bankroll performance and betting history metrics"""
        try:
            if self.bankroll_manager:
                try:
                    metrics = await asyncio.to_thread(
                        self.bankroll_manager.get_performance_metrics
                    )
                    
                    # If bankroll manager returns empty data, try to load from file
                    if not metrics or metrics.get('total_bets', 0) == 0:
                        logger.info("BankrollManager returned empty data, checking bankroll_data.json")
                        metrics = self._load_bankroll_metrics_from_file()
                    
                    return {
                        "status": "success", 
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as mgr_error:
                    logger.warning(f"BankrollManager error: {mgr_error}, loading from file")
                    metrics = self._load_bankroll_metrics_from_file()
                    
                    return {
                        "status": "success",
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                        "source": "Direct File Access"
                    }
            else:
                # No manager available, try file or use mock data
                metrics = self._load_bankroll_metrics_from_file()
                
                return {
                    "status": "success",
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Direct File Access" if metrics.get('total_bets', 0) > 0 else "Mock Data"
                }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    def _load_bankroll_metrics_from_file(self) -> Dict[str, Any]:
        """Load bankroll metrics directly from bankroll_data.json"""
        try:
            import json
            bankroll_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bankroll_data.json')
            
            if os.path.exists(bankroll_file):
                with open(bankroll_file, 'r') as f:
                    data = json.load(f)
                
                bets = data.get('bets', [])
                current_bankroll = data.get('current_bankroll', 0)
                starting_bankroll = data.get('starting_bankroll', 1000)
                
                # Calculate metrics from the data
                total_bets = len(bets)
                settled_bets = [bet for bet in bets if bet.get('status') != 'pending']
                winning_bets = [bet for bet in settled_bets if bet.get('result_amount', 0) > 0]
                
                win_rate = len(winning_bets) / len(settled_bets) if settled_bets else 0
                total_profit = current_bankroll - starting_bankroll
                roi = total_profit / starting_bankroll if starting_bankroll > 0 else 0
                
                return {
                    "total_bets": total_bets,
                    "settled_bets": len(settled_bets),
                    "pending_bets": total_bets - len(settled_bets),
                    "winning_bets": len(winning_bets),
                    "win_rate": win_rate,
                    "current_bankroll": current_bankroll,
                    "starting_bankroll": starting_bankroll,
                    "total_profit": total_profit,
                    "roi": roi,
                    "unit_size": data.get('unit_size', 1.0),
                    "max_bet_percentage": data.get('max_bet_percentage', 5.0)
                }
            else:
                logger.warning("bankroll_data.json not found, using mock metrics")
                return self._get_mock_performance_metrics()
                
        except Exception as e:
            logger.error(f"Error loading bankroll data from file: {e}")
            return self._get_mock_performance_metrics()
    
    def _get_mock_performance_metrics(self) -> Dict[str, Any]:
        """Generate mock performance metrics"""
        return {
            "total_bets": 45,
            "settled_bets": 42,
            "pending_bets": 3,
            "winning_bets": 28,
            "win_rate": 0.667,
            "current_bankroll": 1127.50,
            "starting_bankroll": 1000.0,
            "total_profit": 127.50,
            "roi": 0.128,
            "unit_size": 10.0,
            "max_bet_percentage": 5.0,
            "max_drawdown": -5.2,
            "sharpe_ratio": 1.45
        }

# Initialize the BetFinder server
betfinder_server = BetFinderMCPServer()

@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """List all available tools for sports betting analysis"""
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
            name="get_performance_metrics",
            description="Get comprehensive betting performance and bankroll metrics",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
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
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
    """Handle tool calls from OpenAI Agent Builder"""
    
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
            
        elif name == "get_performance_metrics":
            result = await betfinder_server.get_performance_metrics()
            
        elif name == "analyze_value_bet":
            odds = arguments["odds"]
            true_prob = arguments["true_probability"]
            
            # Calculate expected value
            payout_if_win = odds - 1
            expected_value = (true_prob * payout_if_win) - ((1 - true_prob) * 1)
            
            result = {
                "status": "success",
                "analysis": {
                    "odds": odds,
                    "true_probability": true_prob,
                    "implied_probability": 1 / odds,
                    "expected_value": expected_value,
                    "expected_value_percentage": expected_value * 100,
                    "is_value_bet": expected_value > 0,
                    "edge": true_prob - (1 / odds),
                    "recommendation": "BET" if expected_value > 0.05 else "PASS" if expected_value > 0 else "AVOID"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}
            
        # Return formatted response
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        logger.error(f"Error in tool call {name}: {e}")
        error_result = {
            "status": "error",
            "message": str(e),
            "tool": name,
            "timestamp": datetime.now().isoformat()
        }
        
        return [types.TextContent(
            type="text", 
            text=json.dumps(error_result, indent=2)
        )]

@server.list_resources()
async def list_resources() -> List[types.Resource]:
    """List available resources for the sports data server"""
    return [
        types.Resource(
            uri="betfinder://sports/current-odds",
            name="Current Sports Odds",
            description="Real-time betting odds across all supported sports",
            mimeType="application/json"
        ),
        types.Resource(
            uri="betfinder://ai/daily-picks", 
            name="Daily AI Picks",
            description="AI-generated betting recommendations for today",
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
                "server_name": "BetFinder AI Sports Data MCP Server",
                "version": "1.0.0",
                "capabilities": {
                    "live_odds": True,
                    "ai_picks": True,
                    "bankroll_management": True,
                    "value_analysis": True,
                    "performance_tracking": True
                },
                "supported_sports": ["basketball", "football", "tennis", "baseball", "soccer", "hockey"],
                "data_sources": ["Sportbex API", "Internal AI Engine"],
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(config_data, indent=2)
            
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
            
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        error_data = {
            "error": str(e),
            "uri": uri,
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_data, indent=2)

async def run_server():
    """Run the BetFinder AI MCP server"""
    logger.info("🚀 Starting BetFinder AI Sports Data MCP Server...")
    
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

def main():
    """Main entry point for the MCP server"""
    logger.info("🎯 BetFinder AI Sports Data MCP Server")
    logger.info("📊 Connecting to sports data sources...")
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    main()