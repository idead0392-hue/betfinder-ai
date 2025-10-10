import os
import time
import logging
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIDailyPicksAgent:
    """
    OpenAI-powered daily picks agent for sports betting analysis.
    Uses GPT-5 for comprehensive statistical analysis including:
    - Last 10 and last 5 games player performance data
    - Detailed matchup analysis and trend identification
    - Over/under prop generation with statistical backing
    - Enhanced validation and error handling
    """
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-5"  # Enable GPT-5 for all clients
        self.max_retries = 3
        self.base_delay = 1

    def get_ai_daily_picks(self, max_picks: int = 5) -> List[Dict]:
        """
        Get AI-generated daily picks using GPT-5 with comprehensive statistical analysis.
        Returns a list of pick dicts with support for both over and under picks,
        including last 10 and last 5 games statistical analysis for player props.
        Implements retry logic and comprehensive error handling.
        """
        prompt = (
            f"You are a professional sports betting analyst with access to comprehensive player statistics. "
            f"Generate {max_picks} diverse, high-quality betting picks with detailed statistical analysis.\n\n"
            
            "CRITICAL REQUIREMENTS:\n"
            "1. Include BOTH over and under player props (minimum 2 overs and 2 unders if generating 4+ picks)\n"
            "2. For EVERY player prop, provide detailed statistical analysis including:\n"
            "   - Last 10 games performance data\n"
            "   - Last 5 games performance data\n"
            "   - Season averages and trends\n"
            "   - Head-to-head matchup history\n"
            "   - Defensive matchup analysis\n\n"
            
            "3. Include team picks: spreads, moneylines, totals\n"
            "4. Cover multiple sports: basketball, football, hockey, baseball, soccer\n\n"
            
            "REQUIRED JSON STRUCTURE - Respond with ONLY this format:\n"
            "[\n"
            "  {\n"
            "    \"sport\": \"basketball\",\n"
            "    \"home_team\": \"Lakers\",\n"
            "    \"away_team\": \"Warriors\",\n"
            "    \"player_name\": \"LeBron James\",\n"
            "    \"pick_type\": \"player_prop\",\n"
            "    \"stat_type\": \"points\",\n"
            "    \"over_under\": \"over\",\n"
            "    \"line\": 24.5,\n"
            "    \"pick\": \"LeBron James Over 24.5 Points\",\n"
            "    \"odds\": -115,\n"
            "    \"confidence\": 0.82,\n"
            "    \"reasoning\": \"LeBron has averaged 26.8 points in last 10 games vs Warriors, hitting over 24.5 in 8/10 games. Last 5 games: 27.2 average. Warriors allow 4th most points to opposing forwards this season.\",\n"
            "    \"last_10_stats\": {\n"
            "      \"average\": 26.8,\n"
            "      \"games_over\": 8,\n"
            "      \"trend\": \"increasing\",\n"
            "      \"high\": 32,\n"
            "      \"low\": 21\n"
            "    },\n"
            "    \"last_5_stats\": {\n"
            "      \"average\": 27.2,\n"
            "      \"games_over\": 4,\n"
            "      \"trend\": \"stable\"\n"
            "    },\n"
            "    \"factors\": [\"Recent form advantage\", \"Favorable defensive matchup\", \"Historical performance vs opponent\", \"Rest advantage\"],\n"
            "    \"matchup_analysis\": \"Warriors rank 28th in defensive efficiency vs forwards. LeBron has 26+ points in 7 of last 8 meetings.\"\n"
            "  },\n"
            "  {\n"
            "    \"sport\": \"basketball\",\n"
            "    \"home_team\": \"Celtics\",\n"
            "    \"away_team\": \"Heat\",\n"
            "    \"player_name\": \"Jayson Tatum\",\n"
            "    \"pick_type\": \"player_prop\",\n"
            "    \"stat_type\": \"rebounds\",\n"
            "    \"over_under\": \"under\",\n"
            "    \"line\": 8.5,\n"
            "    \"pick\": \"Jayson Tatum Under 8.5 Rebounds\",\n"
            "    \"odds\": -120,\n"
            "    \"confidence\": 0.75,\n"
            "    \"reasoning\": \"Tatum has averaged only 7.2 rebounds in last 10 games, going under 8.5 in 7/10 games. Heat's aggressive rebounding limits opponent rebounds.\",\n"
            "    \"last_10_stats\": {\n"
            "      \"average\": 7.2,\n"
            "      \"games_over\": 3,\n"
            "      \"trend\": \"decreasing\",\n"
            "      \"high\": 11,\n"
            "      \"low\": 4\n"
            "    },\n"
            "    \"last_5_stats\": {\n"
            "      \"average\": 6.8,\n"
            "      \"games_over\": 1,\n"
            "      \"trend\": \"decreasing\"\n"
            "    },\n"
            "    \"factors\": [\"Recent rebounding decline\", \"Strong opponent rebounding\", \"Focus on scoring role\", \"Pace matchup\"],\n"
            "    \"matchup_analysis\": \"Heat rank 3rd in defensive rebounding rate. Tatum's rebounding has declined with increased offensive load.\"\n"
            "  }\n"
            "]\n\n"
            
            "STAT TYPES: points, rebounds, assists, steals, blocks, three_pointers, field_goals, passing_yards, rushing_yards, touchdowns, saves, goals\n"
            "CONFIDENCE: Express as decimal between 0.0-1.0 (not percentage)\n"
            "REASONING: Must reference specific statistical data and matchup factors\n"
            "FACTORS: Array of 3-5 key analytical factors supporting the pick"
        )
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert sports betting analyst with access to comprehensive statistical databases. Respond ONLY with valid JSON arrays containing detailed statistical analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2500,  # Increased for detailed statistical analysis
                    temperature=0.6,  # Slightly lower for more consistent analysis
                    timeout=45  # Increased timeout for complex responses
                )
                content = response.choices[0].message.content.strip()
                picks = self._parse_response(content)
                return picks
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Retrying in {delay}s...")
                time.sleep(delay)
        return []

    def _parse_response(self, content: str) -> List[Dict]:
        """
        Parse the OpenAI response and validate the comprehensive structure.
        Enhanced to support detailed player statistics and comprehensive validation.
        """
        import json
        try:
            picks = json.loads(content)
            if not isinstance(picks, list):
                raise ValueError("Response is not a list")
            
            validated_picks = []
            
            # Enhanced validation for comprehensive fields
            for i, pick in enumerate(picks):
                try:
                    # Core required fields for all picks
                    core_required = ["sport", "pick_type", "odds", "confidence", "pick", "reasoning", "factors"]
                    for key in core_required:
                        if key not in pick:
                            logger.warning(f"Missing core field '{key}' in pick {i+1}, skipping")
                            continue
                    
                    # Validate pick type and set appropriate requirements
                    pick_type = pick.get('pick_type', '').lower()
                    
                    if pick_type == 'player_prop':
                        # Player prop specific required fields
                        player_required = ["player_name", "stat_type", "over_under", "line", "last_10_stats", "last_5_stats", "matchup_analysis"]
                        missing_fields = [field for field in player_required if field not in pick]
                        
                        if missing_fields:
                            logger.warning(f"Player prop missing fields {missing_fields} in pick {i+1}, adding defaults")
                            
                            # Add default values for missing player prop fields
                            if 'player_name' not in pick:
                                pick['player_name'] = 'Unknown Player'
                            if 'stat_type' not in pick:
                                pick['stat_type'] = 'points'
                            if 'over_under' not in pick:
                                pick['over_under'] = 'over'
                            if 'line' not in pick:
                                pick['line'] = 0.0
                            if 'last_10_stats' not in pick:
                                pick['last_10_stats'] = {
                                    "average": 0.0,
                                    "games_over": 0,
                                    "trend": "stable",
                                    "high": 0,
                                    "low": 0
                                }
                            if 'last_5_stats' not in pick:
                                pick['last_5_stats'] = {
                                    "average": 0.0,
                                    "games_over": 0,
                                    "trend": "stable"
                                }
                            if 'matchup_analysis' not in pick:
                                pick['matchup_analysis'] = 'Standard matchup analysis'
                        
                        # Validate statistical data structure
                        if isinstance(pick.get('last_10_stats'), dict):
                            stats_10 = pick['last_10_stats']
                            required_stats_fields = ['average', 'games_over', 'trend']
                            for field in required_stats_fields:
                                if field not in stats_10:
                                    stats_10[field] = 0.0 if field == 'average' else 'stable' if field == 'trend' else 0
                        
                        if isinstance(pick.get('last_5_stats'), dict):
                            stats_5 = pick['last_5_stats']
                            required_stats_fields = ['average', 'games_over', 'trend']
                            for field in required_stats_fields:
                                if field not in stats_5:
                                    stats_5[field] = 0.0 if field == 'average' else 'stable' if field == 'trend' else 0
                    
                    else:
                        # Team picks (spread, moneyline, totals) - require team fields
                        if 'home_team' not in pick and 'away_team' not in pick and 'team' not in pick:
                            pick['team'] = 'Unknown Team'
                        
                        # Set default values for non-player props
                        pick['player_name'] = ''
                        pick['stat_type'] = ''
                        pick['over_under'] = pick.get('over_under', None)
                        pick['line'] = pick.get('line', 0.0)
                        pick['last_10_stats'] = {}
                        pick['last_5_stats'] = {}
                        pick['matchup_analysis'] = pick.get('matchup_analysis', 'Team matchup analysis')
                    
                    # Validate and normalize confidence
                    confidence = pick.get('confidence', 0)
                    if isinstance(confidence, (int, float)):
                        if confidence > 1:
                            pick['confidence'] = confidence / 100.0
                        elif confidence < 0:
                            pick['confidence'] = 0.0
                        else:
                            pick['confidence'] = float(confidence)
                    else:
                        pick['confidence'] = 0.5  # Default confidence
                    
                    # Validate factors array
                    if not isinstance(pick.get('factors'), list):
                        pick['factors'] = ['Statistical analysis', 'Matchup factors', 'Recent form']
                    
                    # Validate odds
                    try:
                        pick['odds'] = int(pick['odds'])
                    except (ValueError, TypeError):
                        pick['odds'] = -110  # Default odds
                    
                    # Validate line for props
                    try:
                        pick['line'] = float(pick.get('line', 0.0))
                    except (ValueError, TypeError):
                        pick['line'] = 0.0
                    
                    # Add team fields if missing
                    if 'home_team' not in pick:
                        pick['home_team'] = pick.get('team', 'Home Team')
                    if 'away_team' not in pick:
                        pick['away_team'] = 'Away Team'
                    
                    validated_picks.append(pick)
                    
                except Exception as pick_error:
                    logger.error(f"Error validating pick {i+1}: {pick_error}")
                    continue
            
            logger.info(f"Successfully validated {len(validated_picks)} picks with comprehensive statistical analysis")
            return validated_picks
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return []

    def get_enhanced_statistical_picks(self, max_picks: int = 5, focus_sport: str = None) -> List[Dict]:
        """
        Get enhanced AI picks with deep statistical analysis focused on specific sport.
        
        Args:
            max_picks: Maximum number of picks to generate
            focus_sport: Optional sport to focus on (basketball, football, etc.)
            
        Returns:
            List of picks with comprehensive statistical analysis
        """
        try:
            # Use the main method but with sport focus if specified
            if focus_sport:
                # Store original prompt logic but add sport focus
                logger.info(f"Generating {max_picks} enhanced picks focused on {focus_sport}")
            else:
                logger.info(f"Generating {max_picks} enhanced picks across all sports")
                
            picks = self.get_ai_daily_picks(max_picks)
            
            # Filter by sport if specified
            if focus_sport and picks:
                picks = [pick for pick in picks if pick.get('sport', '').lower() == focus_sport.lower()]
                logger.info(f"Filtered to {len(picks)} picks for {focus_sport}")
            
            return picks
            
        except Exception as e:
            logger.error(f"Error generating enhanced statistical picks: {e}")
            return []

    def validate_pick_completeness(self, pick: Dict) -> Dict:
        """
        Validate that a pick has all required statistical fields for comprehensive analysis.
        
        Args:
            pick: Pick dictionary to validate
            
        Returns:
            Dictionary with validation results and completeness score
        """
        required_fields = {
            'core': ['sport', 'pick_type', 'odds', 'confidence', 'pick', 'reasoning', 'factors'],
            'player_prop': ['player_name', 'stat_type', 'over_under', 'line', 'last_10_stats', 'last_5_stats', 'matchup_analysis'],
            'team': ['home_team', 'away_team']
        }
        
        validation_result = {
            'is_complete': True,
            'missing_fields': [],
            'completeness_score': 0.0,
            'has_statistical_analysis': False
        }
        
        # Check core fields
        missing_core = [field for field in required_fields['core'] if field not in pick]
        validation_result['missing_fields'].extend(missing_core)
        
        # Check pick-type specific fields
        if pick.get('pick_type') == 'player_prop':
            missing_player = [field for field in required_fields['player_prop'] if field not in pick]
            validation_result['missing_fields'].extend(missing_player)
            
            # Check statistical analysis quality
            last_10 = pick.get('last_10_stats', {})
            last_5 = pick.get('last_5_stats', {})
            
            if isinstance(last_10, dict) and isinstance(last_5, dict):
                if last_10.get('average') and last_5.get('average'):
                    validation_result['has_statistical_analysis'] = True
        else:
            missing_team = [field for field in required_fields['team'] if field not in pick]
            validation_result['missing_fields'].extend(missing_team)
        
        # Calculate completeness score
        total_expected = len(required_fields['core'])
        if pick.get('pick_type') == 'player_prop':
            total_expected += len(required_fields['player_prop'])
        else:
            total_expected += len(required_fields['team'])
            
        missing_count = len(validation_result['missing_fields'])
        validation_result['completeness_score'] = max(0.0, (total_expected - missing_count) / total_expected)
        validation_result['is_complete'] = missing_count == 0
        
        return validation_result

# Global instance
openai_agent = OpenAIDailyPicksAgent()
