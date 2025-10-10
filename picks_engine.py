"""

AI Picks Engine for BetFinder AI
Analyzes sports data and generates betting recommendations using OpenAI
"""

import requests
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# OpenAI integration
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")

class PicksEngine:
    def __init__(self, api_base_url: str = "http://localhost:5001"):
        self.api_base_url = api_base_url
        self.confidence_threshold = 60  # Minimum confidence to recommend a pick
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized successfully")
            else:
                print("⚠️ OpenAI API key not found in environment variables")
        else:
            print("⚠️ OpenAI library not available")
        
    def calculate_value_score(self, odds: float, estimated_probability: float) -> float:
        """Calculate expected value of a bet"""
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
            
        implied_probability = 1 / decimal_odds
        expected_value = (estimated_probability * (decimal_odds - 1)) - (1 - estimated_probability)
        return expected_value * 100  # Return as percentage
    
    def analyze_odds_movement(self, matchup_data: Dict) -> Dict:
        """Analyze odds movement patterns"""
        analysis = {
            'line_movement': 'stable',
            'public_bias': 'neutral',
            'sharp_action': False,
            'reverse_line_movement': False
        }
        
        # Simulate odds analysis (in real implementation, this would analyze historical odds)
        if 'odds' in matchup_data:
            odds = matchup_data['odds']
            if len(odds) > 0:
                # Simple heuristic: if favorites have short odds, might be public bias
                for market in odds:
                    if 'outcomes' in market:
                        outcomes = market['outcomes']
                        if len(outcomes) >= 2:
                            odds_diff = abs(float(outcomes[0].get('price', 0)) - float(outcomes[1].get('price', 0)))
                            if odds_diff > 200:  # Large spread indicates potential value
                                analysis['sharp_action'] = True
                                
        return analysis
         
        return analysis
    
    def generate_ai_powered_picks(self, max_picks: int = 8) -> List[Dict]:
        """Generate AI-powered betting picks using OpenAI analysis with variety including player props"""
        if not self.openai_client:
            print("OpenAI not available, falling back to traditional picks")
            return self.get_daily_picks_fallback(max_picks)
        
        try:
            # Current date for context
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Enhanced prompt for diverse picks including player props
            prompt = f"""You are a professional sports betting analyst. Generate {max_picks} diverse, high-quality betting picks for {current_date}.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with a valid JSON array - no other text
2. Include exactly {max_picks} picks
3. Include at least 3 player props and 2 team picks
4. Cover at least 3 different sports
5. confidence and expected_value must be numbers only (no % symbols)

Pick Types to Include:
- Player props: points, rebounds, assists, passing yards, touchdowns
- Team picks: spreads, moneylines, totals (over/under)
- Team props: team totals, first quarter totals

Sports: basketball, football, hockey, baseball, soccer, tennis

JSON Format (respond with ONLY this array):
[
  {{
    "sport": "basketball",
    "home_team": "Lakers",
    "away_team": "Warriors", 
    "player_name": "LeBron James",
    "pick_type": "player_prop",
    "pick": "LeBron James Over 24.5 Points",
    "odds": -115,
    "confidence": 78,
    "expected_value": 8.2,
    "reasoning": "Strong performance vs Warriors historically with 26.8 average in last 5 games",
    "key_factors": ["Recent form", "Defensive matchup", "Rest advantage"]
  }},
  {{
    "sport": "football",
    "home_team": "Chiefs", 
    "away_team": "Bills",
    "pick_type": "spread",
    "pick": "Chiefs -3.5",
    "odds": -110,
    "confidence": 72,
    "expected_value": 5.8,
    "reasoning": "Strong home performance and defensive advantages",
    "key_factors": ["Home field", "Defensive stats", "Weather"]
  }}
]"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert sports betting analyst. Respond ONLY with valid JSON arrays. Do not include any text before or after the JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=3000
            )
            
            # Parse the AI response
            ai_content = response.choices[0].message.content
            print(f"🔍 AI Response length: {len(ai_content)} characters")
            
            # Clean the response text
            ai_content = ai_content.strip()
            # Remove any potential BOM or invisible characters
            ai_content = ''.join(char for char in ai_content if ord(char) >= 32 or char in '\n\r\t')
            
            # Extract JSON from the response with improved parsing
            import re
            
            # Find the JSON array boundaries more precisely
            start_idx = ai_content.find('[')
            end_idx = ai_content.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = ai_content[start_idx:end_idx+1]
                try:
                    picks_data = json.loads(json_str)
                    print(f"✅ Successfully parsed JSON with {len(picks_data)} picks")
                except json.JSONDecodeError as je:
                    print(f"⚠️ JSON parse error: {je}")
                    print(f"🔍 Problematic JSON around error: ...{json_str[max(0, je.pos-50):je.pos+50]}...")
                    picks_data = None
            else:
                print("⚠️ Could not find JSON array boundaries")
                picks_data = None
            
            if picks_data:
                # Convert to our pick format with enhanced handling for player props
                ai_picks = []
                for i, pick_data in enumerate(picks_data[:max_picks]):
                    try:
                        # Validate required fields
                        if not pick_data.get('sport') or not pick_data.get('pick'):
                            print(f"⚠️ Skipping invalid pick {i+1}: missing required fields")
                            continue
                            
                        # Clean and parse confidence (remove % if present)
                        confidence_str = str(pick_data.get('confidence', 75))
                        confidence = float(confidence_str.replace('%', ''))
                        
                        # Clean and parse expected_value (remove % if present)
                        ev_str = str(pick_data.get('expected_value', 5))
                        expected_value = float(ev_str.replace('%', ''))
                        
                        # Handle different pick types with enhanced formatting
                        pick_type = pick_data.get('pick_type', 'moneyline')
                        
                        # Ensure we have team names (use defaults for player-only sports)
                        home_team = pick_data.get('home_team', 'Team A')
                        away_team = pick_data.get('away_team', 'Team B')
                        
                        # For tennis/individual sports, create meaningful matchup
                        if pick_data.get('sport') == 'tennis' and pick_data.get('player_name'):
                            player_name = pick_data.get('player_name')
                            home_team = player_name
                            away_team = "vs Opponent"
                        
                        # Create matchup display based on pick type
                        if pick_type == 'player_prop' and pick_data.get('player_name'):
                            player_name = pick_data.get('player_name')
                            matchup_display = f"{home_team} vs {away_team} - {player_name}"
                        elif pick_type == 'team_prop':
                            matchup_display = f"{home_team} vs {away_team} - Team Prop"
                        else:
                            matchup_display = f"{home_team} vs {away_team}"
                        
                        pick = {
                            'game_id': f"ai_pick_{current_date}_{i+1}",
                            'sport': pick_data.get('sport', 'basketball'),
                            'competition': 'AI Analysis',
                            'matchup': matchup_display,
                            'pick': pick_data.get('pick', 'Team A'),
                            'pick_type': pick_type,
                            'player_name': pick_data.get('player_name', ''),  # For player props
                            'odds': int(pick_data.get('odds', -110)),
                            'confidence': confidence,
                            'expected_value': expected_value,
                            'market_analysis': self.analyze_odds_movement({}),  # Basic analysis
                            'start_time': (datetime.now() + timedelta(hours=random.randint(2, 8))).strftime("%Y-%m-%d %H:%M"),
                            'reasoning': pick_data.get('reasoning', 'AI-generated analysis'),
                            'prediction_factors': pick_data.get('key_factors', [])
                        }
                        
                        # Only include picks that meet our quality threshold
                        if pick['confidence'] >= self.confidence_threshold and pick['expected_value'] > 0:
                            ai_picks.append(pick)
                            
                    except (ValueError, KeyError, TypeError) as e:
                        print(f"⚠️ Error processing pick {i+1}: {e}")
                        continue
                
                print(f"✅ Generated {len(ai_picks)} AI-powered picks")
                return ai_picks
            
            else:
                print("⚠️ Could not parse AI response as JSON")
                print(f"🔍 First 200 chars of response: {ai_content[:200]}...")
                print("🔄 Using fallback picks...")
                return self.get_daily_picks_fallback(max_picks)
                
        except json.JSONDecodeError as je:
            print(f"❌ JSON parsing error: {je}")
            print(f"🔍 Response preview: {ai_content[:300] if 'ai_content' in locals() else 'No response'}...")
            print("🔄 Using fallback picks...")
            return self.get_daily_picks_fallback(max_picks)
        except Exception as e:
            print(f"❌ Error generating AI picks: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            print("🔄 Using fallback picks...")
            return self.get_daily_picks_fallback(max_picks)
    
    def get_daily_picks_fallback(self, max_picks: int = 8) -> List[Dict]:
        """Fallback method for when AI or API is unavailable - includes player props"""
        print("Using fallback pick generation with player props...")
        
        # Enhanced mock data with player props and variety
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        mock_picks = [
            {
                'game_id': f"fallback_nba_{current_date}_1",
                'sport': 'basketball',
                'competition': 'NBA',
                'matchup': 'Lakers vs Warriors',
                'pick': 'Lakers +3.5',
                'pick_type': 'spread',
                'odds': -110,
                'confidence': 78.5,
                'expected_value': 8.2,
                'market_analysis': {'line_movement': 'favorable', 'public_bias': 'warriors', 'sharp_action': True, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Lakers have covered 8 of last 10 games as road underdogs. Warriors playing back-to-back without key rotation players. Historical edge in similar spots.',
                'prediction_factors': ['Rest advantage', 'ATS trends', 'Injury report', 'Line value']
            },
            {
                'game_id': f"fallback_nba_{current_date}_2",
                'sport': 'basketball', 
                'competition': 'NBA',
                'matchup': 'Lakers vs Warriors - LeBron James',
                'pick': 'LeBron James Over 24.5 Points',
                'pick_type': 'player_prop',
                'player_name': 'LeBron James',
                'odds': -115,
                'confidence': 82.0,
                'expected_value': 12.3,
                'market_analysis': {'line_movement': 'stable', 'public_bias': 'over', 'sharp_action': True, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'LeBron averaging 26.8 points vs Warriors in last 5 meetings. Warriors allow 4th most points to opposing forwards. He has hit this number in 7 of last 10 games.',
                'prediction_factors': ['Historical matchup', 'Defensive ranking', 'Recent form', 'Usage rate']
            },
            {
                'game_id': f"fallback_nfl_{current_date}_3", 
                'sport': 'football',
                'competition': 'NFL',
                'matchup': 'Chiefs vs Bills',
                'pick': 'Over 52.5',
                'pick_type': 'totals',
                'odds': -105,
                'confidence': 72.3,
                'expected_value': 6.8,
                'market_analysis': {'line_movement': 'stable', 'public_bias': 'over', 'sharp_action': False, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Both offenses rank top-5 in red zone efficiency. Weather forecast shows dome conditions. Historical high-scoring matchups average 56.3 points.',
                'prediction_factors': ['Offensive efficiency', 'Weather conditions', 'Historical totals', 'Pace of play']
            },
            {
                'game_id': f"fallback_nfl_{current_date}_4",
                'sport': 'football',
                'competition': 'NFL', 
                'matchup': 'Chiefs vs Bills - Josh Allen',
                'pick': 'Josh Allen Over 1.5 Passing TDs',
                'pick_type': 'player_prop',
                'player_name': 'Josh Allen',
                'odds': -125,
                'confidence': 75.5,
                'expected_value': 9.8,
                'market_analysis': {'line_movement': 'favorable', 'public_bias': 'over', 'sharp_action': True, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Allen has thrown 2+ TDs in 9 of last 11 games. Chiefs defense allows 24.1 points per game. Red zone efficiency favors passing in big games.',
                'prediction_factors': ['Red zone trends', 'Defensive matchup', 'Game script', 'Weather conditions']
            },
            {
                'game_id': f"fallback_nhl_{current_date}_5",
                'sport': 'hockey',
                'competition': 'NHL',
                'matchup': 'Rangers vs Bruins',
                'pick': 'Rangers +1.5',
                'pick_type': 'spread',
                'odds': -140,
                'confidence': 68.2,
                'expected_value': 7.1,
                'market_analysis': {'line_movement': 'stable', 'public_bias': 'bruins', 'sharp_action': False, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Rangers 12-3 ATS as road underdogs this season. Bruins playing 3rd game in 4 nights. Igor Shesterkin excellent in big road games.',
                'prediction_factors': ['Schedule advantage', 'Goalie matchup', 'ATS trends', 'Rest factors']
            },
            {
                'game_id': f"fallback_nhl_{current_date}_6",
                'sport': 'hockey',
                'competition': 'NHL',
                'matchup': 'Rangers vs Bruins - David Pastrnak',
                'pick': 'David Pastrnak Over 0.5 Goals',
                'pick_type': 'player_prop', 
                'player_name': 'David Pastrnak',
                'odds': +105,
                'confidence': 71.0,
                'expected_value': 11.2,
                'market_analysis': {'line_movement': 'favorable', 'public_bias': 'over', 'sharp_action': True, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Pastrnak has scored in 6 of last 8 home games. Rangers allow 3.2 goals per game on road. Power play opportunities expected.',
                'prediction_factors': ['Home splits', 'Defensive matchup', 'Special teams', 'Shot volume']
            },
            {
                'game_id': f"fallback_soccer_{current_date}_7",
                'sport': 'soccer',
                'competition': 'Premier League',
                'matchup': 'Arsenal vs Chelsea',
                'pick': 'Both Teams To Score - Yes',
                'pick_type': 'team_prop',
                'odds': -110,
                'confidence': 76.8,
                'expected_value': 8.9,
                'market_analysis': {'line_movement': 'stable', 'public_bias': 'yes', 'sharp_action': True, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Both teams scored in 7 of last 8 meetings. Arsenal averaging 2.1 goals at home, Chelsea 1.6 away. Attacking lineups expected.',
                'prediction_factors': ['Historical BTTS', 'Home/away splits', 'Team news', 'Match importance']
            },
            {
                'game_id': f"fallback_tennis_{current_date}_8",
                'sport': 'tennis',
                'competition': 'ATP Tour',
                'matchup': 'Djokovic vs Alcaraz',
                'pick': 'Over 3.5 Sets',
                'pick_type': 'totals',
                'odds': +120,
                'confidence': 69.5,
                'expected_value': 14.2,
                'market_analysis': {'line_movement': 'favorable', 'public_bias': 'under', 'sharp_action': True, 'reverse_line_movement': True},
                'start_time': (datetime.now() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Last 4 meetings went to 4+ sets. Both players in excellent form. Hard court surface favors longer matches between these styles.',
                'prediction_factors': ['Head-to-head', 'Surface performance', 'Current form', 'Playing style matchup']
            }
        ]
        
        # Filter and return requested number of picks
        filtered_picks = [pick for pick in mock_picks if pick['confidence'] >= self.confidence_threshold]
        return filtered_picks[:max_picks]
    
    def calculate_team_strength(self, team_stats: Dict) -> float:
        """Calculate overall team strength rating"""
        # Placeholder algorithm - in production this would use comprehensive stats
        factors = {
            'recent_form': random.uniform(0.7, 0.95),
            'home_advantage': 0.03 if team_stats.get('is_home') else 0,
            'rest_days': min(0.02 * team_stats.get('rest_days', 3), 0.08),
            'injury_impact': random.uniform(-0.05, 0.05)
        }
        
        base_strength = 0.5  # Neutral
        for factor, weight in factors.items():
            base_strength += weight
            
        return min(max(base_strength, 0.1), 0.9)  # Clamp between 0.1 and 0.9
    
    def generate_pick_reasoning(self, pick: Dict) -> str:
        """Generate human-readable reasoning for the pick"""
        reasoning_parts = []
        
        # Confidence level reasoning
        confidence = pick['confidence']
        if confidence > 80:
            reasoning_parts.append("📈 High confidence based on strong statistical indicators")
        elif confidence > 65:
            reasoning_parts.append("📊 Moderate confidence with favorable metrics")
        else:
            reasoning_parts.append("⚠️ Lower confidence, proceed with caution")
            
        # Value reasoning
        if pick['expected_value'] > 5:
            reasoning_parts.append(f"💰 Excellent value bet (+{pick['expected_value']:.1f}% EV)")
        elif pick['expected_value'] > 2:
            reasoning_parts.append(f"💵 Good value (+{pick['expected_value']:.1f}% EV)")
        elif pick['expected_value'] > 0:
            reasoning_parts.append(f"✅ Positive expected value (+{pick['expected_value']:.1f}% EV)")
        else:
            reasoning_parts.append("❌ Negative expected value - avoid this bet")
            
        # Market analysis
        market_analysis = pick.get('market_analysis', {})
        if market_analysis.get('sharp_action'):
            reasoning_parts.append("🎯 Sharp money detected on this line")
        if market_analysis.get('reverse_line_movement'):
            reasoning_parts.append("🔄 Reverse line movement indicates value")
            
        return " • ".join(reasoning_parts)
    
    def analyze_matchup(self, sport: str, competition_id: str, matchup: Dict) -> Optional[Dict]:
        """Analyze a single matchup and generate pick recommendation"""
        try:
            # Get detailed matchup data
            response = requests.get(
                f"{self.api_base_url}/api/{sport}/matchups/{competition_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                return None
                
            matchup_data = response.json()
            if not matchup_data.get('matchups'):
                return None
                
            # Find our specific matchup
            target_matchup = None
            for m in matchup_data['matchups']:
                if m.get('game_id') == matchup.get('game_id'):
                    target_matchup = m
                    break
                    
            if not target_matchup:
                return None
                
            # Analyze odds and calculate probabilities
            odds_analysis = self.analyze_odds_movement(target_matchup)
            
            # Calculate team strengths (simplified)
            home_strength = self.calculate_team_strength({'is_home': True, 'rest_days': 2})
            away_strength = self.calculate_team_strength({'is_home': False, 'rest_days': 3})
            
            # Determine pick based on analysis
            if home_strength > away_strength * 1.1:  # 10% edge threshold
                pick_team = matchup.get('home_team', 'Home')
                estimated_prob = home_strength
                pick_type = 'moneyline'
                odds_value = -150  # Example odds
            elif away_strength > home_strength * 1.1:
                pick_team = matchup.get('away_team', 'Away')
                estimated_prob = away_strength
                pick_type = 'moneyline'
                odds_value = 130  # Example odds
            else:
                # Look for spread/total value
                pick_team = "Under Total Points"
                estimated_prob = 0.55
                pick_type = 'total'
                odds_value = -110
                
            # Calculate metrics
            expected_value = self.calculate_value_score(odds_value, estimated_prob)
            confidence = min(95, max(50, (estimated_prob - 0.5) * 200 + random.uniform(5, 15)))
            
            # Only return picks above confidence threshold with positive EV
            if confidence >= self.confidence_threshold and expected_value > 0:
                pick = {
                    'game_id': matchup.get('game_id'),
                    'sport': sport,
                    'competition': matchup.get('tournament', 'Unknown'),
                    'matchup': f"{matchup.get('home_team', 'Home')} vs {matchup.get('away_team', 'Away')}",
                    'pick': pick_team,
                    'pick_type': pick_type,
                    'odds': odds_value,
                    'confidence': round(confidence, 1),
                    'expected_value': round(expected_value, 2),
                    # NOTE: Unit sizing removed - handled by separate BankrollManager
                    'market_analysis': odds_analysis,
                    'start_time': matchup.get('start_time', 'TBD'),
                    'reasoning': "",
                    'prediction_factors': []  # For transparency in AI reasoning
                }
                
                # Add reasoning
                pick['reasoning'] = self.generate_pick_reasoning(pick)
                
                return pick
                
        except Exception as e:
            print(f"Error analyzing matchup: {e}")
            return None
            
        return None
    
    def get_daily_picks(self, max_picks: int = 8) -> List[Dict]:
        """Generate daily betting picks - uses AI when available, fallback otherwise"""
        
        # Try AI-powered picks first
        if self.openai_client:
            print("🤖 Generating AI-powered picks using OpenAI...")
            return self.generate_ai_powered_picks(max_picks)
        else:
            print("🔄 OpenAI not available, using traditional approach...")
            return self.get_traditional_picks(max_picks)
    
    def get_traditional_picks(self, max_picks: int = 5) -> List[Dict]:
        """Original method - tries to fetch from API, falls back to enhanced mock data"""
        all_picks = []
        sports = ['soccer', 'basketball', 'football', 'tennis', 'hockey']
        
        for sport in sports:
            try:
                # Get competitions for sport
                comp_response = requests.get(f"{self.api_base_url}/api/{sport}/competitions", timeout=10)
                if comp_response.status_code != 200:
                    continue
                    
                competitions = comp_response.json().get('competitions', [])
                
                for comp in competitions[:2]:  # Limit to 2 competitions per sport
                    comp_id = comp.get('id')
                    if not comp_id:
                        continue
                        
                    # Get matchups for competition
                    matchup_response = requests.get(f"{self.api_base_url}/api/{sport}/matchups/{comp_id}", timeout=10)
                    if matchup_response.status_code != 200:
                        continue
                        
                    matchups_data = matchup_response.json()
                    matchups = matchups_data.get('matchups', [])
                    
                    for matchup in matchups[:3]:  # Limit matchups per competition
                        pick = self.analyze_matchup(sport, comp_id, matchup)
                        if pick:
                            all_picks.append(pick)
                            
                        if len(all_picks) >= max_picks:
                            break
                            
                    if len(all_picks) >= max_picks:
                        break
                        
            except Exception as e:
                print(f"Error processing {sport}: {e}")
                continue
        
        # If no picks from API, use fallback
        if not all_picks:
            print("No API data available, using fallback picks...")
            return self.get_daily_picks_fallback(max_picks)
                
        # Sort by expected value and confidence
        all_picks.sort(key=lambda x: (x['expected_value'], x['confidence']), reverse=True)
        
        return all_picks[:max_picks]
    
    def get_pick_summary(self, picks: List[Dict]) -> Dict:
        """Generate summary statistics for the day's picks - ANALYSIS ONLY, no financial advice"""
        if not picks:
            return {
                'total_picks': 0,
                'avg_confidence': 0,
                'avg_expected_value': 0,
                'sports_covered': 0
            }
            
        avg_confidence = sum(pick['confidence'] for pick in picks) / len(picks)
        avg_ev = sum(pick['expected_value'] for pick in picks) / len(picks)
        sports_covered = len(set(pick['sport'] for pick in picks))
        
        return {
            'total_picks': len(picks),
            'avg_confidence': round(avg_confidence, 1),
            'avg_expected_value': round(avg_ev, 2),
            'sports_covered': sports_covered,
            'note': 'Use BankrollManager for bet sizing recommendations'
        }

# Example usage and testing
if __name__ == "__main__":
    engine = PicksEngine()
    picks = engine.get_daily_picks()
    
    print("=== Daily AI Picks (Analysis Only) ===")
    for i, pick in enumerate(picks, 1):
        print(f"{i}. {pick['matchup']} - {pick['pick']}")
        print(f"   Confidence: {pick['confidence']}% | EV: +{pick['expected_value']}%")
        print(f"   {pick['reasoning']}")
        print()
        
    summary = engine.get_pick_summary(picks)
    print("=== Summary ===")
    print(f"Total Picks: {summary['total_picks']}")
    print(f"Average Confidence: {summary['avg_confidence']}%")
    print(f"Average EV: +{summary['avg_expected_value']}%")
    print(f"Sports Covered: {summary['sports_covered']}")
    print(f"\n{summary['note']}")