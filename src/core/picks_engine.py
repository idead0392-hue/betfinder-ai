# src/core/picks_engine.py
import requests
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
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
        self.confidence_threshold = 60
        
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                print("OpenAI client initialized successfully")
            else:
                print("OpenAI API key not found in environment variables")
        else:
            print("OpenAI library not available")
        
    def calculate_value_score(self, odds: float, estimated_probability: float) -> float:
        """Calculate expected value of a bet"""
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
            
        implied_probability = 1 / decimal_odds
        expected_value = (estimated_probability * (decimal_odds - 1)) - (1 - estimated_probability)
        return expected_value * 100
    
    def analyze_odds_movement(self, matchup_data: Dict) -> Dict:
        """Analyze odds movement patterns"""
        analysis = {
            'line_movement': 'stable',
            'public_bias': 'neutral',
            'sharp_action': False,
            'reverse_line_movement': False
        }
        
        if 'odds' in matchup_data:
            odds = matchup_data['odds']
            if len(odds) > 0:
                for market in odds:
                    if 'outcomes' in market:
                        outcomes = market['outcomes']
                        if len(outcomes) >= 2:
                            odds_diff = abs(float(outcomes[0].get('price', 0)) - float(outcomes[1].get('price', 0)))
                            if odds_diff > 200:
                                analysis['sharp_action'] = True
                                
        return analysis
    
    def generate_ai_powered_picks(self, max_picks: int = 8) -> List[Dict]:
        """Generate AI-powered betting picks using OpenAI."""
        if not self.openai_client:
            print("OpenAI not available, falling back to traditional picks")
            return self.get_daily_picks_fallback(max_picks)
        
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            prompt = f"""You are a professional sports betting analyst. Generate {max_picks} diverse, high-quality betting picks for {current_date}.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with a valid JSON array - no other text
2. Include exactly {max_picks} picks
3. Include BOTH over AND under player props (at least 2 overs and 2 unders)
4. Include team picks: spreads, moneylines, totals (over/under)
5. Cover at least 3 different sports
6. confidence and expected_value must be numbers only (no % symbols)
7. For player props, include 'over_under' field with values 'over' or 'under'

JSON Format (respond with ONLY this array):
[
  {{
    "sport": "basketball",
    "home_team": "Lakers",
    "away_team": "Warriors", 
    "player_name": "LeBron James",
    "pick_type": "player_prop",
    "over_under": "over",
    "pick": "LeBron James Over 24.5 Points",
    "odds": -115,
    "confidence": 78,
    "expected_value": 8.2,
    "reasoning": "Strong performance vs Warriors historically with 26.8 average in last 5 games",
    "key_factors": ["Recent form", "Defensive matchup", "Rest advantage"]
  }}
]"""

            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert sports betting analyst. Respond ONLY with valid JSON arrays. Do not include any text before or after the JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=3000
            )
            
            ai_content = response.choices[0].message.content.strip()
            
            start_idx = ai_content.find('[')
            end_idx = ai_content.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = ai_content[start_idx:end_idx+1]
                try:
                    picks_data = json.loads(json_str)
                except json.JSONDecodeError:
                    picks_data = None
            else:
                picks_data = None
            
            if picks_data:
                ai_picks = []
                for i, pick_data in enumerate(picks_data[:max_picks]):
                    try:
                        if not pick_data.get('sport') or not pick_data.get('pick'):
                            continue
                            
                        confidence = float(str(pick_data.get('confidence', 75)).replace('%', ''))
                        expected_value = float(str(pick_data.get('expected_value', 5)).replace('%', ''))
                        
                        pick_type = pick_data.get('pick_type', 'moneyline')
                        home_team = pick_data.get('home_team', 'Team A')
                        away_team = pick_data.get('away_team', 'Team B')
                        
                        if pick_type == 'player_prop' and pick_data.get('player_name'):
                            player_name = pick_data.get('player_name')
                            matchup_display = f"{home_team} vs {away_team} - {player_name}"
                        else:
                            matchup_display = f"{home_team} vs {away_team}"
                        
                        pick = {
                            'game_id': f"ai_pick_{current_date}_{i+1}",
                            'sport': pick_data.get('sport', 'basketball'),
                            'competition': 'AI Analysis',
                            'matchup': matchup_display,
                            'pick': pick_data.get('pick', 'Team A'),
                            'pick_type': pick_type,
                            'over_under': pick_data.get('over_under'),
                            'player_name': pick_data.get('player_name', ''),
                            'odds': int(pick_data.get('odds', -110)),
                            'confidence': confidence,
                            'expected_value': expected_value,
                            'market_analysis': self.analyze_odds_movement({}),
                            'start_time': (datetime.now() + timedelta(hours=random.randint(2, 8))).strftime("%Y-%m-%d %H:%M"),
                            'reasoning': pick_data.get('reasoning', 'AI-generated analysis'),
                            'prediction_factors': pick_data.get('key_factors', [])
                        }
                        
                        if pick['confidence'] >= self.confidence_threshold and pick['expected_value'] > 0:
                            ai_picks.append(pick)
                            
                    except (ValueError, KeyError, TypeError) as e:
                        print(f"Error processing pick {i+1}: {e}")
                        continue
                
                return ai_picks
            
            else:
                print("Could not parse AI response as JSON")
                return self.get_daily_picks_fallback(max_picks)
                
        except Exception as e:
            print(f"Error generating AI picks: {e}")
            return self.get_daily_picks_fallback(max_picks)
    
    def get_daily_picks_fallback(self, max_picks: int = 8) -> List[Dict]:
        """Fallback method for when AI or API is unavailable."""
        print("Using fallback pick generation...")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        mock_picks = [
            {
                'game_id': f"fallback_nba_{current_date}_1",
                'sport': 'basketball',
                'competition': 'NBA',
                'matchup': 'Lakers vs Warriors',
                'pick': 'Lakers +3.5',
                'pick_type': 'spread',
                'over_under': None,
                'odds': -110,
                'confidence': 78.5,
                'expected_value': 8.2,
                'market_analysis': {'line_movement': 'favorable', 'public_bias': 'warriors', 'sharp_action': True, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'Lakers have covered 8 of last 10 games as road underdogs.',
                'prediction_factors': ['Rest advantage', 'ATS trends', 'Injury report']
            },
            {
                'game_id': f"fallback_nba_{current_date}_2",
                'sport': 'basketball', 
                'competition': 'NBA',
                'matchup': 'Lakers vs Warriors - LeBron James',
                'pick': 'LeBron James Over 24.5 Points',
                'pick_type': 'player_prop',
                'over_under': 'over',
                'player_name': 'LeBron James',
                'odds': -115,
                'confidence': 82.0,
                'expected_value': 12.3,
                'market_analysis': {'line_movement': 'stable', 'public_bias': 'over', 'sharp_action': True, 'reverse_line_movement': False},
                'start_time': (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
                'reasoning': 'LeBron averaging 26.8 points vs Warriors in last 5 meetings.',
                'prediction_factors': ['Historical matchup', 'Defensive ranking', 'Recent form']
            }
        ]
        
        filtered_picks = [pick for pick in mock_picks if pick['confidence'] >= self.confidence_threshold]
        return filtered_picks[:max_picks]
