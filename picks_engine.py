"""
AI Picks Engine for BetFinder AI
Analyzes sports data and generates betting recommendations
"""

import requests
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

class PicksEngine:
    def __init__(self, api_base_url: str = "http://localhost:5001"):
        self.api_base_url = api_base_url
        self.confidence_threshold = 60  # Minimum confidence to recommend a pick
        
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
                    'unit_size': self.calculate_unit_size(confidence, expected_value),
                    'market_analysis': odds_analysis,
                    'start_time': matchup.get('start_time', 'TBD'),
                    'reasoning': ""
                }
                
                # Add reasoning
                pick['reasoning'] = self.generate_pick_reasoning(pick)
                
                return pick
                
        except Exception as e:
            print(f"Error analyzing matchup: {e}")
            return None
            
        return None
    
    def calculate_unit_size(self, confidence: float, expected_value: float) -> float:
        """Calculate recommended unit size using Kelly Criterion"""
        # Simplified Kelly Criterion
        if expected_value <= 0:
            return 0
            
        # Conservative approach: cap at 3 units max
        kelly_fraction = min(expected_value / 100, 0.03)
        
        # Adjust based on confidence
        confidence_multiplier = confidence / 100
        
        unit_size = kelly_fraction * confidence_multiplier * 10  # Scale to units
        return round(min(unit_size, 3.0), 1)  # Cap at 3 units
    
    def get_daily_picks(self, max_picks: int = 5) -> List[Dict]:
        """Generate daily betting picks across all sports"""
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
                
        # Sort by expected value and confidence
        all_picks.sort(key=lambda x: (x['expected_value'], x['confidence']), reverse=True)
        
        return all_picks[:max_picks]
    
    def get_pick_summary(self, picks: List[Dict]) -> Dict:
        """Generate summary statistics for the day's picks"""
        if not picks:
            return {
                'total_picks': 0,
                'avg_confidence': 0,
                'avg_expected_value': 0,
                'total_units': 0,
                'sports_covered': 0
            }
            
        avg_confidence = sum(pick['confidence'] for pick in picks) / len(picks)
        avg_ev = sum(pick['expected_value'] for pick in picks) / len(picks)
        total_units = sum(pick['unit_size'] for pick in picks)
        sports_covered = len(set(pick['sport'] for pick in picks))
        
        return {
            'total_picks': len(picks),
            'avg_confidence': round(avg_confidence, 1),
            'avg_expected_value': round(avg_ev, 2),
            'total_units': round(total_units, 1),
            'sports_covered': sports_covered
        }

# Example usage and testing
if __name__ == "__main__":
    engine = PicksEngine()
    picks = engine.get_daily_picks()
    
    print("=== Daily Picks ===")
    for i, pick in enumerate(picks, 1):
        print(f"{i}. {pick['matchup']} - {pick['pick']}")
        print(f"   Confidence: {pick['confidence']}% | EV: +{pick['expected_value']}%")
        print(f"   Units: {pick['unit_size']} | {pick['reasoning']}")
        print()
        
    summary = engine.get_pick_summary(picks)
    print("=== Summary ===")
    print(f"Total Picks: {summary['total_picks']}")
    print(f"Average Confidence: {summary['avg_confidence']}%")
    print(f"Average EV: +{summary['avg_expected_value']}%")
    print(f"Total Units: {summary['total_units']}")