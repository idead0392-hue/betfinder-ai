#!/usr/bin/env python3
"""
Props Data Fetcher

This module handles fetching betting props from various data sources
including SportBex API, mock data, and other providers.
Normalizes data into a consistent format for the application.
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import random

try:
    from sportbex_provider import SportbexProvider
    from api_providers import APIResponse
    SPORTBEX_AVAILABLE = True
except ImportError:
    SPORTBEX_AVAILABLE = False
    print("⚠️ SportBex provider not available, using mock data only")


class PropsDataFetcher:
    """
    Unified data fetcher for betting props from multiple sources
    Provides normalized prop data with consistent structure
    """
    
    def __init__(self):
        self.sportbex_provider = None
        
        # Initialize SportBex provider if available
        if SPORTBEX_AVAILABLE:
            try:
                self.sportbex_provider = SportbexProvider()
                print("✅ SportBex provider initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize SportBex provider: {e}")
                self.sportbex_provider = None
        
        # Define sport mappings
        self.sport_mappings = {
            'basketball': '7522',
            'football': '7521',
            'soccer': '7520',
            'baseball': '7523',
            'hockey': '7524',
            'tennis': '7525'
        }
    
    def fetch_all_props(self, max_props: int = 50) -> List[Dict]:
        """
        Fetch all available props from all data sources
        
        Args:
            max_props: Maximum number of props to return
            
        Returns:
            List of normalized prop dictionaries
        """
        all_props = []
        
        # Try to fetch from SportBex first
        if self.sportbex_provider:
            try:
                sportbex_props = self._fetch_sportbex_props()
                all_props.extend(sportbex_props)
                print(f"📊 Fetched {len(sportbex_props)} props from SportBex")
            except Exception as e:
                print(f"⚠️ SportBex fetch error: {e}")
        
        # Always include mock data to ensure we have props to display
        mock_props = self._generate_mock_props(max_props - len(all_props))
        all_props.extend(mock_props)
        print(f"🎲 Added {len(mock_props)} mock props")
        
        # Limit total props
        if len(all_props) > max_props:
            all_props = all_props[:max_props]
        
        print(f"✅ Total props fetched: {len(all_props)}")
        return all_props
    
    def _fetch_sportbex_props(self) -> List[Dict]:
        """Fetch props from SportBex API and normalize format"""
        props = []
        
        # Fetch from multiple sports
        for sport_name, sport_id in self.sport_mappings.items():
            try:
                # Get competitions for this sport
                competitions_response = self.sportbex_provider.get_competitions(sport_id=sport_id)
                
                if competitions_response.success and competitions_response.data:
                    # Handle both nested and direct data formats
                    competitions_data = competitions_response.data
                    if isinstance(competitions_data, dict) and 'data' in competitions_data:
                        competitions_list = competitions_data['data']
                    else:
                        competitions_list = competitions_data if isinstance(competitions_data, list) else []
                    
                    # Get events for each competition
                    for competition in competitions_list[:2]:  # Limit to 2 competitions per sport
                        comp_id = competition.get('id')
                        comp_name = competition.get('name', 'Unknown Competition')
                        
                        if comp_id:
                            events_response = self.sportbex_provider.get_events(
                                competition_id=comp_id, 
                                sport_id=sport_id
                            )
                            
                            if events_response.success and events_response.data:
                                # Handle both nested and direct data formats
                                events_data = events_response.data
                                if isinstance(events_data, dict) and 'data' in events_data:
                                    events_list = events_data['data']
                                else:
                                    events_list = events_data if isinstance(events_data, list) else []
                                
                                # Convert events to normalized props
                                sport_props = self._normalize_sportbex_data(
                                    events_list, 
                                    sport_name,
                                    comp_name
                                )
                                props.extend(sport_props)
                                
            except Exception as e:
                print(f"⚠️ Error fetching {sport_name} props: {e}")
                continue
        
        return props
    
    def _normalize_sportbex_data(self, events_data: List[Dict], sport: str, competition: str) -> List[Dict]:
        """Convert SportBex event data to normalized prop format"""
        normalized_props = []
        
        for event in events_data:
            try:
                # Extract basic event info
                home_team = event.get('home_team', {})
                away_team = event.get('away_team', {})
                
                # Handle both string and dict formats for team names
                if isinstance(home_team, dict):
                    home_team_name = home_team.get('name', 'Home Team')
                else:
                    home_team_name = str(home_team) if home_team else 'Home Team'
                    
                if isinstance(away_team, dict):
                    away_team_name = away_team.get('name', 'Away Team')
                else:
                    away_team_name = str(away_team) if away_team else 'Away Team'
                
                start_time = event.get('start_time', '')
                
                # Create matchup string
                matchup = f"{home_team_name} vs {away_team_name}"
                
                # Extract odds/markets if available
                markets = event.get('markets', [])
                
                if markets:
                    for market in markets[:5]:  # Limit markets per event
                        # Create props from market data
                        prop = self._create_prop_from_market(
                            market, sport, matchup, start_time, competition
                        )
                        if prop:
                            normalized_props.append(prop)
                else:
                    # Create basic spread/moneyline props if no detailed markets
                    basic_props = self._create_basic_props(
                        home_team_name, away_team_name, sport, start_time, competition
                    )
                    normalized_props.extend(basic_props)
                    
            except Exception as e:
                print(f"⚠️ Error normalizing event data: {e}")
                continue
        
        return normalized_props
    
    def _create_prop_from_market(self, market: Dict, sport: str, matchup: str, 
                                start_time: str, competition: str) -> Optional[Dict]:
        """Create normalized prop from market data"""
        try:
            market_name = market.get('name', 'Unknown Market')
            outcomes = market.get('outcomes', [])
            
            if not outcomes:
                return None
            
            # Take first outcome for simplicity
            outcome = outcomes[0]
            
            # Determine prop type and over/under
            prop_type = 'player_prop' if any(word in market_name.lower() 
                                           for word in ['points', 'rebounds', 'assists', 'yards']) else 'team'
            over_under = 'over' if 'over' in market_name.lower() else 'under' if 'under' in market_name.lower() else None
            
            # Extract odds
            odds = outcome.get('price', -110)
            if isinstance(odds, str):
                try:
                    odds = int(odds)
                except:
                    odds = -110
            
            # Calculate confidence from odds
            confidence = self._calculate_confidence_from_odds(odds)
            
            return {
                'game_id': f"sportbex_{market.get('id', 'unknown')}",
                'sport': sport,
                'competition': competition,
                'matchup': matchup,
                'pick': market_name,
                'pick_type': prop_type,
                'over_under': over_under,
                'player_name': self._extract_player_name(market_name),
                'stat_type': self._extract_stat_type(market_name),
                'line': self._extract_line(market_name),
                'odds': odds,
                'confidence': confidence,
                'expected_value': max(0, (confidence - 50) * 0.2),  # Simple EV calculation
                'start_time': start_time,
                'reasoning': f"Market analysis for {market_name}",
                'factors': ['Market data', 'Historical odds', 'Line movement'],
                'last_10_stats': {},
                'last_5_stats': {},
                'matchup_analysis': f"Analysis for {matchup} in {competition}"
            }
            
        except Exception as e:
            print(f"⚠️ Error creating prop from market: {e}")
            return None
    
    def _create_basic_props(self, home_team: str, away_team: str, sport: str, 
                           start_time: str, competition: str) -> List[Dict]:
        """Create basic spread/moneyline props when detailed market data isn't available"""
        matchup = f"{home_team} vs {away_team}"
        props = []
        
        # Create spread prop
        spread = random.choice([-7.5, -3.5, -1.5, 1.5, 3.5, 7.5])
        spread_team = home_team if spread < 0 else away_team
        
        props.append({
            'game_id': f"basic_spread_{home_team}_{away_team}".replace(' ', '_'),
            'sport': sport,
            'competition': competition,
            'matchup': matchup,
            'pick': f"{spread_team} {spread:+}",
            'pick_type': 'spread',
            'over_under': None,
            'player_name': '',
            'stat_type': '',
            'line': abs(spread),
            'odds': random.choice([-110, -105, -115]),
            'confidence': random.uniform(65, 85),
            'expected_value': random.uniform(3, 12),
            'start_time': start_time,
            'reasoning': f"Spread analysis for {matchup}",
            'factors': ['Team strength', 'Home advantage', 'Recent form'],
            'last_10_stats': {},
            'last_5_stats': {},
            'matchup_analysis': f"Spread analysis for {matchup}"
        })
        
        # Create total prop
        total = random.choice([215.5, 220.5, 225.5, 230.5, 235.5])
        props.append({
            'game_id': f"basic_total_{home_team}_{away_team}".replace(' ', '_'),
            'sport': sport,
            'competition': competition,
            'matchup': matchup,
            'pick': f"Over {total}",
            'pick_type': 'totals',
            'over_under': 'over',
            'player_name': '',
            'stat_type': 'points',
            'line': total,
            'odds': random.choice([-110, -105, -115]),
            'confidence': random.uniform(60, 80),
            'expected_value': random.uniform(2, 8),
            'start_time': start_time,
            'reasoning': f"Total points analysis for {matchup}",
            'factors': ['Pace of play', 'Defensive efficiency', 'Weather'],
            'last_10_stats': {},
            'last_5_stats': {},
            'matchup_analysis': f"Scoring analysis for {matchup}"
        })
        
        return props
    
    def _generate_mock_props(self, count: int = 20) -> List[Dict]:
        """Generate realistic mock props for testing and fallback"""
        if count <= 0:
            return []
            
        mock_props = []
        
        # Define realistic prop templates
        prop_templates = [
            # Basketball player props
            {
                'sport': 'basketball',
                'competition': 'NBA',
                'teams': [('Lakers', 'Warriors'), ('Celtics', 'Heat'), ('Nets', 'Knicks')],
                'props': [
                    ('LeBron James', 'points', 'over', 24.5),
                    ('Stephen Curry', 'three_pointers', 'over', 4.5),
                    ('Anthony Davis', 'rebounds', 'under', 10.5),
                    ('Jayson Tatum', 'assists', 'over', 6.5),
                ]
            },
            # Football player props
            {
                'sport': 'football',
                'competition': 'NFL',
                'teams': [('Chiefs', 'Bills'), ('Cowboys', 'Giants'), ('49ers', 'Seahawks')],
                'props': [
                    ('Patrick Mahomes', 'passing_yards', 'over', 275.5),
                    ('Josh Allen', 'touchdowns', 'over', 2.5),
                    ('Dak Prescott', 'completions', 'under', 22.5),
                    ('Christian McCaffrey', 'rushing_yards', 'over', 85.5),
                ]
            }
        ]
        
        for template in prop_templates:
            for home_team, away_team in template['teams']:
                matchup = f"{home_team} vs {away_team}"
                
                for player_name, stat_type, over_under, line in template['props']:
                    if len(mock_props) >= count:
                        break
                        
                    # Generate start time (next few hours)
                    start_time = (datetime.now() + timedelta(hours=random.randint(2, 10))).strftime("%Y-%m-%d %H:%M")
                    
                    # Generate realistic stats
                    confidence = random.uniform(65, 90)
                    odds = random.choice([-120, -115, -110, -105, 105, 110, 115])
                    
                    mock_props.append({
                        'game_id': f"mock_{len(mock_props)}_{player_name.replace(' ', '_')}",
                        'sport': template['sport'],
                        'competition': template['competition'],
                        'matchup': matchup,
                        'pick': f"{player_name} {over_under.title()} {line} {stat_type.replace('_', ' ').title()}",
                        'pick_type': 'player_prop',
                        'over_under': over_under,
                        'player_name': player_name,
                        'stat_type': stat_type,
                        'line': line,
                        'odds': odds,
                        'confidence': confidence,
                        'expected_value': max(0, (confidence - 50) * 0.15),
                        'start_time': start_time,
                        'reasoning': f"{player_name} has been performing {over_under} this line recently with favorable matchup factors.",
                        'factors': ['Recent form', 'Matchup advantage', 'Historical performance', 'Line value'],
                        'last_10_stats': {
                            'average': line + random.uniform(-3, 5),
                            'games_over': random.randint(6, 9) if over_under == 'over' else random.randint(3, 6),
                            'trend': random.choice(['increasing', 'stable', 'decreasing']),
                            'high': line + random.uniform(5, 15),
                            'low': max(0, line - random.uniform(3, 8))
                        },
                        'last_5_stats': {
                            'average': line + random.uniform(-2, 4),
                            'games_over': random.randint(3, 5) if over_under == 'over' else random.randint(1, 3),
                            'trend': random.choice(['increasing', 'stable', 'decreasing'])
                        },
                        'matchup_analysis': f"Favorable matchup for {player_name} based on opponent's defensive rankings and recent trends."
                    })
        
        return mock_props[:count]
    
    def _calculate_confidence_from_odds(self, odds: int) -> float:
        """Calculate confidence percentage from American odds"""
        try:
            if odds > 0:
                implied_prob = 100 / (odds + 100)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100)
            
            # Convert to confidence (slightly higher than implied probability)
            confidence = min(95, max(50, implied_prob * 100 + random.uniform(5, 15)))
            return round(confidence, 1)
        except:
            return 70.0  # Default confidence
    
    def _extract_player_name(self, market_name: str) -> str:
        """Extract player name from market name"""
        # Simple extraction - can be improved with better parsing
        words = market_name.split()
        if len(words) >= 2:
            # Look for name pattern (FirstName LastName)
            for i in range(len(words) - 1):
                if words[i][0].isupper() and words[i+1][0].isupper():
                    return f"{words[i]} {words[i+1]}"
        return ""
    
    def _extract_stat_type(self, market_name: str) -> str:
        """Extract stat type from market name"""
        name_lower = market_name.lower()
        
        stat_mappings = {
            'points': 'points',
            'rebounds': 'rebounds', 
            'assists': 'assists',
            'steals': 'steals',
            'blocks': 'blocks',
            'threes': 'three_pointers',
            '3-pointers': 'three_pointers',
            'yards': 'yards',
            'touchdowns': 'touchdowns',
            'completions': 'completions'
        }
        
        for key, value in stat_mappings.items():
            if key in name_lower:
                return value
        
        return 'points'  # Default
    
    def _extract_line(self, market_name: str) -> float:
        """Extract betting line from market name"""
        import re
        
        # Look for decimal numbers in the market name
        numbers = re.findall(r'\d+\.?\d*', market_name)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        
        return 0.0  # Default


# Global instance
props_fetcher = PropsDataFetcher()