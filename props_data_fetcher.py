"""
Props data fetcher for PrizePicks and Underdog.

Since we don't have live API integrations here, these functions generate
well-structured mock props aligned with the existing schema used across the app:

Required/common fields:
- game_id (str)
- player_name (str)
- stat_type (str)
- line (float | int)
- odds (int)  # American odds; PrizePicks/Underdog approximated
- event_start_time (ISO str)
- matchup (str)
- sportsbook (str)  # "PrizePicks" | "Underdog"
- recent_form (float 0-10)
- matchup_difficulty (float 0-10)
- injury_status (str)

Optional esports fields (added when sport is an esports title):
- map (str)
- side_preference (str)
- team_chemistry (float)
- recent_map_performance (float)
- opponent_map_ban_rate (float)

These mocks are deterministic per run and sport-agnostic but adapt stat types/names.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import random
from typing import Dict, List


# Seed randomness lightly for variability per run while keeping consistency
random.seed()


ESPORTS_TITLES = {
    "csgo", "league_of_legends", "dota2", "valorant", "overwatch", "rocket_league"
}

# Allowed esports stat types per new spec (default for most esports)
ALLOWED_ESPORTS_STATS = [
    'combined_map_1_2_kills',
    'combined_map_1_2_headshots',
    'fantasy_points'
]

# LoL-specific extension: allow assists as 4th type
ALLOWED_ESPORTS_STATS_PER_SPORT = {
    "league_of_legends": ALLOWED_ESPORTS_STATS + ['combined_map_1_2_assists']
}


def _now_plus_hours(hours: int) -> str:
    return (datetime.utcnow() + timedelta(hours=hours)).isoformat()


def _sample_players_for_sport(sport: str) -> List[str]:
    sport = (sport or "").lower()
    if sport in {"basketball", "nba"}:
        return ["LeBron James", "Stephen Curry", "Luka Doncic", "Jayson Tatum"]
    if sport in {"football", "nfl"}:
        return ["Patrick Mahomes", "Travis Kelce", "Justin Jefferson", "Josh Allen"]
    if sport in {"baseball", "mlb"}:
        return ["Shohei Ohtani", "Aaron Judge", "Mookie Betts", "Juan Soto"]
    if sport in {"hockey", "nhl"}:
        return ["Connor McDavid", "Auston Matthews", "Nathan MacKinnon", "Sidney Crosby"]
    if sport in {"tennis"}:
        return ["Novak Djokovic", "Carlos Alcaraz", "Iga Swiatek", "Jannik Sinner"]
    if sport in {"soccer", "football_global"}:
        return ["Erling Haaland", "Kylian Mbappé", "Lionel Messi", "Harry Kane"]

    # Esports
    if sport == "csgo":
        return ["s1mple", "ZywOo", "NiKo", "sh1ro", "blameF", "stavn"]
    if sport == "league_of_legends":
        return ["Faker", "Chovy", "Canyon", "Ruler", "Knight", "Caps"]
    if sport == "dota2":
        return ["N0tail", "Topson", "Yatoro", "Abed", "Mira", "SumaiL"]
    if sport == "valorant":
        return ["TenZ", "yay", "aspas", "Derke", "ShahZaM", "ScreaM"]
    if sport == "overwatch":
        return ["Carpe", "Fleta", "Alarm", "Profit", "JJoNak", "Meko"]
    if sport == "rocket_league":
        return ["jstn", "GarrettG", "Squishy", "Firstkiller", "ApparentlyJack", "BeastMode"]

    # Fallback generic names
    return ["Player A", "Player B", "Player C", "Player D"]


def _stat_types_for_sport(sport: str) -> List[str]:
    sport = (sport or "").lower()
    if sport in {"basketball", "nba"}:
        return ["points", "rebounds", "assists", "threes"]
    if sport in {"football", "nfl"}:
        return ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
    if sport in {"baseball", "mlb"}:
        return ["total_bases", "hits", "strikeouts", "runs"]
    if sport in {"hockey", "nhl"}:
        return ["shots_on_goal", "points", "assists", "saves"]
    if sport in {"tennis"}:
        return ["aces", "double_faults", "winners", "break_points_saved"]
    if sport in {"soccer", "football_global"}:
        return ["shots", "shots_on_target", "passes", "tackles"]

    # Esports: restrict to allowed stats only; LoL allows assists as well
    if sport in ESPORTS_TITLES:
        return ALLOWED_ESPORTS_STATS_PER_SPORT.get(sport, ALLOWED_ESPORTS_STATS)

    # Fallback
    return ["stat"]


def _matchups_for_sport(sport: str) -> List[str]:
    sport = (sport or "").lower()
    mapping = {
        "basketball": ["LAL vs BOS", "GSW vs DEN", "DAL vs PHX"],
        "football": ["KC vs BUF", "PHI vs SF", "DAL vs GB"],
        "baseball": ["LAD vs NYY", "ATL vs HOU", "BOS vs TOR"],
        "hockey": ["EDM vs TOR", "COL vs VGK", "PIT vs WSH"],
        "tennis": ["Djokovic vs Alcaraz", "Sinner vs Medvedev"],
        "soccer": ["Man City vs Real Madrid", "PSG vs Bayern"],
        # Esports
        "csgo": ["G2 vs FURIA", "FaZe vs NAVI", "Vitality vs ENCE"],
        "league_of_legends": ["T1 vs GEN", "G2 vs FNC", "TES vs JDG"],
        "dota2": ["OG vs Liquid", "Spirit vs EG", "GG vs Tundra"],
        "valorant": ["LOUD vs SEN", "FNATIC vs PRX", "NRG vs EG"],
        "overwatch": ["Shock vs Fuel", "Dragons vs Dynasty"],
        "rocket_league": ["NRG vs G2", "BDS vs Vitality", "Endpoint vs Karmine"]
    }
    # generalize keys
    key = sport
    if sport in {"nba", "nfl", "mlb", "nhl", "football_global"}:
        key = {
            "nba": "basketball",
            "nfl": "football",
            "mlb": "baseball",
            "nhl": "hockey",
            "football_global": "soccer",
        }[sport]
    return mapping.get(key, ["Team A vs Team B"])


def _create_prop(sportsbook: str, sport: str, player: str, stat: str, matchup: str, idx: int) -> Dict:
    # Lines tuned by stat type rough ranges
    base_line = {
        # sports
        "points": (15, 35), "rebounds": (5, 15), "assists": (4, 12), "threes": (2, 6),
        "passing_yards": (200, 330), "rushing_yards": (40, 110), "receiving_yards": (40, 110), "receptions": (3, 9),
        "total_bases": (1, 3), "hits": (0.5, 2.5), "strikeouts": (4, 9), "runs": (0.5, 1.5),
        "shots_on_goal": (2, 5), "saves": (20, 35),
        "aces": (4, 14), "double_faults": (1, 6), "winners": (20, 45), "break_points_saved": (2, 10),
        "shots": (1, 5), "shots_on_target": (1, 3), "passes": (40, 90), "tackles": (1, 5),
    # esports (restricted)
    "combined_map_1_2_kills": (10, 45),
    "combined_map_1_2_headshots": (3, 28),
    "fantasy_points": (20, 100),
    "combined_map_1_2_assists": (8, 30),
        # fallback
        "stat": (1, 10)
    }
    # For esports, ensure stat is one of allowed (sport-specific); otherwise skip by returning marker
    if sport in ESPORTS_TITLES:
        allowed = ALLOWED_ESPORTS_STATS_PER_SPORT.get(sport, ALLOWED_ESPORTS_STATS)
        if stat not in allowed:
            return {"_skip": True}

    low, high = base_line.get(stat, (1, 10))
    line = round(random.uniform(low, high), 1)

    american_odds = random.choice([-137, -119, -110, 100, 105, 115])
    injury_status = random.choice(["healthy", "questionable", "healthy", "healthy"])  # mostly healthy

    prop: Dict = {
        "game_id": f"{sport}_{sportsbook.lower()}_{idx}",
        "player_name": player,
        "stat_type": stat,
        "line": line,
        "odds": american_odds,
        "event_start_time": _now_plus_hours(random.randint(1, 72)),
        "matchup": matchup,
        "sportsbook": sportsbook,
        "recent_form": round(random.uniform(5.0, 9.8), 3),
        "matchup_difficulty": round(random.uniform(4.5, 8.5), 3),
        "injury_status": injury_status,
    }

    # Enrich esports with extra context fields to match UI expectations
    if sport in ESPORTS_TITLES:
        prop.update({
            "map": random.choice(["Mirage", "Inferno", "Overpass", "Ascent", "Bind", "Haven", "King's Row", "Utopia"]),
            "side_preference": random.choice(["T", "CT", "Attack", "Defense", "balanced"]),
            "team_chemistry": round(random.uniform(0.6, 1.0), 3),
            "recent_map_performance": round(random.uniform(0.3, 0.95), 3),
            "opponent_map_ban_rate": round(random.uniform(0.05, 0.35), 3),
        })

    return prop


def _generate_props_for_provider(provider: str, sport: str, count: int = 10) -> List[Dict]:
    players = _sample_players_for_sport(sport)
    stats = _stat_types_for_sport(sport)
    matchups = _matchups_for_sport(sport)
    props: List[Dict] = []
    for i in range(count):
        player = random.choice(players)
        stat = random.choice(stats)
        matchup = random.choice(matchups)
        created = _create_prop(provider, sport, player, stat, matchup, i)
        # Filter out non-allowed esports stats
        if created.get("_skip"):
            continue
        if sport in ESPORTS_TITLES:
            allowed = ALLOWED_ESPORTS_STATS_PER_SPORT.get(sport, ALLOWED_ESPORTS_STATS)
            # Ensure stat_type exists and enforce allowed set
            created["stat_type"] = stat
            if created["stat_type"] not in allowed:
                continue
        props.append(created)
    # Additional safety: filter esports props strictly
    if sport in ESPORTS_TITLES:
        allowed = ALLOWED_ESPORTS_STATS_PER_SPORT.get(sport, ALLOWED_ESPORTS_STATS)
        props = [p for p in props if p.get("stat_type") in allowed]
    return props


def fetch_prizepicks_props(sport: str) -> List[Dict]:
    """Fetch PrizePicks props for a given sport (mocked).

    Args:
        sport: sport or esports key, e.g., 'basketball', 'csgo'.

    Returns:
        List of prop dicts conforming to the app schema.
    """
    try:
        return _generate_props_for_provider("PrizePicks", sport, count=10)
    except Exception:
        # Fail safe
        return []


def fetch_underdog_props(sport: str) -> List[Dict]:
    """Fetch Underdog props for a given sport (mocked).

    Args:
        sport: sport or esports key, e.g., 'football', 'valorant'.

    Returns:
        List of prop dicts conforming to the app schema.
    """
    try:
        return _generate_props_for_provider("Underdog", sport, count=10)
    except Exception:
        # Fail safe
        return []
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