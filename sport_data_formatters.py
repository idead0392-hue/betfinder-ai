"""
Sport-Specific Data Formatters for OpenAI Agent Router
Handles formatting of props, stats, and context data for each sport
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)

class BaseSportFormatter:
    """Base class for sport-specific data formatting"""
    
    def __init__(self, sport_name: str):
        self.sport_name = sport_name
    
    def format_props(self, props_data: List[Dict]) -> List[Dict]:
        """Format props data for the sport"""
        return props_data
    
    def format_stats(self, stats_data: Dict) -> Dict:
        """Format statistical data for the sport"""
        return stats_data
    
    def format_context(self, context_data: Dict) -> Dict:
        """Format context data for the sport"""
        return context_data
    
    def extract_player_stats(self, player_name: str, props_data: List[Dict]) -> Dict:
        """Extract relevant stats for a specific player"""
        return {}

class NBAFormatter(BaseSportFormatter):
    """NBA-specific data formatting"""
    
    def __init__(self):
        super().__init__("NBA")
        self.stat_categories = {
            'points': ['PTS', 'points', 'scoring'],
            'rebounds': ['REB', 'rebounds', 'boards'],
            'assists': ['AST', 'assists', 'dimes'],
            'steals': ['STL', 'steals'],
            'blocks': ['BLK', 'blocks'],
            'threes': ['3PM', '3P', 'threes', 'three_pointers'],
            'turnovers': ['TO', 'turnovers'],
            'minutes': ['MIN', 'minutes']
        }
    
    def format_props(self, props_data: List[Dict]) -> List[Dict]:
        """Format NBA props with enhanced context"""
        formatted_props = []
        
        for prop in props_data:
            formatted_prop = {
                'player': prop.get('Name', prop.get('player', '')),
                'prop': prop.get('Prop', prop.get('prop', '')),
                'line': float(prop.get('Points', prop.get('line', 0))),
                'sport': 'NBA',
                'matchup': self._format_matchup(prop),
                'position': self._extract_position(prop),
                'team': self._extract_team(prop),
                'opponent': self._extract_opponent(prop),
                'home_away': self._determine_home_away(prop),
                'game_time': self._extract_game_time(prop),
                'odds': prop.get('odds', -110)
            }
            formatted_props.append(formatted_prop)
        
        return formatted_props
    
    def format_stats(self, stats_data: Dict) -> Dict:
        """Format NBA statistical data"""
        return {
            'season_averages': stats_data.get('season_avg', {}),
            'last_5_games': stats_data.get('last_5', {}),
            'last_10_games': stats_data.get('last_10', {}),
            'home_stats': stats_data.get('home', {}),
            'away_stats': stats_data.get('away', {}),
            'vs_opponent': stats_data.get('vs_opp', {}),
            'usage_rate': stats_data.get('usage', 0),
            'pace': stats_data.get('pace', 0),
            'def_rating': stats_data.get('def_rating', 0),
            'off_rating': stats_data.get('off_rating', 0)
        }
    
    def format_context(self, context_data: Dict) -> Dict:
        """Format NBA context data"""
        return {
            'injuries': context_data.get('injuries', []),
            'lineup_changes': context_data.get('lineup_changes', []),
            'rest_days': context_data.get('rest_days', 0),
            'back_to_back': context_data.get('back_to_back', False),
            'altitude': context_data.get('altitude', 0),
            'referee_crew': context_data.get('refs', []),
            'weather': 'Indoor',  # NBA is always indoor
            'playoff_implications': context_data.get('playoff_implications', ''),
            'revenge_game': context_data.get('revenge_game', False)
        }
    
    def _format_matchup(self, prop: Dict) -> str:
        """Format NBA matchup string"""
        team = self._extract_team(prop)
        opponent = self._extract_opponent(prop)
        return f"{team} vs {opponent}" if team and opponent else "TBD"
    
    def _extract_position(self, prop: Dict) -> str:
        """Extract player position from prop data"""
        return prop.get('position', prop.get('pos', 'N/A'))
    
    def _extract_team(self, prop: Dict) -> str:
        """Extract team from prop data"""
        return prop.get('team', prop.get('Team', ''))
    
    def _extract_opponent(self, prop: Dict) -> str:
        """Extract opponent from prop data"""
        return prop.get('opponent', prop.get('Opponent', ''))
    
    def _determine_home_away(self, prop: Dict) -> str:
        """Determine if team is home or away"""
        return prop.get('home_away', 'N/A')
    
    def _extract_game_time(self, prop: Dict) -> str:
        """Extract game time"""
        return prop.get('game_time', prop.get('time', ''))

class NFLFormatter(BaseSportFormatter):
    """NFL-specific data formatting"""
    
    def __init__(self):
        super().__init__("NFL")
        self.stat_categories = {
            'passing_yards': ['pass_yds', 'passing_yards'],
            'passing_tds': ['pass_tds', 'passing_touchdowns'],
            'rushing_yards': ['rush_yds', 'rushing_yards'],
            'rushing_tds': ['rush_tds', 'rushing_touchdowns'],
            'receiving_yards': ['rec_yds', 'receiving_yards'],
            'receiving_tds': ['rec_tds', 'receiving_touchdowns'],
            'receptions': ['rec', 'receptions', 'catches'],
            'interceptions': ['int', 'interceptions'],
            'sacks': ['sacks'],
            'tackles': ['tackles']
        }
    
    def format_context(self, context_data: Dict) -> Dict:
        """Format NFL context data"""
        return {
            'injuries': context_data.get('injuries', []),
            'weather': context_data.get('weather', {}),
            'dome_game': context_data.get('dome', False),
            'prime_time': context_data.get('prime_time', False),
            'division_game': context_data.get('division_game', False),
            'playoff_implications': context_data.get('playoff_implications', ''),
            'spread': context_data.get('spread', 0),
            'total': context_data.get('total', 0),
            'surface': context_data.get('surface', 'Grass'),
            'elevation': context_data.get('elevation', 0)
        }

class SoccerFormatter(BaseSportFormatter):
    """Soccer-specific data formatting"""
    
    def __init__(self):
        super().__init__("Soccer")
        self.stat_categories = {
            'goals': ['goals', 'G'],
            'assists': ['assists', 'A'],
            'shots': ['shots', 'SOT'],
            'cards': ['yellow_cards', 'red_cards', 'cards'],
            'corners': ['corners'],
            'fouls': ['fouls']
        }
    
    def format_context(self, context_data: Dict) -> Dict:
        """Format Soccer context data"""
        return {
            'injuries': context_data.get('injuries', []),
            'suspensions': context_data.get('suspensions', []),
            'weather': context_data.get('weather', {}),
            'pitch_conditions': context_data.get('pitch', 'Good'),
            'referee': context_data.get('referee', ''),
            'competition': context_data.get('competition', ''),
            'leg_of_tie': context_data.get('leg', 1),
            'aggregate_score': context_data.get('aggregate', ''),
            'international_break': context_data.get('international_break', False)
        }

class EsportsFormatter(BaseSportFormatter):
    """Esports-specific data formatting (CS:GO, LoL, Dota2, etc.)"""
    
    def __init__(self, game: str):
        super().__init__(f"{game} Esports")
        self.game = game
        
        # Game-specific stat categories
        if game == "CSGO":
            self.stat_categories = {
                'kills': ['kills', 'K'],
                'deaths': ['deaths', 'D'],
                'assists': ['assists', 'A'],
                'adr': ['adr', 'damage'],
                'rating': ['rating', 'hltv_rating'],
                'maps': ['maps_played', 'maps']
            }
        elif game == "LoL":
            self.stat_categories = {
                'kills': ['kills', 'K'],
                'deaths': ['deaths', 'D'],
                'assists': ['assists', 'A'],
                'cs': ['cs', 'creep_score'],
                'gold': ['gold'],
                'damage': ['damage']
            }
        elif game == "Dota2":
            self.stat_categories = {
                'kills': ['kills', 'K'],
                'deaths': ['deaths', 'D'],
                'assists': ['assists', 'A'],
                'last_hits': ['last_hits', 'cs'],
                'gpm': ['gpm', 'gold_per_minute'],
                'xpm': ['xpm', 'experience_per_minute']
            }
    
    def format_context(self, context_data: Dict) -> Dict:
        """Format Esports context data"""
        base_context = {
            'tournament': context_data.get('tournament', ''),
            'match_format': context_data.get('format', 'Bo1'),
            'patch_version': context_data.get('patch', ''),
            'map_pool': context_data.get('maps', []),
            'side_selection': context_data.get('side', ''),
            'recent_form': context_data.get('form', {}),
            'head_to_head': context_data.get('h2h', {}),
            'lan_online': context_data.get('lan_online', 'Online')
        }
        
        # Game-specific additions
        if self.game == "CSGO":
            base_context.update({
                'map_vetos': context_data.get('vetos', []),
                'pistol_round_stats': context_data.get('pistol_stats', {}),
                'economy_stats': context_data.get('economy', {})
            })
        elif self.game == "LoL":
            base_context.update({
                'draft_priority': context_data.get('draft', {}),
                'meta_picks': context_data.get('meta', []),
                'early_game_stats': context_data.get('early_game', {}),
                'objective_control': context_data.get('objectives', {})
            })
        
        return base_context

class SportFormatterFactory:
    """Factory for creating sport-specific formatters"""
    
    @staticmethod
    def create_formatter(sport: str) -> BaseSportFormatter:
        """Create appropriate formatter for sport"""
        sport_lower = sport.lower().strip()
        
        if sport_lower in ['nba', 'basketball']:
            return NBAFormatter()
        elif sport_lower in ['nfl', 'football']:
            return NFLFormatter()
        elif sport_lower in ['soccer', 'football_intl']:
            return SoccerFormatter()
        elif sport_lower in ['csgo', 'cs']:
            return EsportsFormatter("CSGO")
        elif sport_lower in ['league_of_legends', 'lol']:
            return EsportsFormatter("LoL")
        elif sport_lower in ['dota2', 'dota']:
            return EsportsFormatter("Dota2")
        elif sport_lower in ['valorant', 'val']:
            return EsportsFormatter("Valorant")
        elif sport_lower in ['overwatch', 'ow']:
            return EsportsFormatter("Overwatch")
        else:
            # Return base formatter for unsupported sports
            logger.warning(f"No specific formatter for sport: {sport}, using base formatter")
            return BaseSportFormatter(sport)

def format_props_for_sport(sport: str, props_data: List[Dict], 
                          stats_data: Optional[Dict] = None,
                          context_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main function to format props data for a specific sport
    
    Args:
        sport: Sport name (e.g., 'NBA', 'NFL', 'Soccer')
        props_data: List of prop betting data
        stats_data: Optional statistical data
        context_data: Optional context information
    
    Returns:
        Formatted data ready for OpenAI Agent
    """
    
    formatter = SportFormatterFactory.create_formatter(sport)
    
    result = {
        'sport': sport,
        'formatted_at': datetime.now().isoformat(),
        'props': formatter.format_props(props_data),
        'total_props': len(props_data)
    }
    
    if stats_data:
        result['stats'] = formatter.format_stats(stats_data)
    
    if context_data:
        result['context'] = formatter.format_context(context_data)
    
    return result

def extract_prop_insights(formatted_props: List[Dict]) -> Dict[str, Any]:
    """Extract key insights from formatted props data"""
    
    insights = {
        'total_props': len(formatted_props),
        'players': [],
        'prop_types': {},
        'teams': set(),
        'opponents': set(),
        'average_line': 0,
        'line_range': {}
    }
    
    total_line_value = 0
    line_values = []
    
    for prop in formatted_props:
        # Collect players
        player = prop.get('player', '')
        if player and player not in insights['players']:
            insights['players'].append(player)
        
        # Collect prop types
        prop_type = prop.get('prop', '').lower()
        insights['prop_types'][prop_type] = insights['prop_types'].get(prop_type, 0) + 1
        
        # Collect teams
        team = prop.get('team', '')
        if team:
            insights['teams'].add(team)
        
        opponent = prop.get('opponent', '')
        if opponent:
            insights['opponents'].add(opponent)
        
        # Calculate line statistics
        line = prop.get('line', 0)
        if line:
            total_line_value += line
            line_values.append(line)
    
    # Calculate line statistics
    if line_values:
        insights['average_line'] = total_line_value / len(line_values)
        insights['line_range'] = {
            'min': min(line_values),
            'max': max(line_values),
            'median': sorted(line_values)[len(line_values)//2]
        }
    
    # Convert sets to lists for JSON serialization
    insights['teams'] = list(insights['teams'])
    insights['opponents'] = list(insights['opponents'])
    
    return insights

# Example usage and testing
if __name__ == "__main__":
    # Test NBA formatting
    sample_nba_props = [
        {
            'Name': 'LeBron James',
            'Prop': 'Over 24.5 Points',
            'Points': 24.5,
            'team': 'LAL',
            'opponent': 'GSW'
        }
    ]
    
    formatted = format_props_for_sport('NBA', sample_nba_props)
    print("NBA Formatted Props:")
    print(formatted)
    
    # Test insights extraction
    insights = extract_prop_insights(formatted['props'])
    print("\nProp Insights:")
    print(insights)