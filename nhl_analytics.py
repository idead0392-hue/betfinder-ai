"""
NHL Analytics Integration using wsba-hockey
Advanced NHL data analysis and prediction capabilities for BetFinder AI
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe wsba-hockey import with fallback
try:
    import wsba_hockey as wsba
    WSBA_AVAILABLE = True
    logger.info("✅ wsba-hockey package loaded successfully")
except Exception as e:
    WSBA_AVAILABLE = False
    logger.warning(f"⚠️ wsba-hockey not available: {e}")
    logger.info("🔄 Falling back to mock NHL analytics")

class NHLAnalyticsEngine:
    """Advanced NHL analytics using wsba-hockey for enhanced prop predictions"""
    
    def __init__(self, cache_dir: str = "analytics_data/hockey"):
        self.cache_dir = cache_dir
        self.season = self._get_current_season()
        self._ensure_cache_dir()
        
        # Mock data for fallback mode
        self.mock_players = {
            'connor mcdavid': {
                'goals_per_game': 1.2,
                'assists_per_game': 1.8,
                'points_per_game': 3.0,
                'shots_per_game': 4.5,
                'shot_attempts_per_game': 6.2,
                'xg_per_game': 1.1
            },
            'leon draisaitl': {
                'goals_per_game': 1.1,
                'assists_per_game': 1.5,
                'points_per_game': 2.6,
                'shots_per_game': 3.8,
                'shot_attempts_per_game': 5.5,
                'xg_per_game': 0.9
            },
            'david pastrnak': {
                'goals_per_game': 1.0,
                'assists_per_game': 1.2,
                'points_per_game': 2.2,
                'shots_per_game': 4.2,
                'shot_attempts_per_game': 5.8,
                'xg_per_game': 0.8
            },
            'erik karlsson': {
                'goals_per_game': 0.3,
                'assists_per_game': 1.4,
                'points_per_game': 1.7,
                'shots_per_game': 3.5,
                'shot_attempts_per_game': 4.8,
                'xg_per_game': 0.25
            }
        }
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_current_season(self) -> int:
        """Get current NHL season ID (e.g., 20242025)"""
        now = datetime.now()
        if now.month >= 10:  # Season starts in October
            return int(f"{now.year}{now.year + 1}")
        else:
            return int(f"{now.year - 1}{now.year}")
    
    def get_schedule(self, season: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get NHL schedule for analysis"""
        if not WSBA_AVAILABLE:
            logger.warning("wsba-hockey not available, using mock schedule")
            # Return mock schedule data
            return pd.DataFrame([
                {
                    'game_id': 2024020918,
                    'date': '2024-10-12',
                    'home_team': 'EDM',
                    'away_team': 'BOS',
                    'status': 'scheduled'
                }
            ])
            
        try:
            season = season or self.season
            cache_file = f"{self.cache_dir}/schedule_{season}.json"
            
            # Check cache first
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    schedule_data = json.load(f)
                logger.info(f"📅 Loaded cached schedule for {season}")
                return pd.DataFrame(schedule_data)
            
            # Fetch fresh data
            logger.info(f"🔄 Fetching NHL schedule for season {season}")
            schedule = wsba.nhl_scrape_schedule(season)
            
            if schedule is not None:
                # Cache the data
                schedule_dict = schedule.to_dict('records')
                with open(cache_file, 'w') as f:
                    json.dump(schedule_dict, f, indent=2, default=str)
                logger.info(f"✅ Cached schedule: {len(schedule)} games")
                return schedule
            
        except Exception as e:
            logger.error(f"❌ Error fetching schedule: {e}")
            
        return None
    
    def get_game_pbp(self, game_id: int) -> Optional[pd.DataFrame]:
        """Get play-by-play data for a specific game"""
        if not WSBA_AVAILABLE:
            logger.warning("wsba-hockey not available, skipping PBP")
            return None
            
        try:
            cache_file = f"{self.cache_dir}/game_{game_id}_pbp.json"
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    pbp_data = json.load(f)
                return pd.DataFrame(pbp_data)
            
            # Fetch fresh data
            logger.info(f"🏒 Fetching play-by-play for game {game_id}")
            pbp = wsba.nhl_scrape_game(game_id)
            
            if pbp is not None:
                # Cache the data
                pbp_dict = pbp.to_dict('records')
                with open(cache_file, 'w') as f:
                    json.dump(pbp_dict, f, indent=2, default=str)
                logger.info(f"✅ Cached PBP: {len(pbp)} events")
                return pbp
                
        except Exception as e:
            logger.error(f"❌ Error fetching game PBP: {e}")
            
        return None
    
    def get_xg_analysis(self, pbp_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply expected goals (xG) analysis to play-by-play data"""
        if not WSBA_AVAILABLE or pbp_data is None:
            return None
            
        try:
            logger.info("📊 Applying xG analysis")
            xg_data = wsba.nhl_apply_xG(pbp_data)
            logger.info("✅ xG analysis complete")
            return xg_data
        except Exception as e:
            logger.error(f"❌ Error in xG analysis: {e}")
            return None
    
    def get_skater_stats(self, season: Optional[int] = None, 
                        game_types: List[int] = [2], 
                        situations: List[str] = ['5v5', '4v4', '3v3']) -> Optional[pd.DataFrame]:
        """Get aggregated skater statistics"""
        if not WSBA_AVAILABLE:
            logger.warning("wsba-hockey not available, using mock stats")
            # Return mock player stats
            mock_stats = []
            for player, stats in self.mock_players.items():
                mock_stats.append({
                    'player_name': player.title(),
                    **stats
                })
            return pd.DataFrame(mock_stats)
            
        try:
            season = season or self.season
            cache_file = f"{self.cache_dir}/skater_stats_{season}.json"
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    stats_data = json.load(f)
                logger.info(f"📈 Loaded cached skater stats for {season}")
                return pd.DataFrame(stats_data)
            
            # Get season play-by-play first
            logger.info(f"🔄 Fetching season PBP for {season}")
            season_pbp = wsba.nhl_scrape_season(season)
            
            if season_pbp is not None:
                # Calculate skater stats
                logger.info("📊 Calculating skater statistics")
                skater_stats = wsba.nhl_calculate_stats(
                    season_pbp, 'skater', game_types, situations, shot_impact=True
                )
                
                if skater_stats is not None:
                    # Cache the data
                    stats_dict = skater_stats.to_dict('records')
                    with open(cache_file, 'w') as f:
                        json.dump(stats_dict, f, indent=2, default=str)
                    logger.info(f"✅ Cached skater stats: {len(skater_stats)} players")
                    return skater_stats
                    
        except Exception as e:
            logger.error(f"❌ Error fetching skater stats: {e}")
            
        return None
    
    def analyze_player_performance(self, player_name: str, stat_type: str) -> Dict[str, Any]:
        """Analyze player performance for specific stat type"""
        analysis = {
            'player': player_name,
            'stat_type': stat_type,
            'confidence': 0.5,
            'prediction': None,
            'factors': []
        }
        
        try:
            # Get skater stats (real or mock)
            skater_stats = self.get_skater_stats()
            if skater_stats is None:
                return analysis
            
            # Find player data
            player_data = skater_stats[
                skater_stats['player_name'].str.contains(player_name, case=False, na=False)
            ]
            
            if player_data.empty:
                logger.warning(f"⚠️ No data found for player: {player_name}")
                return analysis
            
            player_row = player_data.iloc[0]
            
            # Analyze based on stat type
            if stat_type.lower() in ['goals', 'goal']:
                goals_per_game = player_row.get('goals_per_game', 0)
                xg_per_game = player_row.get('xg_per_game', 0)
                
                analysis['prediction'] = goals_per_game
                analysis['confidence'] = min(0.9, max(0.3, goals_per_game * 2))
                analysis['factors'].append(f"Goals/game: {goals_per_game:.2f}")
                analysis['factors'].append(f"xG/game: {xg_per_game:.2f}")
                
            elif stat_type.lower() in ['shots', 'shot']:
                shots_per_game = player_row.get('shots_per_game', 0)
                shot_attempts_per_game = player_row.get('shot_attempts_per_game', 0)
                
                analysis['prediction'] = shots_per_game
                analysis['confidence'] = min(0.9, max(0.4, shots_per_game * 0.2))
                analysis['factors'].append(f"Shots/game: {shots_per_game:.2f}")
                analysis['factors'].append(f"Shot attempts/game: {shot_attempts_per_game:.2f}")
                
            elif stat_type.lower() in ['assists', 'assist']:
                assists_per_game = player_row.get('assists_per_game', 0)
                
                analysis['prediction'] = assists_per_game
                analysis['confidence'] = min(0.9, max(0.3, assists_per_game * 1.5))
                analysis['factors'].append(f"Assists/game: {assists_per_game:.2f}")
                
            elif stat_type.lower() in ['points', 'point']:
                points_per_game = player_row.get('points_per_game', 0)
                
                analysis['prediction'] = points_per_game
                analysis['confidence'] = min(0.9, max(0.4, points_per_game * 0.8))
                analysis['factors'].append(f"Points/game: {points_per_game:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing player performance: {e}")
            
        return analysis
    
    def get_recent_games(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get recent games for analysis"""
        if not WSBA_AVAILABLE:
            # Return mock recent games
            return [{
                'game_id': 2024020918,
                'date': '2024-10-12',
                'home_team': 'EDM',
                'away_team': 'BOS',
                'status': 'scheduled'
            }]
            
        try:
            schedule = self.get_schedule()
            if schedule is None:
                return []
            
            # Filter recent games
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_games = []
            
            for _, game in schedule.iterrows():
                game_date = pd.to_datetime(game.get('date', ''))
                if game_date >= cutoff_date:
                    recent_games.append({
                        'game_id': game.get('game_id'),
                        'date': game.get('date'),
                        'home_team': game.get('home_team'),
                        'away_team': game.get('away_team'),
                        'status': game.get('status', 'scheduled')
                    })
            
            return sorted(recent_games, key=lambda x: x['date'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error getting recent games: {e}")
            return []
    
    def enhance_prop_prediction(self, prop_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance prop prediction using NHL analytics"""
        enhanced_prop = prop_data.copy()
        
        try:
            player_name = prop_data.get('player_name', '')
            stat_type = prop_data.get('stat_type', '')
            
            if not player_name or not stat_type:
                return enhanced_prop
            
            # Get player analysis
            analysis = self.analyze_player_performance(player_name, stat_type)
            
            # Enhance with analytics
            enhanced_prop['nhl_analytics'] = {
                'wsba_prediction': analysis.get('prediction'),
                'wsba_confidence': analysis.get('confidence'),
                'wsba_factors': analysis.get('factors', []),
                'enhanced_confidence': min(0.95, 
                    (enhanced_prop.get('confidence', 0.5) + analysis.get('confidence', 0.5)) / 2
                ),
                'data_source': 'wsba-hockey' if WSBA_AVAILABLE else 'mock_data'
            }
            
            # Update overall confidence
            enhanced_prop['confidence'] = enhanced_prop['nhl_analytics']['enhanced_confidence']
            
            logger.info(f"🏒 Enhanced {player_name} {stat_type} prediction: "
                       f"{enhanced_prop['confidence']:.2f} confidence")
            
        except Exception as e:
            logger.error(f"❌ Error enhancing prop prediction: {e}")
            
        return enhanced_prop

def create_nhl_analytics_demo():
    """Create demo script for NHL analytics"""
    demo_code = f'''
# NHL Analytics Demo using wsba-hockey
from nhl_analytics import NHLAnalyticsEngine

# Initialize the analytics engine
nhl = NHLAnalyticsEngine()

# Get current season schedule
schedule = nhl.get_schedule()
print(f"📅 Schedule loaded: {{len(schedule) if schedule is not None else 0}} games")

# Get recent games
recent_games = nhl.get_recent_games(days_back=3)
print(f"🏒 Recent games: {{len(recent_games)}}")

# Analyze a player (example)
analysis = nhl.analyze_player_performance("Connor McDavid", "goals")
print(f"🥅 McDavid analysis: {{analysis['confidence']:.2f}} confidence")

# Enhance a prop prediction
prop_example = {{
    'player_name': 'Leon Draisaitl', 
    'stat_type': 'points', 
    'line': 1.5,
    'confidence': 0.6
}}
enhanced = nhl.enhance_prop_prediction(prop_example)
print(f"📈 Enhanced prediction: {{enhanced['confidence']:.2f}}")

# Data source info
print(f"📊 Using data source: {{'wsba-hockey' if {WSBA_AVAILABLE} else 'mock_data'}}")
'''
    
    with open('/workspaces/betfinder-ai/nhl_analytics_demo.py', 'w') as f:
        f.write(demo_code)

if __name__ == "__main__":
    # Create demo
    create_nhl_analytics_demo()
    
    # Test basic functionality
    print("🏒 Testing NHL Analytics Engine...")
    engine = NHLAnalyticsEngine()
    
    # Test recent games
    recent = engine.get_recent_games(days_back=1)
    print(f"📅 Found {len(recent)} recent games")
    
    # Test player analysis
    analysis = engine.analyze_player_performance("Connor McDavid", "goals")
    print(f"🥅 McDavid analysis: {analysis['confidence']:.2f} confidence")
    
    print("✅ NHL Analytics Engine ready!")