
# NHL Analytics Demo using wsba-hockey
from nhl_analytics import NHLAnalyticsEngine

# Initialize the analytics engine
nhl = NHLAnalyticsEngine()

# Get current season schedule
schedule = nhl.get_schedule()
print(f"📅 Schedule loaded: {len(schedule) if schedule is not None else 0} games")

# Get recent games
recent_games = nhl.get_recent_games(days_back=3)
print(f"🏒 Recent games: {len(recent_games)}")

# Analyze a player (example)
analysis = nhl.analyze_player_performance("Connor McDavid", "goals")
print(f"🥅 McDavid analysis: {analysis['confidence']:.2f} confidence")

# Enhance a prop prediction
prop_example = {
    'player_name': 'Leon Draisaitl', 
    'stat_type': 'points', 
    'line': 1.5,
    'confidence': 0.6
}
enhanced = nhl.enhance_prop_prediction(prop_example)
print(f"📈 Enhanced prediction: {enhanced['confidence']:.2f}")

# Data source info
print(f"📊 Using data source: {'wsba-hockey' if False else 'mock_data'}")
