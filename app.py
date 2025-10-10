import streamlit as st
import pandas as pd
import requests
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from lxml import etree

# Page configuration
st.set_page_config(
    page_title="BetFinder AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for caching
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'cache_timestamp' not in st.session_state:
    st.session_state.cache_timestamp = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Cache duration (5 minutes)
CACHE_DURATION = 300

def is_cache_valid(cache_key):
    """Check if cached data is still valid"""
    if cache_key not in st.session_state.cache_timestamp:
        return False
    elapsed = time.time() - st.session_state.cache_timestamp[cache_key]
    return elapsed < CACHE_DURATION

def get_cached_data(cache_key):
    """Get data from cache if valid"""
    if is_cache_valid(cache_key):
        return st.session_state.data_cache.get(cache_key)
    return None

def set_cached_data(cache_key, data):
    """Store data in cache with timestamp"""
    st.session_state.data_cache[cache_key] = data
    st.session_state.cache_timestamp[cache_key] = time.time()

def load_sports_data_with_agents():
    """Load all sports data using sport agents"""
    if st.session_state.data_loaded:
        return
    
    with st.spinner("Loading sports data using AI agents..."):
        try:
            # Import sport agents
            from sport_agents import (
                BasketballAgent, FootballAgent, TennisAgent, 
                BaseballAgent, HockeyAgent, SoccerAgent, EsportsAgent
            )
            
            # Load data using sport agents
            agents_and_sports = [
                (BasketballAgent(), "basketball"),
                (FootballAgent(), "football"), 
                (TennisAgent(), "tennis"),
                (BaseballAgent(), "baseball"),
                (HockeyAgent(), "hockey"),
                (SoccerAgent(), "soccer"),
                (EsportsAgent(), "esports")
            ]
            
            for agent, sport in agents_and_sports:
                try:
                    picks = agent.make_picks()
                    set_cached_data(f"{sport}_props", {"data": picks})
                except Exception as e:
                    st.error(f"Error loading {sport} data: {e}")
                    set_cached_data(f"{sport}_props", {"data": []})
            
            st.session_state.data_loaded = True
            
        except Exception as e:
            st.error(f"Error loading sports data: {e}")
            # Set empty data to prevent infinite loading
            sports = ["basketball", "football", "tennis", "baseball", "hockey", "soccer", "esports"]
            for sport in sports:
                set_cached_data(f"{sport}_props", {"data": []})

def display_sport_picks(sport_name, picks, sport_emoji):
    """Display picks for a specific sport with esports-style cards"""
    st.markdown(f'<div class="section-title">{sport_name}<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    if picks:
        # Display picks as esports-style cards
        for i, pick in enumerate(picks[:12]):  # Show top 12 picks in grid
            if isinstance(pick, dict):
                # Extract pick data
                player_name = pick.get('player_name', pick.get('description', 'Unknown Player'))
                stat_type = pick.get('stat_type', 'Unknown Stat')
                line = pick.get('line', 'N/A')
                bet_type = pick.get('bet_type', 'over')
                confidence = pick.get('confidence', 0)
                odds = pick.get('odds', -110)
                
                # Get ML prediction data
                edge = 0
                odds_display = f"{abs(odds):,.0f}" if odds else "N/A"
                if 'ml_prediction' in pick:
                    ml_pred = pick['ml_prediction']
                    edge = ml_pred.get('edge', 0)
                
                # Format match/game info
                game_info = pick.get('game', pick.get('matchup', 'TBD vs TBD'))
                event_time = pick.get('event_start_time', 'TBD')
                
                # Determine card styling based on confidence and edge
                if confidence >= 80 or edge > 0.05:
                    card_class = "prop-card-high"
                    odds_class = "odds-high"
                elif confidence >= 70 or edge > 0.02:
                    card_class = "prop-card-medium" 
                    odds_class = "odds-medium"
                else:
                    card_class = "prop-card-low"
                    odds_class = "odds-low"
                
                # Create the esports-style card
                card_html = f"""
                <div class="prop-card {card_class}">
                    <div class="card-header">
                        <div class="team-icon">
                            <div class="team-logo">{sport_emoji}</div>
                        </div>
                        <div class="odds-badge {odds_class}">
                            🔥 {odds_display}
                        </div>
                    </div>
                    <div class="player-info">
                        <div class="team-name">{game_info[:15]}...</div>
                        <div class="player-name">{player_name}</div>
                        <div class="match-details">vs {event_time}</div>
                    </div>
                    <div class="stat-line">
                        <div class="stat-number">{line}</div>
                        <div class="stat-type">{stat_type.upper()} {bet_type.upper()}</div>
                    </div>
                    <div class="card-footer">
                        <button class="less-btn">↓ Less</button>
                        <div class="confidence-indicator">
                            <span class="confidence-text">{confidence}%</span>
                            {"<span class='edge-text'>+" + f"{edge:.1%}</span>" if edge > 0.01 else ""}
                        </div>
                        <button class="more-btn">↑ More</button>
                    </div>
                </div>
                """
                
                # Display card in columns (3 cards per row)
                if i % 3 == 0:
                    cols = st.columns(3)
                
                with cols[i % 3]:
                    st.markdown(card_html, unsafe_allow_html=True)
        
        # Show additional statistics
        st.markdown("---")
        st.markdown(f"### 📊 {sport_name} Analytics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_picks = len(picks)
            st.metric("Total Picks", total_picks)
        
        with col2:
            high_conf_picks = len([p for p in picks if p.get('confidence', 0) >= 80])
            st.metric("High Confidence", high_conf_picks)
        
        with col3:
            avg_confidence = sum(p.get('confidence', 0) for p in picks) / len(picks) if picks else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
    else:
        st.info(f"{sport_emoji} Loading {sport_name.lower()} props...")
        st.button(f"Refresh {sport_name} Data", key=f"refresh_{sport_name.lower()}")

# CSS for esports-style cards
st.markdown("""
<style>
    .section-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .time {
        font-size: 14px;
        opacity: 0.8;
    }
    
    /* Esports-style prop cards */
    .prop-card {
        background: linear-gradient(135deg, #2a2a3a 0%, #1a1a2e 100%);
        border: 1px solid #444;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prop-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        border-color: #666;
    }
    
    .prop-card-high {
        border-color: #00ff88 !important;
        box-shadow: 0 4px 12px rgba(0, 255, 136, 0.2);
    }
    
    .prop-card-medium {
        border-color: #ffaa00 !important;
        box-shadow: 0 4px 12px rgba(255, 170, 0, 0.2);
    }
    
    .prop-card-low {
        border-color: #ff4444 !important;
        box-shadow: 0 4px 12px rgba(255, 68, 68, 0.2);
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .team-icon {
        display: flex;
        align-items: center;
    }
    
    .team-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(45deg, #0099ff, #00ccff);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        border: 2px solid #333;
    }
    
    .odds-badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    
    .odds-high {
        background: linear-gradient(45deg, #00ff88, #00cc6a);
    }
    
    .odds-medium {
        background: linear-gradient(45deg, #ffaa00, #ff8800);
    }
    
    .odds-low {
        background: linear-gradient(45deg, #ff4444, #cc0000);
    }
    
    .player-info {
        text-align: center;
        margin-bottom: 16px;
    }
    
    .team-name {
        color: #999;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .player-name {
        color: white;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 4px;
    }
    
    .match-details {
        color: #666;
        font-size: 10px;
        text-transform: uppercase;
    }
    
    .stat-line {
        text-align: center;
        margin-bottom: 16px;
        padding: 12px 0;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
    }
    
    .stat-number {
        color: #00ff88;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 4px;
    }
    
    .stat-type {
        color: #ccc;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .less-btn, .more-btn {
        background: transparent;
        border: 1px solid #444;
        color: #888;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 11px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .less-btn:hover, .more-btn:hover {
        border-color: #666;
        color: #ccc;
    }
    
    .confidence-indicator {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
    }
    
    .confidence-text {
        color: white;
        font-size: 12px;
        font-weight: bold;
    }
    
    .edge-text {
        color: #00ff88;
        font-size: 10px;
        font-weight: bold;
    }
    
    /* Dark theme overrides */
    .stApp {
        background-color: #0a0a0a;
    }
    
    .stMarkdown h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1>🎯 BetFinder AI</h1>
    <p style="font-size: 18px; color: #666;">Advanced Sports Betting Analysis with ML Predictions</p>
</div>
""", unsafe_allow_html=True)

# Load data
load_sports_data_with_agents()


# New tab names: add College Football and individual esports
tab_names = [
    "🏀 Basketball", "🏈 Football", "🎾 Tennis", "⚾ Baseball", "🏒 Hockey", "⚽ Soccer", "� College Football",
    "🔫 CSGO", "🧙 League of Legends", "🐉 Dota2", "🎯 Valorant", "🛡️ Overwatch", "🚗 Rocket League"
]
tabs = st.tabs(tab_names)

# Load PickFinder CSV and group props by sport/esport
def load_pickfinder_csv(csv_path: str) -> dict:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    props = []
    for _, row in df.iterrows():
        prop = {
            'player_name': row.get('Player', ''),
            'team': row.get('Team/Opp', ''),
            'pick': row.get('Prop', ''),
            'stat_type': row.get('Prop', '').lower(),
            'line': float(row.get('Line', 0)) if row.get('Line', '') else 0.0,
            'odds': int(row.get('Odds', -110)) if str(row.get('Odds', '')).replace('+','').replace('-','').isdigit() else -110,
            'confidence': float(row.get('IP', 50)) if row.get('IP', '') else 50.0,
            'expected_value': float(row.get('Diff', 0)) if row.get('Diff', '') else 0.0,
            'avg_l10': float(row.get('Avg_L10', 0)) if row.get('Avg_L10', '') else 0.0,
            'start_time': '',
            'sport': '',
            'over_under': 'over' if 'over' in str(row.get('Prop', '')).lower() else 'under' if 'under' in str(row.get('Prop', '')).lower() else None
        }
        props.append(prop)
    # Group by sport/esport
    grouped = {k: [] for k in [
        'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'rocket_league']}
    for p in props:
        stat = str(p.get('stat_type', '')).lower()
        # Heuristic mapping
        if 'college' in stat or 'ncaa' in stat:
            grouped['college_football'].append(p)
        elif 'csgo' in stat or 'kill' in stat or 'headshot' in stat or 'fantasy' in stat:
            grouped['csgo'].append(p)
        elif 'league' in stat or 'lol' in stat or 'assist' in stat:
            grouped['league_of_legends'].append(p)
        elif 'dota' in stat:
            grouped['dota2'].append(p)
        elif 'valorant' in stat:
            grouped['valorant'].append(p)
        elif 'overwatch' in stat:
            grouped['overwatch'].append(p)
        elif 'rocket' in stat:
            grouped['rocket_league'].append(p)
        elif 'basket' in stat or 'point' in stat or 'rebound' in stat or 'assist' in stat or 'block' in stat or 'steal' in stat or 'three' in stat:
            grouped['basketball'].append(p)
        elif 'foot' in stat or 'yard' in stat or 'touchdown' in stat or 'reception' in stat or 'passing' in stat or 'rushing' in stat:
            grouped['football'].append(p)
        elif 'base' in stat or 'hit' in stat or 'run' in stat or 'home_run' in stat or 'strikeout' in stat:
            grouped['baseball'].append(p)
        elif 'hock' in stat or 'goal' in stat or 'shot' in stat or 'penalty' in stat:
            grouped['hockey'].append(p)
        elif 'soccer' in stat or 'card' in stat or 'goal' in stat or 'assist' in stat:
            grouped['soccer'].append(p)
        elif 'tennis' in stat:
            grouped['tennis'].append(p)
    return grouped

csv_props = load_pickfinder_csv('pickfinder_all_projections.csv')

# Assign agents for all tabs
from sport_agents import (
    BasketballAgent, FootballAgent, TennisAgent, BaseballAgent, HockeyAgent, SoccerAgent,
    CollegeFootballAgent, CSGOAgent, LeagueOfLegendsAgent, Dota2Agent, VALORANTAgent, OverwatchAgent, RocketLeagueAgent
)
agents = [
    (BasketballAgent(), 'basketball', '🏀'),
    (FootballAgent(), 'football', '🏈'),
    (TennisAgent(), 'tennis', '🎾'),
    (BaseballAgent(), 'baseball', '⚾'),
    (HockeyAgent(), 'hockey', '🏒'),
    (SoccerAgent(), 'soccer', '⚽'),
    (CollegeFootballAgent(), 'college_football', '🎓'),
    (CSGOAgent(), 'csgo', '🔫'),
    (LeagueOfLegendsAgent(), 'league_of_legends', '🧙'),
    (Dota2Agent(), 'dota2', '🐉'),
    (VALORANTAgent(), 'valorant', '🎯'),
    (OverwatchAgent(), 'overwatch', '🛡️'),
    (RocketLeagueAgent(), 'rocket_league', '🚗')
]

for i, (agent, key, emoji) in enumerate(agents):
    with tabs[i]:
        props = csv_props.get(key, [])
        if not props:
            # fallback to agent's own props if CSV empty
            props = agent.make_picks()
        else:
            props = agent.make_picks(props_data=props)
        display_sport_picks(agent.__class__.__name__.replace('Agent',''), props, emoji)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 30px;">
    <p>🤖 Powered by AI Sport Agents with ML Prediction Engine</p>
    <p>📊 Real-time prop analysis with confidence scoring and edge calculation</p>
</div>
""", unsafe_allow_html=True)