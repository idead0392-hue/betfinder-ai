import streamlit as st
import pandas as pd
import requests
import os
import time
import threading
import importlib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from lxml import etree
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

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
if 'auto_scraper_started' not in st.session_state:
    st.session_state.auto_scraper_started = False
if 'csv_last_mtime' not in st.session_state:
    st.session_state.csv_last_mtime = 0.0
if 'auto_scrape_interval_sec' not in st.session_state:
    # Default: scrape every 60 seconds for near real-time updates
    st.session_state.auto_scrape_interval_sec = int(os.environ.get('AUTO_SCRAPE_INTERVAL_SEC', '60'))

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

def _run_prizepicks_scraper_loop(csv_path: str, interval_sec: int = 300):
    """Background loop to refresh PrizePicks CSV on an interval."""
    # Import locally to avoid import-time overhead on main thread
    from prizepicks_scrape import main as scrape_main
    while True:
        try:
            # Ensure scraper writes to the same path the app reads
            os.environ['PRIZEPICKS_CSV'] = csv_path
            scrape_main()
        except Exception:
            # Keep loop alive even on failures
            pass
        finally:
            try:
                # Small guard sleep before checking/writing again
                time.sleep(max(10, int(interval_sec)))
            except Exception:
                time.sleep(300)

def start_auto_scraper(csv_path: str, interval_sec: int = 300):
    """Start the background auto-scraper thread once per session."""
    if not st.session_state.auto_scraper_started:
        t = threading.Thread(target=_run_prizepicks_scraper_loop, args=(csv_path, interval_sec), daemon=True)
        t.start()
        st.session_state.auto_scraper_started = True

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

# Sidebar: Automation status (no manual upload/refresh)
st.sidebar.header("Live Data")
if 'prizepicks_csv_path' not in st.session_state:
    st.session_state['prizepicks_csv_path'] = 'prizepicks_props.csv'

st.sidebar.success("Automatic mode enabled. Scraper runs in the background and UI refreshes automatically.")
st.sidebar.write(f"CSV path: `{st.session_state['prizepicks_csv_path']}`")
last_ts = st.session_state.get('csv_last_mtime', 0)
if last_ts:
    st.sidebar.write(f"Last update: {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.sidebar.write("Last update: pending…")
st.sidebar.write(f"Interval: {st.session_state.get('auto_scrape_interval_sec', 300)}s")

# Start auto-scraper in the background (no manual uploads or refreshes needed)
effective_csv_path = st.session_state.get('prizepicks_csv_path', 'prizepicks_props.csv')

# Kick off an immediate scrape on startup if file missing or stale (>2*interval)
def _ensure_fresh_csv(path: str, max_age_sec: int):
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0
    age = time.time() - mtime if mtime else 1e9
    if age > max(30, 2 * max_age_sec):
        try:
            os.environ['PRIZEPICKS_CSV'] = path
            from prizepicks_scrape import main as scrape_main
            scrape_main()
        except Exception:
            pass

_ensure_fresh_csv(effective_csv_path, st.session_state.auto_scrape_interval_sec)
start_auto_scraper(effective_csv_path, st.session_state.auto_scrape_interval_sec)

# Auto-refresh trigger: prefer streamlit_autorefresh; no full-page HTML reload
st_autorefresh_counter = st.experimental_memo.clear if False else None  # placeholder to keep linter calm
st_autorefresh = None
try:
    spec = importlib.util.find_spec("streamlit_autorefresh")
    if spec is not None:
        st_autorefresh = importlib.import_module("streamlit_autorefresh").st_autorefresh
except Exception:
    st_autorefresh = None

if st_autorefresh is not None:
    st_autorefresh(interval=20000, key="auto_refresh_interval")


# =============== Live Props JSON server (polling by client) ===============

PROPS_ENDPOINT_PORT = int(os.environ.get('PROPS_ENDPOINT_PORT', '8765'))

_last_props_payload = {"mtime": 0, "by_sport": {}}
_server_started = False

class PropsRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, code=200):
        payload = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        # CORS for local browser use
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self._send_json({}, 200)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/props.json':
            global _last_props_payload
            try:
                # Serve latest cached payload
                self._send_json(_last_props_payload, 200)
            except Exception:
                self._send_json({"error": "internal"}, 500)
        else:
            self._send_json({"error": "not found"}, 404)


def _start_props_server():
    global _server_started
    if _server_started:
        return
    def _serve():
        try:
            httpd = ThreadingHTTPServer(('127.0.0.1', PROPS_ENDPOINT_PORT), PropsRequestHandler)
            httpd.serve_forever()
        except Exception:
            pass
    th = threading.Thread(target=_serve, daemon=True)
    th.start()
    _server_started = True

# New tab names: add College Football and individual esports
tab_names = [
    "🏀 Basketball", "🏈 Football", "🎾 Tennis", "⚾ Baseball", "🏒 Hockey", "⚽ Soccer", "🎓 College Football",
    "🔫 CSGO", "🧙 League of Legends", "🐉 Dota2", "🎯 Valorant", "🛡️ Overwatch", "🚗 Rocket League"
]
tabs = st.tabs(tab_names)

# Load PrizePicks CSV and group props by sport/esport (cached for 5 minutes)
def load_prizepicks_csv_cached(csv_path: str) -> dict:
    # Incorporate file path and mtime into cache key so cache updates when CSV changes
    try:
        mtime = os.path.getmtime(csv_path)
    except Exception:
        mtime = 0
    cache_key = f"prizepicks_csv_props::{csv_path}::{mtime}"
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        set_cached_data(cache_key, {})
        return {}
    props = []
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get('name') or cols.get('player')
    line_col = cols.get('points') or cols.get('line')
    prop_col = cols.get('prop')
    sport_col = cols.get('sport')  # optional
    for _, row in df.iterrows():
        player_name = row.get(name_col, '') if name_col else ''
        stat_type = str(row.get(prop_col, '')).strip() if prop_col else ''
        line_val = row.get(line_col, '') if line_col else ''
        try:
            line = float(line_val) if line_val != '' else 0.0
        except Exception:
            try:
                line = float(str(line_val).split()[0])
            except Exception:
                line = 0.0
        sport_val = str(row.get(sport_col, '')).strip().lower() if sport_col else ''
        prop = {
            'player_name': player_name,
            'team': '',
            'pick': stat_type,
            'stat_type': stat_type.lower(),
            'line': line,
            'odds': -110,
            'confidence': 50.0,
            'expected_value': 0.0,
            'avg_l10': 0.0,
            'start_time': '',
            'sport': sport_val,
            'over_under': None
        }
        props.append(prop)
    grouped = {k: [] for k in [
        'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'rocket_league']}
    def map_to_sport(p: dict) -> str:
        if p.get('sport'):
            s = p['sport']
            aliases = {
                'nba': 'basketball', 'wnba': 'basketball', 'nfl': 'football', 'cfb': 'college_football', 'cbb': 'basketball',
                'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer', 'lol': 'league_of_legends', 'cs2': 'csgo', 'cs:go': 'csgo'
            }
            return aliases.get(s, s)
        stat = str(p.get('stat_type', '')).lower()
        if any(k in stat for k in ['csgo', 'cs2', 'kill', 'headshot']):
            return 'csgo'
        if any(k in stat for k in ['league', 'lol', 'assists per game (lol)']):
            return 'league_of_legends'
        if 'dota' in stat:
            return 'dota2'
        if 'valorant' in stat:
            return 'valorant'
        if 'overwatch' in stat:
            return 'overwatch'
        if 'rocket' in stat:
            return 'rocket_league'
        if any(k in stat for k in ['point', 'rebound', 'assist', 'block', 'steal', '3pt', 'three']):
            return 'basketball'
        if any(k in stat for k in ['passing', 'rushing', 'receiving', 'touchdown', 'yards', 'receptions']):
            return 'football'
        if any(k in stat for k in ['hits', 'home run', 'strikeout', 'total bases', 'rbis']):
            return 'baseball'
        if any(k in stat for k in ['shots on goal', 'goal', 'assist (nhl)', 'saves']):
            return 'hockey'
        if any(k in stat for k in ['goals', 'cards', 'clean sheets']):
            return 'soccer'
        if 'ace' in stat or 'double fault' in stat:
            return 'tennis'
        return 'basketball'
    for p in props:
        sport_key = map_to_sport(p)
        if sport_key in grouped:
            grouped[sport_key].append(p)
    set_cached_data(cache_key, grouped)
    return grouped

# Track mtime and surface a small status indicator
try:
    current_mtime = os.path.getmtime(effective_csv_path)
except Exception:
    current_mtime = 0.0

if current_mtime != st.session_state.csv_last_mtime:
    st.session_state.csv_last_mtime = current_mtime
    if current_mtime:
        st.toast(f"PrizePicks CSV updated at {datetime.fromtimestamp(current_mtime).strftime('%H:%M:%S')}", icon="✅")

csv_props = load_prizepicks_csv_cached(effective_csv_path)

# Compute per-sport props HTML using agents (no ledger logging for live view)
from sport_agents import (
    BasketballAgent, FootballAgent, TennisAgent, BaseballAgent, HockeyAgent, SoccerAgent,
    CollegeFootballAgent, CSGOAgent, LeagueOfLegendsAgent, Dota2Agent, VALORANTAgent, OverwatchAgent, RocketLeagueAgent
)
agent_classes = {
    'basketball': BasketballAgent,
    'football': FootballAgent,
    'tennis': TennisAgent,
    'baseball': BaseballAgent,
    'hockey': HockeyAgent,
    'soccer': SoccerAgent,
    'college_football': CollegeFootballAgent,
    'csgo': CSGOAgent,
    'league_of_legends': LeagueOfLegendsAgent,
    'dota2': Dota2Agent,
    'valorant': VALORANTAgent,
    'overwatch': OverwatchAgent,
    'rocket_league': RocketLeagueAgent,
}

# Render to HTML cards for client-side reuse

def render_prop_card_html(prop: dict, sport_emoji: str) -> str:
    player_name = prop.get('player_name', prop.get('description', 'Unknown Player'))
    stat_type = prop.get('stat_type', 'Unknown Stat')
    line = prop.get('line', 'N/A')
    bet_type = prop.get('bet_type', 'over')
    confidence = prop.get('confidence', 0)
    odds = prop.get('odds', -110)
    odds_display = f"{abs(odds):,.0f}" if odds else "N/A"
    edge = 0
    if 'ml_prediction' in prop:
        edge = prop['ml_prediction'].get('edge', 0)
    game_info = prop.get('game', prop.get('matchup', 'TBD vs TBD'))
    event_time = prop.get('event_start_time', 'TBD')
    if confidence >= 80 or edge > 0.05:
        card_class = "prop-card-high"; odds_class = "odds-high"
    elif confidence >= 70 or edge > 0.02:
        card_class = "prop-card-medium"; odds_class = "odds-medium"
    else:
        card_class = "prop-card-low"; odds_class = "odds-low"
    return f"""
    <div class=\"prop-card {card_class}\">
        <div class=\"card-header\">
            <div class=\"team-icon\"><div class=\"team-logo\">{sport_emoji}</div></div>
            <div class=\"odds-badge {odds_class}\">🔥 {odds_display}</div>
        </div>
        <div class=\"player-info\">
            <div class=\"team-name\">{game_info[:15]}...</div>
            <div class=\"player-name\">{player_name}</div>
            <div class=\"match-details\">vs {event_time}</div>
        </div>
        <div class=\"stat-line\">
            <div class=\"stat-number\">{line}</div>
            <div class=\"stat-type\">{stat_type.upper()} {bet_type.upper()}</div>
        </div>
        <div class=\"card-footer\">
            <button class=\"less-btn\">↓ Less</button>
            <div class=\"confidence-indicator\">
                <span class=\"confidence-text\">{confidence}%</span>
                {"<span class='edge-text'>+" + f"{edge:.1%}</span>" if edge > 0.01 else ""}
            </div>
            <button class=\"more-btn\">↑ More</button>
        </div>
    </div>
    """

# Build per-sport HTML blocks
sport_emojis = {
    'basketball': '🏀', 'football': '🏈', 'tennis': '🎾', 'baseball': '⚾', 'hockey': '🏒', 'soccer': '⚽',
    'college_football': '🎓', 'csgo': '🔫', 'league_of_legends': '🧙', 'dota2': '🐉', 'valorant': '🎯', 'overwatch': '🛡️', 'rocket_league': '🚗'
}

by_sport_html = {}
for key, cls in agent_classes.items():
    props = csv_props.get(key, [])
    if props:
        agent = cls()
        try:
            picks = agent.make_picks(props_data=props, log_to_ledger=False)
        except Exception:
            picks = []
    else:
        picks = []
    # Render top 12
    cards = []
    for i, p in enumerate(picks[:12]):
        cards.append(render_prop_card_html(p, sport_emojis.get(key, '')))
    by_sport_html[key] = "".join(cards)

# Update in-memory payload for JSON server
_last_props_payload = {
    "mtime": int(current_mtime or 0),
    "by_sport": by_sport_html
}

# Ensure JSON server is running
_start_props_server()

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

# Render tabs with both static content and live updates
for i, (agent, key, emoji) in enumerate(agents):
    with tabs[i]:
        st.markdown(f'<div class="section-title">{agent.__class__.__name__.replace("Agent","")}<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
        
        # Get initial props for immediate display
        props = csv_props.get(key, [])
        if props:
            try:
                picks = agent.make_picks(props_data=props, log_to_ledger=False)
            except Exception:
                picks = []
        else:
            picks = []
        
        # Display initial static content
        if picks:
            display_sport_picks(agent.__class__.__name__.replace('Agent',''), picks, emoji)
        else:
            st.info(f"{emoji} Loading {key.replace('_',' ').title()} props...")
        
        # Add live update component that will replace the content above
        try:
            import streamlit.components.v1 as components
            base_url = f"http://127.0.0.1:{PROPS_ENDPOINT_PORT}/props.json"
            container_id = f"live-props-{key}"
            
            components.html(f"""
                <script>
                  async function updateProps_{key}() {{
                    try {{
                      const res = await fetch('{base_url}', {{ cache: 'no-store' }});
                      const data = await res.json();
                      const html = (data.by_sport && data.by_sport['{key}']) ? data.by_sport['{key}'] : '';
                      
                      // Find the parent streamlit container and update it
                      const containers = document.querySelectorAll('[data-testid="stMarkdownContainer"]');
                      for (let container of containers) {{
                        if (container.innerHTML.includes('props-grid-{key}')) {{
                          if (html) {{
                            container.innerHTML = '<div class="props-grid-{key}">' + html + '</div>';
                          }}
                          break;
                        }}
                      }}
                    }} catch (e) {{
                      console.log('Props update failed:', e);
                    }}
                  }}
                  
                  // Start polling after initial load
                  setTimeout(function() {{
                    updateProps_{key}();
                    setInterval(updateProps_{key}, 15000);
                  }}, 2000);
                </script>
            """, height=0)
        except Exception:
            pass

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 30px;">
    <p>🤖 Powered by AI Sport Agents with ML Prediction Engine</p>
    <p>📊 Real-time prop analysis with confidence scoring and edge calculation</p>
</div>
""", unsafe_allow_html=True)