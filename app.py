import streamlit as st
import pandas as pd
import requests
import os
import time
import threading
import importlib
from datetime import datetime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from lxml import etree
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from page_utils import render_prop_row_html

# Page configuration
st.set_page_config(
    page_title="BetFinder AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Disable auto-refresh
st.config.set_option('server.runOnSave', False)
st.config.set_option('server.fileWatcherType', 'none')

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
                BaseballAgent, HockeyAgent, SoccerAgent, EsportsAgent, CollegeFootballAgent,
                CSGOAgent, LeagueOfLegendsAgent, Dota2Agent, VALORANTAgent, ApexAgent, GolfAgent
            )
            
            # Load data using sport agents
            agents_and_sports = [
                (BasketballAgent(), "basketball"),
                (FootballAgent(), "football"), 
                (CollegeFootballAgent(), "college_football"),
                (TennisAgent(), "tennis"),
                (BaseballAgent(), "baseball"),
                (HockeyAgent(), "hockey"),
                (SoccerAgent(), "soccer"),
                (CSGOAgent(), "csgo"),
                (LeagueOfLegendsAgent(), "league_of_legends"),
                (Dota2Agent(), "dota2"),
                (VALORANTAgent(), "valorant"),
                (ApexAgent(), "apex"),
                (GolfAgent(), "golf")
            ]
            
            for agent, sport in agents_and_sports:
                try:
                    # Do not log to ledger during initial load to avoid heavy disk writes and rerun storms
                    picks = agent.make_picks(log_to_ledger=False)
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

def display_sport_picks(sport_name, picks, sport_emoji, sport_key=None):
    """Display picks for a specific sport with sleek compact layout"""
    count = len(picks) if isinstance(picks, list) else 0
    
    # Compact header with live props count
    st.markdown(
        f"""<div style='display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);border-radius:8px;margin:4px 0;border:1px solid #333;'>
            <div style='display:flex;align-items:center;gap:8px;'>
                <span style='font-size:18px;'>{sport_emoji}</span>
                <span style='font-weight:600;color:#fff;font-size:14px;'>{sport_name}</span>
            </div>
            <div style='display:flex;align-items:center;gap:12px;'>
                <span style='background:#0f9d58;color:#fff;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600;'>{count} LIVE</span>
                <span style='color:#9aa0a6;font-size:10px;'>Updated: {datetime.now().strftime("%H:%M")}</span>
            </div>
        </div>""",
        unsafe_allow_html=True
    )
    
    if picks:
        # Compact filters for specific sports
        original_picks = picks
        if sport_key in ('csgo', 'valorant'):
            map_filter = st.selectbox('📍 Map Filter', ['All Maps', 'Map 1', 'Map 2', 'Map 3'], key=f"{sport_key}_map")
            if map_filter != 'All Maps':
                picks = [p for p in picks if map_filter.lower().replace(' ', '') in str(p.get('stat_type','')).lower()]
        elif sport_key == 'league_of_legends':
            stat_filter = st.selectbox('⚔️ Stat Filter', ['All Stats', 'Kills+Assists', 'KDA', 'Creep Score'], key=f"{sport_key}_stat")
            if stat_filter != 'All Stats':
                stat_map = {'Kills+Assists': ['kills+assists'], 'KDA': ['kda'], 'Creep Score': ['creep', 'cs']}
                needles = stat_map.get(stat_filter, [])
                picks = [p for p in picks if any(n in str(p.get('stat_type','')).lower() for n in needles)]
        
        # Display filtered count
        if len(picks) != len(original_picks):
            st.markdown(f"<div style='color:#9aa0a6;font-size:11px;padding:0 12px;'>Showing {len(picks)} of {len(original_picks)} props</div>", unsafe_allow_html=True)
        
        # Render props in ultra-compact grid
        rows_html = []
        for i, pick in enumerate(picks[:24]):  # Show up to 24 props per tab
            if isinstance(pick, dict):
                rows_html.append(render_compact_prop_row(pick, sport_emoji, i))
        
        if rows_html:
            # Sleek table container with compact headers
            full_html = f"""
            <div style='background:#111;border-radius:8px;padding:6px;margin:8px 0;border:1px solid #333;'>
                <div style='display:grid;grid-template-columns:20px 1.8fr 1.4fr 1fr 80px 60px 60px 50px;gap:8px;padding:6px 8px;border-bottom:1px solid #333;color:#9aa0a6;font-size:9px;font-weight:600;text-transform:uppercase;'>
                    <span></span>
                    <span>PLAYER • BET</span>
                    <span>MATCHUP</span>
                    <span>TYPE</span>
                    <span>CONF</span>
                    <span>EDGE</span>
                    <span>ODDS</span>
                    <span>TIME</span>
                </div>
                <div style='max-height:600px;overflow-y:auto;'>
                    {"".join(rows_html)}
                </div>
            </div>
            """
            st.markdown(full_html, unsafe_allow_html=True)
            
    else:
        st.markdown(f"""<div style='text-align:center;padding:40px;color:#666;'>
            <div style='font-size:24px;margin-bottom:8px;'>{sport_emoji}</div>
            <div>Loading {sport_name.lower()} props...</div>
            <div style='font-size:10px;color:#999;margin-top:4px;'>Fresh data incoming...</div>
        </div>""", unsafe_allow_html=True)


def render_compact_prop_row(pick: dict, sport_emoji: str, index: int) -> str:
    """Render ultra-compact prop row for dense display"""
    player_name = pick.get('player_name', 'Unknown')[:18]  # Truncate long names
    stat_label = pick.get('stat_type', '')
    line_val = pick.get('line', '')
    bet_label = pick.get('bet_label', pick.get('over_under', ''))
    matchup = pick.get('matchup', '')[:15]  # Truncate long matchups
    confidence = pick.get('confidence', 0)
    edge = (pick.get('ml_edge') or pick.get('expected_value', 0) / 100.0) * 100
    odds = pick.get('odds', -110)
    
    # Get classification for styling
    classification = pick.get('prizepicks_classification', '')
    if isinstance(classification, dict):
        classification = classification.get('classification', '')
    
    # Color coding based on confidence/classification
    if confidence >= 85 or '👹' in str(classification):
        row_bg = '#2d1b2d'  # Dark purple for demon picks
        conf_color = '#ff6b9d'
    elif confidence >= 75 or '💰' in str(classification):
        row_bg = '#1b2d1b'  # Dark green for discount picks
        conf_color = '#4ade80'
    elif confidence >= 65:
        row_bg = '#1b1b2d'  # Dark blue for decent picks  
        conf_color = '#60a5fa'
    else:
        row_bg = '#2d1b1b'  # Dark red for goblin picks
        conf_color = '#f87171'
    
    # Format time compactly
    time_display = ''
    try:
        event_time = pick.get('event_time_et', '')
        if event_time:
            # Extract just hour:minute if possible
            if 'PM' in event_time or 'AM' in event_time:
                time_display = event_time.split()[-2] if len(event_time.split()) >= 2 else event_time[:5]
            else:
                time_display = event_time[:5]
    except:
        time_display = ''
    
    # Hover effect and zebra striping
    zebra_bg = '#161616' if index % 2 == 0 else '#1a1a1a'
    
    return f"""
    <div style='display:grid;grid-template-columns:20px 1.8fr 1.4fr 1fr 80px 60px 60px 50px;gap:8px;padding:6px 8px;border-bottom:1px solid #222;background:{zebra_bg};font-size:11px;transition:all 0.2s;' 
         onmouseover="this.style.background='{row_bg}'" 
         onmouseout="this.style.background='{zebra_bg}'">
        <span style='font-size:12px;text-align:center;'>{sport_emoji}</span>
        <div style='color:#fff;'>
            <div style='font-weight:600;font-size:12px;line-height:1.2;'>{player_name}</div>
            <div style='color:#bbb;font-size:10px;line-height:1.1;'>{bet_label} {line_val} {stat_label[:20]}</div>
        </div>
        <span style='color:#ccc;font-size:10px;line-height:1.3;'>{matchup}</span>
        <span style='color:#888;font-size:9px;'>{str(classification)[:8] if classification else '—'}</span>
        <span style='color:{conf_color};font-weight:600;text-align:right;'>{confidence:.0f}%</span>
        <span style='color:#0f9d58;text-align:right;font-size:10px;'>+{edge:.1f}%</span>
        <span style='color:#1a73e8;text-align:right;font-size:10px;'>{int(abs(odds)) if isinstance(odds, (int, float)) else odds}</span>
        <span style='color:#666;text-align:center;font-size:9px;'>{time_display}</span>
    </div>
    """

st.markdown("""
<style>
    /* Modern dark theme with enhanced density */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #ffffff;
    }
    
    /* Ultra-compact tabs for more space */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        flex-wrap: wrap;
        justify-content: flex-start;
        background: #1a1a1a;
        border-radius: 6px;
        padding: 2px;
        border: 1px solid #333;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 9px;
        padding: 3px 8px;
        height: 20px;
        min-width: auto;
        white-space: nowrap;
        border-radius: 4px;
        margin: 0;
        flex-shrink: 0;
        background: transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #333;
        transform: translateY(-1px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    
    /* Compact tab text */
    .stTabs [data-baseweb="tab"] div {
        line-height: 1;
        font-weight: 500;
    }
    
    /* Reduce overall padding and margins */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    
    /* Sleek headers */
    .stMarkdown h1 {
        margin: 0.5rem 0;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stMarkdown h3 {
        color: #fff !important;
        font-size: 1.1rem;
        margin: 8px 0 4px 0;
        font-weight: 600;
    }
    
    /* Compact containers */
    .stContainer {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Sidebar enhancements */
    .stSidebar {
        background: #111;
        border-right: 1px solid #333;
    }
    
    .stSidebar .stMarkdown {
        padding: 0.25rem 0;
    }
    
    /* Button improvements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 11px;
        padding: 4px 12px;
        height: 28px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox enhancements */
    .stSelectbox > div > div {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 6px;
        font-size: 11px;
    }
    
    /* Remove excess spacing */
    .element-container {
        margin-bottom: 0.25rem;
    }
    
    /* Metric styling */
    .stMetric {
        background: #1a1a1a;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #333;
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 14px;
        font-weight: 600;
        color: #4ade80;
    }
    
    .stMetric [data-testid="metric-label"] {
        font-size: 10px;
        color: #9aa0a6;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #666;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #888;
    }
</style>""", unsafe_allow_html=True)

# Sleek header
st.markdown("""
<div style="text-align: center; margin: 8px 0 16px 0; padding: 12px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 12px; border: 1px solid #333;">
    <h1 style="margin: 0; font-size: 24px; font-weight: 700;">🎯 BetFinder AI</h1>
    <p style="margin: 4px 0 0 0; font-size: 12px; color: #9aa0a6; font-weight: 500;">Real-time Sports Betting Intelligence • ML-Powered Predictions</p>
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
    try:
        ct = datetime.fromtimestamp(last_ts, ZoneInfo('America/Chicago'))
        st.sidebar.write(f"Last update: {ct.strftime('%Y-%m-%d %H:%M:%S')} CT")
    except Exception:
        st.sidebar.write(f"Last update: {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.sidebar.write("Last update: pending…")
st.sidebar.write(f"Interval: {st.session_state.get('auto_scrape_interval_sec', 300)}s")

# View options
st.sidebar.header("View Options")
st.sidebar.checkbox("Show agent reasoning", value=False, key="show_reasoning")

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
# Allow disabling auto-scraper in dev with AUTO_SCRAPER=off
if os.environ.get('AUTO_SCRAPER', 'on').lower() != 'off':
    start_auto_scraper(effective_csv_path, st.session_state.auto_scrape_interval_sec)
else:
    st.sidebar.warning("Auto-scraper disabled (AUTO_SCRAPER=off)")

# Auto-refresh disabled to prevent full page reloads - using JSON polling instead
# st_autorefresh = None
# if st_autorefresh is not None:
#     st_autorefresh(interval=20000, key="auto_refresh_interval")


# =============== Live Props JSON server (polling by client) ===============

# Bind host is where the HTTP server listens; client host is what we use for in-app probes
PROPS_BIND_HOST = os.environ.get('PROPS_BIND_HOST', '0.0.0.0')
PROPS_CLIENT_HOST = os.environ.get('PROPS_CLIENT_HOST', '127.0.0.1')
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
            httpd = ThreadingHTTPServer((PROPS_BIND_HOST, PROPS_ENDPOINT_PORT), PropsRequestHandler)
            httpd.serve_forever()
        except Exception:
            pass
    th = threading.Thread(target=_serve, daemon=True)
    th.start()
    _server_started = True

# Tabs UI (replaces multipage navigation)
tab_names = [
    "🏠 Home",
    "🏀 Basketball", 
    "🏈 NFL",
    "🎓 College FB",
    "🎾 Tennis",
    "⚾ Baseball",
    "🏒 Hockey",
    "⚽ Soccer",
    "🔫 CS:GO",
    "🧙 LoL",
    "🐉 Dota 2",
    "🎯 Valorant",
    "� Apex",
    "⛳ Golf",
]
tabs = st.tabs(tab_names)

# Helper to filter picks for today and select agent's favorite
def _is_today_pick(pick):
    """Check if a pick is for today's games"""
    try:
        # Check both 'start_time' and 'event_start_time' fields
        start_time = pick.get('start_time') or pick.get('event_start_time', '')
        if not start_time:
            return True  # Include picks without explicit time (likely today)
        
        # Parse various time formats that might be in start_time
        from datetime import datetime
        today = datetime.now().date()
        
        # Handle ISO format first (most common from agents)
        try:
            if 'T' in str(start_time):
                pick_date = datetime.fromisoformat(str(start_time).replace('Z', '+00:00')).date()
                return pick_date == today
        except:
            pass
        
        # Try common formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y']:
            try:
                pick_date = datetime.strptime(str(start_time).split()[0], fmt).date()
                return pick_date == today
            except:
                continue
        
        # If no format matches, include by default (likely today)
        return True
    except:
        return True

def _select_favorite_today(picks: list):
    """Select agent's favorite pick from today's games only"""
    try:
        if not picks:
            return None
        
        # Filter for today's picks first
        today_picks = [p for p in picks if isinstance(p, dict) and _is_today_pick(p)]
        if not today_picks:
            return None
            
        # Sort by confidence desc, then expected_value desc if present
        def _score(p):
            conf = float(p.get('confidence', 0) or 0)
            ev = float(p.get('expected_value', 0) or 0)
            return (conf, ev)
        return sorted(today_picks, key=_score, reverse=True)[0]
    except Exception:
        return None

# Home tab: Agent contributions board (favorite prop per agent for TODAY)
with tabs[0]:
    st.markdown("### 🏠 Today's Agent Contributions")
    st.caption("Each agent's highest-confidence pick for today's games")
    
    # Map of (agent_name, emoji, cache_key, sport_key for renderer)
    boards = [
        ("Basketball Agent", "🏀", "basketball_props", "basketball"),
        ("NFL Agent", "🏈", "football_props", "football"),
        ("College Football Agent", "🎓", "college_football_props", "college_football"),
        ("Tennis Agent", "🎾", "tennis_props", "tennis"),
        ("Baseball Agent", "⚾", "baseball_props", "baseball"),
        ("Hockey Agent", "🏒", "hockey_props", "hockey"),
        ("Soccer Agent", "⚽", "soccer_props", "soccer"),
        ("CS:GO Agent", "🔫", "csgo_props", "csgo"),
        ("League of Legends Agent", "🧙", "league_of_legends_props", "league_of_legends"),
        ("Dota 2 Agent", "🐉", "dota2_props", "dota2"),
    ("Valorant Agent", "🎯", "valorant_props", "valorant"),
    ("Apex Agent", "🔺", "apex_props", "apex"),
        ("Golf Agent", "⛳", "golf_props", "golf"),
    ]
    cols = st.columns(3)  # Use 3 columns to accommodate more agents
    shown = 0
    for idx, (agent_name, emoji, cache_key, sport_key) in enumerate(boards):
        data = (get_cached_data(cache_key) or {}).get('data', [])
        fav = _select_favorite_today(data)
        if not fav:
            continue
        # Render favorite as a single compact row with agent attribution
        with cols[idx % 3]:  # Use modulo 3 for 3-column layout
            st.markdown(
                f"<div style='margin:6px 0;font-weight:600;color:#667eea;font-size:13px;border-bottom:1px solid #333;padding-bottom:2px;'>{emoji} {agent_name}</div>",
                unsafe_allow_html=True,
            )
            try:
                st.markdown(render_prop_row_html(fav, emoji), unsafe_allow_html=True)
                shown += 1
            except Exception:
                # Fallback simple render with agent attribution
                player = fav.get('player_name') or fav.get('description') or 'Unknown'
                stat = fav.get('stat_type', '').title()
                line = fav.get('line', 'N/A')
                conf = fav.get('confidence', 0)
                st.markdown(f"{emoji} {player} — {stat} {line} • {conf}%", unsafe_allow_html=True)
    if shown == 0:
        from datetime import datetime
        today_str = datetime.now().strftime("%B %d, %Y")
        st.warning(f"⚠️ No props available for today's matchups ({today_str}). All agents are configured to show only today's games.")
    st.markdown("---")
    from datetime import datetime
    today_str = datetime.now().strftime("%B %d, %Y")
    st.caption(f"Showing today's picks only ({today_str}) • Favorites auto-select the highest-confidence pick per agent for today's games • Toggle reasoning in sidebar for analysis")

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
        empty = {k: [] for k in [
            'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
            'csgo', 'league_of_legends', 'dota2', 'valorant', 'apex', 'golf']}
        set_cached_data(cache_key, empty)
        return empty
    
    props = []
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get('name') or cols.get('player')
    line_col = cols.get('points') or cols.get('line')
    prop_col = cols.get('prop')
    sport_col = cols.get('sport')  # optional in real data
    league_col = cols.get('league')  # many PrizePicks exports use League
    game_col = cols.get('game')  # sometimes 'Game' carries the title

    # League to sport normalization
    def normalize_league_to_sport(value: str) -> str:
        if not value:
            return ''
        s = str(value).strip().lower()
        aliases = {
            # Core traditional sports
            'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
            'nbaszn': 'basketball', 'nbap': 'basketball', 'bbl': 'basketball',
            'nfl': 'football', 'nfl1h': 'football', 'nfl1q': 'football', 'nfl2h': 'football',
            # College football aliases
            'cfb': 'college_football', 'ncaa football': 'college_football', 'ncaaf': 'college_football',
            'ncaa': 'college_football', 'college football': 'college_football', 'college-football': 'college_football',
            'ncaa fb': 'college_football', 'ncaa-fb': 'college_football',
            'mlb': 'baseball', 'mlblive': 'baseball', 'kbo': 'baseball',
            'nhl': 'hockey', 'nhl1p': 'hockey',
            'epl': 'soccer', 'soccer': 'soccer',
            'tennis': 'tennis',
            # Esports titles
            'league of legends': 'league_of_legends', 'lol': 'league_of_legends',
            'valorant': 'valorant', 'valo': 'valorant',
            'dota 2': 'dota2', 'dota2': 'dota2',
            'apex': 'apex', 'apex legends': 'apex',
            'golf': 'golf', 'pga': 'golf', 'masters': 'golf', 'tour': 'golf',
            'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo',
            'r6': 'valorant'  # Rainbow Six Siege - group with tactical FPS
        }
        return aliases.get(s, s)
    
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
        # Prefer league/game columns when present
        league_val = str(row.get(league_col, '')).strip() if league_col else ''
        game_val = str(row.get(game_col, '')).strip() if game_col else ''
        sport_raw = str(row.get(sport_col, '')).strip() if sport_col else ''
        norm = normalize_league_to_sport(league_val or game_val or sport_raw)
        sport_val = norm.lower()

        prop = {
            'player_name': player_name,
            'team': '',
            'pick': stat_type,
            'stat_type': stat_type.lower(),
            'line': line,
            'odds': -110,
            'confidence': 50.0,
            'expected_value': 0.0,
            'avg_l10': row.get('Avg_L10') or row.get('L10') or None,
            'l5_average': row.get('L5') or None,
            'h2h': row.get('H2H') or row.get('h2h') or None,
            'start_time': '',
            'sport': sport_val or '',
            'league': (league_val or game_val or sport_raw).strip(),
            'over_under': None,
            # Additional fields for proper display
            'matchup': row.get('Matchup', ''),
            'bet_label': 'Over' if str(row.get('Allow_Under', '')).lower() != 'false' else '',
            'ml_edge': 0.05,  # Default 5% edge for CSV props
            'event_time_et': row.get('Game_Time', ''),
            'event_date': row.get('Game_Date', ''),
            'last_updated': row.get('Last_Updated', ''),
            'prizepicks_classification': 'DECK 📊',  # Default classification for CSV props
        }
        props.append(prop)
    
    grouped = {k: [] for k in [
        'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'apex', 'golf']}
    
    def map_to_sport(p: dict) -> str:
        player = str(p.get('player_name', '')).lower()
        stat = str(p.get('stat_type', '')).lower()

        # 1) If explicit sport provided, honor it via aliases
        if p.get('sport'):
            s = str(p['sport']).lower()
            aliases = {
                'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
                'nbaszn': 'basketball', 'nbap': 'basketball', 'bbl': 'basketball',
                'nfl': 'football', 'nfl1h': 'football', 'nfl1q': 'football', 'nfl2h': 'football',
                # College football aliases
                'cfb': 'college_football', 'ncaa football': 'college_football', 'ncaaf': 'college_football',
                'ncaa': 'college_football', 'college football': 'college_football', 'college-football': 'college_football',
                'ncaa fb': 'college_football', 'ncaa-fb': 'college_football',
                'mlb': 'baseball', 'mlblive': 'baseball', 'kbo': 'baseball',
                'nhl': 'hockey', 'nhl1p': 'hockey',
                'epl': 'soccer', 'soccer': 'soccer',
                'tennis': 'tennis',
                # Esports aliases
                'lol': 'league_of_legends', 'league of legends': 'league_of_legends', 'league_of_legends': 'league_of_legends',
                'dota2': 'dota2', 'dota 2': 'dota2',
                'valorant': 'valorant', 'valo': 'valorant', 'r6': 'valorant',
                'apex': 'apex', 'apex legends': 'apex',
                'golf': 'golf', 'pga': 'golf', 'masters': 'golf', 'tour': 'golf',
                'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo'
            }
            mapped = aliases.get(s)
            if mapped:
                return mapped
            # Unknown/ambiguous explicit sport (e.g., 'esports'): fall through to pattern detection

        # 2) NBA/Basketball check
        nba_players = [
            'stephen curry', 'kevin durant', 'lebron james', 'giannis antetokounmpo', 'luka doncic',
            'jayson tatum', 'nikola jokic', 'joel embiid', 'anthony edwards', 'victor wembanyama',
            'shai gilgeous-alexander', 'anthony davis', 'devin booker', 'ja morant', 'paolo banchero',
            'scottie barnes', 'franz wagner', 'cade cunningham', 'evan mobley', 'jalen green',
            'alperen sengun', 'tyrese haliburton', 'donovan mitchell', 'darius garland', 'jarrett allen'
        ]
        basketball_stats = [
            'points', 'rebounds', 'assists', 'blocks', 'steals', '3pt made', 'pts+rebs+asts',
            'triple-doubles', 'double-double', 'pts+rebs', 'pts+asts', 'rebs+asts',
            'points per game avg', 'rebounds per game avg', 'assists per game avg', '3pt made per game avg',
            '40+ points games', '50+ point games'
        ]
        if any(n in player for n in nba_players) or any(k in stat for k in basketball_stats):
            return 'basketball'

    # 3) College football vs NFL
        college_players = [
            'quinn ewers', 'carson beck', 'cam ward', 'dillon gabriel', 'jalen milroe',
            'tyler van dyke', 'travis hunter', 'tetairoa mcmillan', 'ryan williams',
            'luther burden iii', 'tre harris', 'ashton jeanty', 'will howard',
            'diego pavia', 'kurtis rourke', 'nico iamaleava', 'garrett nussmeier'
        ]
        football_stats = [
            'pass yards', 'passing yards', 'rush yards', 'rushing yards', 'receiving yards', 'receptions',
            'rush+rec tds', 'rush+rec yds', 'fantasy score', 'longest reception', 'longest rush',
            'rush attempts', 'rec targets', 'touchdown', 'sacks', 'tackles', 'interceptions',
            'yards on first reception', 'yards on first rush attempt', 'halves with', 'quarters with'
        ]
        if any(n in player for n in college_players) and any(k in stat for k in football_stats):
            return 'college_football'
        if any(k in stat for k in football_stats):
            return 'football'

        # 4) Esports detection BEFORE hockey/soccer to avoid misclassification
        # Prefer player-name detection across titles, then fall back to stat keywords
        csgo_players = [
            'zywoo', 'sh1ro', 'donk', 'ax1le', 'jl', 'ropz', 'frozen', 'siuhy', 'jimpphat',
            'aleksib', 'niko', 'm0nesy', 'jks', 'hunter', 'nexa', 'electronic', 'nafany'
        ]
        dota_players = [
            'yatoro', 'torontotokyo', 'collapse', 'mira', 'miposhka', 'pure', 'malr1ne',
            'larl', 'wisper', 'akbar', 'crystallis', 'bzm', 'ammar_the_f', 'ceb', 'jerax'
        ]
        lol_players = [
            'faker', 'zeus', 'oner', 'gumayusi', 'keria', 'chovy', 'canyon', 'showmaker',
            'ruler', 'lehends', 'caps', 'razork', 'humanoid', 'noah', 'mikyx', 'jankos'
        ]
        valorant_players = [
            'aspas', 'less', 'saadhak', 'cauanzin', 'tuyz', 'demon1', 'jawgemo', 'ethan',
            'boostio', 'c0m', 'tenz', 'zekken', 'sacy', 'pancada', 'johnqt', 'f0rsaken'
        ]
        apex_players = [
            'imperialhal', 'reps', 'verhulst', 'genburten', 'noyou', 'zera', 'skittlecakes', 'sweet',
            'nokokopuffs', 'tsm', 'ascend', 'optic', 'furia', 'phony', 'staynaughtyy'
        ]
        golf_players = [
            'scottie scheffler', 'xander schauffele', 'ludvig aberg', 'viktor hovland',
            'collin morikawa', 'wyndham clark', 'patrick cantlay', 'sahith theegala',
            'rory mcilroy', 'jon rahm', 'bryson dechambeau', 'justin thomas', 'max homa'
        ]

        if any(n in player for n in dota_players):
            return 'dota2'
        if any(n in player for n in lol_players):
            return 'league_of_legends'
        if any(n in player for n in valorant_players):
            return 'valorant'
        if any(n in player for n in apex_players):
            return 'apex'
        if any(n in player for n in golf_players):
            return 'golf'
        if any(n in player for n in csgo_players):
            return 'csgo'

        # Stat-based fallbacks
        lol_keywords = ['kda', 'k/d/a', 'kills+assists', 'kills + assists', 'creep score', 'cs ', 'wards', 'dragon', 'baron']
        if any(k in stat for k in lol_keywords):
            return 'league_of_legends'
        dota_keywords = ['gpm', 'xpm', 'last hits', 'denies', 'roshan', 'towers destroyed']
        if any(k in stat for k in dota_keywords):
            return 'dota2'
        # Generic map-based kill props common to CS/VAL/Dota tournament formats
        if ('map' in stat or 'maps' in stat) and 'kills' in stat:
            if any(n in player for n in csgo_players):
                return 'csgo'
            if any(n in player for n in valorant_players):
                return 'valorant'
            if any(n in player for n in dota_players):
                return 'dota2'
            # Ambiguous: leave unclassified to avoid wrong-tab routing
            return ''
        valorant_keywords = ['acs', 'first bloods', 'first kills', 'spike', 'plant', 'defuse']
        if any(k in stat for k in valorant_keywords):
            return 'valorant'
        apex_keywords = ['knockdowns', 'revives', 'placement', 'legends', 'ring damage', 'finishers']
        if any(k in stat for k in apex_keywords):
            return 'apex'
        golf_keywords = ['strokes', 'birdies', 'eagles', 'pars', 'bogeys', 'fairways', 'greens in regulation', 'putts']
        if any(k in stat for k in golf_keywords):
            return 'golf'
        # CS:GO only with CS-specific terms (avoid generic 'map(s)')
        csgo_keywords = ['headshot', 'headshots', 'awp', 'adr', 'clutch', 'clutches', 'bomb', 'round kills']
        if any(k in stat for k in csgo_keywords):
            return 'csgo'

        # 5) NHL/Hockey before Soccer to avoid overlap on 'goals/assists/saves'
        nhl_players = [
            'connor mcdavid', 'leon draisaitl', 'nathan mackinnon', 'david pastrnak', 'auston matthews',
            'mitch marner', 'nikita kucherov', 'cale makar', 'quinn hughes', 'jack hughes',
            'kirill kaprizov', 'erik karlsson', 'artemi panarin', 'igor shesterkin', 'sidney crosby',
            'alexander ovechkin', 'brad marchand', 'mikko rantanen', 'victor hedman', 'william nylander'
        ]
        if any(n in player for n in nhl_players):
            return 'hockey'
        hockey_specific = [
            'penalty minutes', 'power play', 'faceoff', 'time on ice', 'plus/minus', 'blocked shots',
            'goalie saves', 'save percentage', 'goals against', 'shutouts'
        ]
        if any(k in stat for k in hockey_specific):
            return 'hockey'
        if 'saves' in stat and any(k in stat for k in ['goalie', 'save percentage', 'goals against']):
            return 'hockey'

    # 6) Soccer after hockey
        soccer_stats = [
            'goals', 'assists', 'shots on goal', 'shots on target', 'shots', 'goal + assist', 'fouls',
            'cards', 'clean sheets', 'saves', 'goalie saves'
        ]
        if any(k in stat for k in soccer_stats):
            return 'soccer'

        # 7) Golf / Tennis / Baseball
        # Do NOT misroute golf to soccer; if we detect golf-like stats and we don't have a golf tab, skip classification
        if any(k in stat for k in ['strokes', 'birdies', 'eagles', 'pars', 'bogeys']):
            return ''  # unclassified (prevents polluting other tabs)
        if any(k in stat for k in ['aces', 'double faults', 'games won', 'sets won']):
            return 'tennis'
        if any(k in stat for k in ['hits', 'home runs', 'rbis', 'strikeouts', 'total bases', 'stolen bases']):
            return 'baseball'

        # 8) Default: leave unclassified so it won't appear in any tab
        return ''
    
    # After defining map_to_sport, classify and group props
    for p in props:
        sport_key = map_to_sport(p)
        # Only include when confidently classified
        if sport_key and sport_key in grouped:
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
        try:
            ct = datetime.fromtimestamp(current_mtime, ZoneInfo('America/Chicago'))
            st.toast(f"PrizePicks CSV updated at {ct.strftime('%H:%M:%S')} CT", icon="✅")
        except Exception:
            st.toast(f"PrizePicks CSV updated at {datetime.fromtimestamp(current_mtime).strftime('%H:%M:%S')}", icon="✅")

csv_props = load_prizepicks_csv_cached(effective_csv_path)
# Safety: ensure we have a dict mapping; if not, reset to empty groups
if not isinstance(csv_props, dict):
    csv_props = {k: [] for k in [
        'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'apex', 'golf']}

# Debug: sidebar summary of grouped counts to verify routing
try:
    st.sidebar.subheader("Props by sport (debug)")
    for k in ['basketball', 'football', 'college_football', 'hockey', 'soccer', 'tennis', 'baseball',
              'csgo', 'league_of_legends', 'dota2', 'valorant', 'apex', 'golf']:
        items = csv_props.get(k, []) or []
        if items:
            sample = str(items[0].get('stat_type', ''))[:28]
            st.sidebar.write(f"{k.replace('_',' ').title()}: {len(items)} (e.g., {sample}…)")
        else:
            st.sidebar.write(f"{k.replace('_',' ').title()}: 0")
except Exception:
    pass

# ===================== Health Check =====================
def _status_chip(text: str, color: str) -> str:
    return f"<span style='background:{color};color:#111;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:6px;'>{text}</span>"

def _fmt_bytes(n: int) -> str:
    try:
        for unit in ['B','KB','MB','GB']:
            if n < 1024:
                return f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"
    except Exception:
        return "-"

try:
    # File status
    file_exists = os.path.exists(effective_csv_path)
    file_size = os.path.getsize(effective_csv_path) if file_exists else 0
    mtime = os.path.getmtime(effective_csv_path) if file_exists else 0
    age_sec = (time.time() - mtime) if mtime else 1e9
    max_age = max(30, 2 * st.session_state.get('auto_scrape_interval_sec', 300))
    fresh = age_sec < max_age

    # Scraper & server status
    scraper_ok = bool(st.session_state.get('auto_scraper_started', False))
    server_ok = bool('_server_started' in globals() and _server_started)

    # Endpoint probe
    endpoint_ok = False
    try:
        resp = requests.get(f"http://{PROPS_CLIENT_HOST}:{PROPS_ENDPOINT_PORT}/props.json", timeout=0.5)
        endpoint_ok = (resp.status_code == 200)
    except Exception:
        endpoint_ok = False

    # Props distribution
    total_props = sum(len(v) for v in csv_props.values()) if isinstance(csv_props, dict) else 0
    any_props = total_props > 0

    overall_ok = file_exists and fresh and server_ok and endpoint_ok and any_props
    banner_color = '#34a853' if overall_ok else ('#fbbc05' if file_exists else '#ea4335')
    banner_text = 'Healthy' if overall_ok else ('Degraded' if file_exists else 'Unavailable')

    st.markdown(
        f"""
        <div style='margin:8px 0;padding:8px 12px;border-radius:8px;background:{banner_color}22;border:1px solid {banner_color};'>
            <strong>System Health:</strong> {_status_chip(banner_text, banner_color)}
            {_status_chip('CSV OK' if (file_exists and fresh) else 'CSV STALE' if file_exists else 'CSV MISSING', '#34a853' if (file_exists and fresh) else ('#fbbc05' if file_exists else '#ea4335'))}
            {_status_chip('Scraper ON' if scraper_ok else 'Scraper OFF', '#34a853' if scraper_ok else '#fbbc05')}
            {_status_chip('Server UP' if (server_ok and endpoint_ok) else 'Server DOWN', '#34a853' if (server_ok and endpoint_ok) else '#ea4335')}
            {_status_chip(f"Props {total_props}", '#34a853' if any_props else '#fbbc05')}
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Health check details"):
        st.write({
            'csv_path': effective_csv_path,
            'csv_exists': file_exists,
            'csv_size': _fmt_bytes(file_size),
            'csv_age_seconds': int(age_sec if age_sec < 1e8 else -1),
            'fresh_threshold_seconds': int(max_age),
            'scraper_started': scraper_ok,
            'json_server_started': server_ok,
            'endpoint_ok': endpoint_ok,
            'props_total': total_props,
        })
        if file_exists:
            try:
                ct = datetime.fromtimestamp(mtime, ZoneInfo('America/Chicago'))
                st.write(f"CSV last update (CT): {ct.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                pass
        # Per-sport quick counts
        if isinstance(csv_props, dict):
            st.write({k: len(v) for k, v in csv_props.items()})
except Exception:
    pass

_last_props_payload = {"mtime": int(current_mtime or 0), "by_sport": {}}

# Ensure JSON server is running
_start_props_server()

# Sport tabs
with tabs[1]:
    picks = (get_cached_data("basketball_props") or {}).get('data', [])
    csv_picks = csv_props.get('basketball', [])
    combined_picks = picks + csv_picks
    display_sport_picks("Basketball", combined_picks, "🏀", sport_key="basketball")

with tabs[2]:
    picks = (get_cached_data("football_props") or {}).get('data', [])
    csv_picks = csv_props.get('football', [])
    combined_picks = picks + csv_picks
    display_sport_picks("NFL", combined_picks, "🏈", sport_key="football")

with tabs[3]:
    picks = (get_cached_data("college_football_props") or {}).get('data', [])
    csv_picks = csv_props.get('college_football', [])
    combined_picks = picks + csv_picks
    display_sport_picks("College Football", combined_picks, "🎓", sport_key="college_football")

with tabs[4]:
    picks = (get_cached_data("tennis_props") or {}).get('data', [])
    csv_picks = csv_props.get('tennis', [])
    combined_picks = picks + csv_picks
    display_sport_picks("Tennis", combined_picks, "🎾", sport_key="tennis")

with tabs[5]:
    picks = (get_cached_data("baseball_props") or {}).get('data', [])
    csv_picks = csv_props.get('baseball', [])
    combined_picks = picks + csv_picks
    display_sport_picks("Baseball", combined_picks, "⚾", sport_key="baseball")

with tabs[6]:
    picks = (get_cached_data("hockey_props") or {}).get('data', [])
    csv_picks = csv_props.get('hockey', [])
    combined_picks = picks + csv_picks
    display_sport_picks("Hockey", combined_picks, "🏒", sport_key="hockey")

with tabs[7]:
    picks = (get_cached_data("soccer_props") or {}).get('data', [])
    csv_picks = csv_props.get('soccer', [])
    combined_picks = picks + csv_picks
    display_sport_picks("Soccer", combined_picks, "⚽", sport_key="soccer")

with tabs[8]:
    picks = (get_cached_data("csgo_props") or {}).get('data', [])
    csv_picks = csv_props.get('csgo', [])
    combined_picks = picks + csv_picks
    display_sport_picks("CS:GO", combined_picks, "🔫", sport_key="csgo")

with tabs[9]:
    picks = (get_cached_data("league_of_legends_props") or {}).get('data', [])
    csv_picks = csv_props.get('league_of_legends', [])
    combined_picks = picks + csv_picks
    display_sport_picks("League of Legends", combined_picks, "🧙", sport_key="league_of_legends")

with tabs[10]:
    picks = (get_cached_data("dota2_props") or {}).get('data', [])
    csv_picks = csv_props.get('dota2', [])
    combined_picks = picks + csv_picks
    display_sport_picks("Dota 2", combined_picks, "🐉", sport_key="dota2")

with tabs[11]:
    picks = (get_cached_data("valorant_props") or {}).get('data', [])
    csv_picks = csv_props.get('valorant', [])
    combined_picks = picks + csv_picks
    display_sport_picks("Valorant", combined_picks, "🎯", sport_key="valorant")

with tabs[12]:
    picks = (get_cached_data("apex_props") or {}).get('data', [])
    display_sport_picks("Apex Legends", picks, "�", sport_key="apex")

with tabs[13]:
    picks = (get_cached_data("golf_props") or {}).get('data', [])
    display_sport_picks("Golf", picks, "⛳", sport_key="golf")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 30px;">
    <p>🤖 Powered by AI Sport Agents with ML Prediction Engine</p>
    <p>📊 Real-time prop analysis with confidence scoring and edge calculation</p>
</div>
""", unsafe_allow_html=True)