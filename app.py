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
                CSGOAgent, LeagueOfLegendsAgent, Dota2Agent, VALORANTAgent, OverwatchAgent, GolfAgent
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
                (OverwatchAgent(), "overwatch"),
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
    """Display picks for a specific sport with row-based layout"""
    count = len(picks) if isinstance(picks, list) else 0
    st.markdown(
        f'<div class="section-title">{sport_name}'
        f'<span class="time">Live & Upcoming</span>'
        f'<span class="time" style="margin-left:8px;opacity:0.8;">{count} shown</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    if picks:
        # Per-tab compact filters (esports examples)
        original_picks = picks
        if sport_key in ('csgo', 'valorant'):
            # Map filters
            map_opts = ['Any', 'Map 1', 'Map 2', 'Map 3']
            choice = st.selectbox('Filter by map', map_opts, index=0, key=f"{sport_key}_map_filter")
            if choice != 'Any':
                needle = choice.lower()
                picks = [p for p in picks if needle in str(p.get('stat_type','')).lower()]
        elif sport_key == 'league_of_legends':
            # Role/stat filters
            stat_opts = ['Any', 'Kills+Assists', 'KDA', 'Creep Score']
            choice = st.selectbox('Filter by stat', stat_opts, index=0, key=f"{sport_key}_stat_filter")
            if choice != 'Any':
                needles = {
                    'Kills+Assists': ['kills+assists', 'kills + assists'],
                    'KDA': ['kda', 'k/d/a'],
                    'Creep Score': ['creep score', ' cs ']
                }[choice]
                picks = [p for p in picks if any(n in str(p.get('stat_type','')).lower() for n in needles)]
        elif sport_key == 'golf':
            stat_opts = ['Any', 'Strokes', 'Birdies', 'Eagles', 'Pars']
            choice = st.selectbox('Filter by stat', stat_opts, index=0, key=f"{sport_key}_stat_filter")
            if choice != 'Any':
                needles = choice.lower()
                picks = [p for p in picks if needles in str(p.get('stat_type','')).lower()]
        # Update count after filters
        count = len(picks)
        st.markdown(
            f"<div style='margin-top:-6px;color:#9aa0a6;font-size:11px;'>Showing {count} of {len(original_picks)} picks</div>",
            unsafe_allow_html=True
        )
        # Compact top-factor badges (aggregate strongest signals across shown picks)
        factor_scores = {}
        for p in picks[:12]:
            af = p.get('analysis_factors') or {}
            for k, v in af.items():
                # skip meta keys
                if k == 'overall_score':
                    continue
                score = v.get('score') if isinstance(v, dict) else None
                if isinstance(score, (int, float)):
                    factor_scores[k] = factor_scores.get(k, 0.0) + float(score)
        top = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        if top:
            badges = []
            rename = {
                'player_form': 'Form', 'matchup_analysis': 'Matchup', 'historical_performance': 'History',
                'injury_impact': 'Injuries', 'situational_factors': 'Situational', 'line_value': 'Line Value',
                'weather_conditions': 'Weather', 'team_dynamics': 'Team'
            }
            for name, total in top:
                label = rename.get(name, name.replace('_',' ').title())
                badges.append(f"<span class='pill' style='background:#263238;color:#bfefff;border:1px solid #335;padding:2px 6px;font-size:10px;margin-right:6px;'>{label}</span>")
            st.markdown(f"<div style='margin-bottom:4px;'>{''.join(badges)}</div>", unsafe_allow_html=True)
        # Display picks as rows
        rows_html = []
        for pick in picks[:12]:  # Show top 12 picks post-filter
            if isinstance(pick, dict):
                rows_html.append(render_prop_row_html(pick, sport_emoji))
        
        # Display all rows in a container
        if rows_html:
            # Check if any picks have historical data to determine if we should show L5/L10/H2H columns
            has_historical = any(pick.get('l5_average') or pick.get('avg_l10') or pick.get('h2h') for pick in picks[:12] if isinstance(pick, dict))
            
            historical_headers = """
                    <span style='flex:0.6;'>L5</span>
                    <span style='flex:0.6;'>L10</span>
                    <span style='flex:0.6;'>H2H</span>""" if has_historical else ""
            
            full_html = f"""
            <div style='background:#1a1a1a;border-radius:6px;padding:8px;margin:8px 0;'>
                <div style='display:flex;align-items:center;padding:4px 0;border-bottom:2px solid #333;font-weight:600;color:#aaa;font-size:10px;'>
                    <span style='width:20px;'></span>
                    <span style='flex:1.5;'>PLAYER</span>
                    <span style='flex:1.2;'>BET</span>
                    <span style='flex:1;'>MATCHUP</span>{historical_headers}
                    <span style='min-width:60px;text-align:right;'>TYPE</span>
                    <span style='min-width:50px;text-align:right;'>CONF</span>
                    <span style='min-width:50px;text-align:right;'>EV</span>
                    <span style='min-width:40px;text-align:right;'>ODDS</span>
                </div>
                {"".join(rows_html)}
            </div>
            """
            st.markdown(full_html, unsafe_allow_html=True)
            
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

st.markdown("""
<style>
    /* Make overall UI more compact */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Reduce tab spacing and make tabs more compact */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        flex-wrap: wrap;
        justify-content: flex-start;
        max-width: 100%;
        overflow-x: auto;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 10px;
        padding: 4px 6px;
        height: 22px;
        min-width: auto;
        white-space: nowrap;
        border-radius: 4px;
        margin: 1px;
        flex-shrink: 0;
    }
    
    /* Make tab text more compact */
    .stTabs [data-baseweb="tab"] div {
        line-height: 1.1;
        font-weight: 500;
    }
    
    /* Ensure tabs container doesn't overflow */
    .stTabs {
        width: 100%;
        overflow: visible;
    }
    
    /* Compact section titles */
    .section-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .time {
        font-size: 11px;
        opacity: 0.8;
    }
    
    /* Compact pills */
    .pill { 
        display: inline-block; 
        padding: 2px 6px; 
        border-radius: 999px; 
        font-size: 9px; 
        font-weight: 700; 
        color: #fff; 
        border: 1px solid transparent; 
    }
    .badge-high { background: rgba(32,201,151,0.18); border-color: #20c997; }
    .badge-medium { background: rgba(240,173,78,0.18); border-color: #f0ad4e; }
    .badge-low { background: rgba(220,53,69,0.18); border-color: #dc3545; }
    .pill-edge { background: rgba(0,255,136,0.16); border-color: #00ff88; color: #b7ffd8; }
    .pill-odds { background: rgba(0,123,255,0.16); border-color: #0d6efd; color: #b8d7ff; }
    
    /* Compact metrics */
    .stMetric {
        padding: 4px 0;
    }
    
    .stMetric > div {
        font-size: 12px;
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 16px;
    }
    
    /* Reduce header spacing */
    .stMarkdown h1 {
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }
    
    .stMarkdown h3 {
        color: white !important;
        font-size: 1.2rem;
        margin: 8px 0;
    }
    
    /* Compact containers */
    .stContainer {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Reduce sidebar spacing */
    .stSidebar .stMarkdown {
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h1>🎯 BetFinder AI</h1>
    <p style="font-size: 16px; color: #666;">Advanced Sports Betting Analysis with ML Predictions</p>
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
    "🛡️ Overwatch",
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
        ("Overwatch Agent", "🛡️", "overwatch_props", "overwatch"),
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
            'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'golf']}
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
            'nfl': 'football',
            # College football aliases
            'cfb': 'college_football', 'ncaa football': 'college_football', 'ncaaf': 'college_football',
            'ncaa': 'college_football', 'college football': 'college_football', 'college-football': 'college_football',
            'ncaa fb': 'college_football', 'ncaa-fb': 'college_football',
            'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer', 'soccer': 'soccer',
            # Esports titles
            'league of legends': 'league_of_legends', 'lol': 'league_of_legends',
            'valorant': 'valorant', 'valo': 'valorant',
            'dota 2': 'dota2', 'dota2': 'dota2',
            'overwatch': 'overwatch', 'overwatch 2': 'overwatch', 'ow': 'overwatch',
            'golf': 'golf', 'pga': 'golf', 'masters': 'golf', 'tour': 'golf',
            'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo'
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
            'over_under': None
        }
        props.append(prop)
    
    grouped = {k: [] for k in [
        'basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer', 'college_football',
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'golf']}
    
    def map_to_sport(p: dict) -> str:
        player = str(p.get('player_name', '')).lower()
        stat = str(p.get('stat_type', '')).lower()

        # 1) If explicit sport provided, honor it via aliases
        if p.get('sport'):
            s = str(p['sport']).lower()
            aliases = {
                'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
                'nfl': 'football',
                # College football aliases
                'cfb': 'college_football', 'ncaa football': 'college_football', 'ncaaf': 'college_football',
                'ncaa': 'college_football', 'college football': 'college_football', 'college-football': 'college_football',
                'ncaa fb': 'college_football', 'ncaa-fb': 'college_football',
                'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer',
                # Esports aliases
                'lol': 'league_of_legends', 'league of legends': 'league_of_legends', 'league_of_legends': 'league_of_legends',
                'dota2': 'dota2', 'dota 2': 'dota2',
                'valorant': 'valorant', 'valo': 'valorant',
                'overwatch': 'overwatch', 'overwatch 2': 'overwatch', 'ow': 'overwatch',
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
        overwatch_players = [
            'proper', 'coluge', 'violet', 's9mm', 'chiyou', 'kevster', 'happy', 'space',
            'skewed', 'ultraviolet', 'kai', 'hanbin', 'edison', 'fearless', 'lip', 'shu'
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
        if any(n in player for n in overwatch_players):
            return 'overwatch'
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
        overwatch_keywords = ['eliminations', 'final blows', 'objective', 'healing', 'damage done']
        if any(k in stat for k in overwatch_keywords):
            return 'overwatch'
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
        'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'golf']}

# Debug: sidebar summary of grouped counts to verify routing
try:
    st.sidebar.subheader("Props by sport (debug)")
    for k in ['basketball', 'football', 'college_football', 'hockey', 'soccer', 'tennis', 'baseball',
              'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'golf']:
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
    display_sport_picks("Basketball", picks, "🏀", sport_key="basketball")

with tabs[2]:
    picks = (get_cached_data("football_props") or {}).get('data', [])
    display_sport_picks("NFL", picks, "🏈", sport_key="football")

with tabs[3]:
    picks = (get_cached_data("college_football_props") or {}).get('data', [])
    display_sport_picks("College Football", picks, "🎓", sport_key="college_football")

with tabs[4]:
    picks = (get_cached_data("tennis_props") or {}).get('data', [])
    display_sport_picks("Tennis", picks, "🎾", sport_key="tennis")

with tabs[5]:
    picks = (get_cached_data("baseball_props") or {}).get('data', [])
    display_sport_picks("Baseball", picks, "⚾", sport_key="baseball")

with tabs[6]:
    picks = (get_cached_data("hockey_props") or {}).get('data', [])
    display_sport_picks("Hockey", picks, "🏒", sport_key="hockey")

with tabs[7]:
    picks = (get_cached_data("soccer_props") or {}).get('data', [])
    display_sport_picks("Soccer", picks, "⚽", sport_key="soccer")

with tabs[8]:
    picks = (get_cached_data("csgo_props") or {}).get('data', [])
    display_sport_picks("CS:GO", picks, "🔫", sport_key="csgo")

with tabs[9]:
    picks = (get_cached_data("league_of_legends_props") or {}).get('data', [])
    display_sport_picks("League of Legends", picks, "🧙", sport_key="league_of_legends")

with tabs[10]:
    picks = (get_cached_data("dota2_props") or {}).get('data', [])
    display_sport_picks("Dota 2", picks, "🐉", sport_key="dota2")

with tabs[11]:
    picks = (get_cached_data("valorant_props") or {}).get('data', [])
    display_sport_picks("Valorant", picks, "�", sport_key="valorant")

with tabs[12]:
    picks = (get_cached_data("overwatch_props") or {}).get('data', [])
    display_sport_picks("Overwatch", picks, "🛡️", sport_key="overwatch")

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