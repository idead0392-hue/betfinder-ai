"""
BetFinder AI - Clean Redesigned Version
Professional dark theme with modern interface
"""

import streamlit as st
import pandas as pd
import os
import time
import json
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:  # Python <3.9 fallback
    ZoneInfo = None
from ui_redesign import (
    apply_dark_theme,
    render_header, 
    render_sport_section,
    render_stats_sidebar,
    render_agent_controls,
    render_routing_status,
    render_live_data_status,
    get_sport_emoji
)
from page_utils import (
    load_prizepicks_csv_grouped,
    get_effective_csv_path,
    ensure_fresh_csv,
    render_validated_props_for_sport,
)
from sport_agents import SportAgent

# Page configuration
st.set_page_config(
    page_title="BetFinder AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
apply_dark_theme()

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'cache_timestamp' not in st.session_state:
    st.session_state.cache_timestamp = {}
if 'agent_manager' not in st.session_state:
    # Initialize the agent manager with OpenAI integration
    try:
        from agent_integration import create_agent_manager
        st.session_state.agent_manager = create_agent_manager(
            use_openai=True,  # Enable OpenAI routing
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        st.session_state.openai_enabled = True
    except Exception as e:
        # Fallback to local agents only
        st.session_state.agent_manager = None
        st.session_state.openai_enabled = False

# Helpers for banner and top agents (no sample data)
def _get_last_updated_et() -> str:
    """Return the most recent last_updated timestamp from CSV rendered in ET."""
    try:
        csv_path = get_effective_csv_path()
        grouped = load_prizepicks_csv_grouped(csv_path)
        latest_iso = None
        for props in grouped.values():
            for p in props:
                lu = p.get('last_updated') or p.get('Last_Updated')
                if lu and (latest_iso is None or lu > latest_iso):
                    latest_iso = lu
        if not latest_iso:
            return "Unknown"
        # Normalize and convert to ET
        try:
            dt = datetime.fromisoformat(latest_iso.replace('Z', '+00:00'))
        except Exception:
            try:
                dt = datetime.strptime(latest_iso, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            except Exception:
                return "Unknown"
        if ZoneInfo:
            dt_et = dt.astimezone(ZoneInfo('America/New_York'))
            return dt_et.strftime('%I:%M %p ET').lstrip('0')
        return dt.strftime('%I:%M %p UTC').lstrip('0')
    except Exception:
        return "Unknown"

def _load_top_agents_from_ledger(path: str, top_n: int = 3):
    """Aggregate real agent stats from picks_ledger.json into a compact structure."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        picks = data.get('picks', [])
        agents = {}
        for p in picks:
            agent_key = p.get('agent_type') or 'unknown_agent'
            agent_entry = agents.setdefault(agent_key, {
                'name': agent_key.replace('_', ' ').title(),
                'wins': 0,
                'picks': []
            })
            agent_entry['picks'].append(p)
            if (p.get('outcome') or '').lower() == 'win':
                agent_entry['wins'] += 1
        agent_list = []
        for ag in agents.values():
            ag['picks'].sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            pending = [x for x in ag['picks'] if (x.get('outcome') or 'pending').lower() == 'pending']
            best_hot_pick = pending[0].get('pick_description') if pending else None
            # Build recent picks with structured fields
            recent_picks = []
            for x in ag['picks'][:5]:
                desc = x.get('pick_description') or 'Pick'
                outcome_raw = (x.get('outcome') or 'pending').lower()
                if outcome_raw not in ('win', 'loss'):
                    outcome = 'Pending'
                else:
                    outcome = outcome_raw.title()
                ts_raw = x.get('timestamp') or ''
                try:
                    dt = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
                    if ZoneInfo:
                        dt = dt.astimezone(ZoneInfo('America/New_York'))
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    date_str = ts_raw.replace('T', ' ')[:16]
                recent_picks.append({'pick': desc, 'result': outcome, 'date': date_str})
            agent_list.append({
                'name': ag['name'],
                'wins': ag['wins'],
                'best_hot_pick': best_hot_pick,
                'recent_picks': recent_picks
            })
        agent_list.sort(key=lambda a: (a['wins'], len(a['recent_picks'])), reverse=True)
        return agent_list[:top_n]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def load_sport_picks(sport_key, cap=50):
    """Load picks for a specific sport using enhanced agent routing"""
    try:
        # Load props data
        csv_path = get_effective_csv_path()
        ensure_fresh_csv(csv_path, 60, target_sport=sport_key)  # Refresh every minute (sport-scoped)
        grouped = load_prizepicks_csv_grouped(csv_path)
        items = grouped.get(sport_key, [])

        # STRICT validation before any analysis
        items = render_validated_props_for_sport(items, sport_key)
        
        if not items:
            return []
            
        # Limit props for analysis
        capped = items[:cap]
        
        # Try enhanced agent manager first
        if st.session_state.agent_manager:
            try:
                result = st.session_state.agent_manager.analyze_props(
                    sport=sport_key,
                    props_data=capped,
                    include_stats=True,
                    include_context=True
                )
                
                if result.get('success', False):
                    return result.get('picks', [])
                else:
                    st.warning(f"Enhanced analysis failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.warning(f"Enhanced agent error: {e}")
        
        # Fallback to original agent system
        from sport_agents import (
            BasketballAgent, FootballAgent, CollegeFootballAgent,
            TennisAgent, BaseballAgent, HockeyAgent, SoccerAgent,
            CSGOAgent, LeagueOfLegendsAgent, Dota2Agent,
            VALORANTAgent, OverwatchAgent, GolfAgent
        )
        
        agent_map = {
            'basketball': BasketballAgent,
            'football': FootballAgent,
            'college_football': CollegeFootballAgent,
            'tennis': TennisAgent,
            'baseball': BaseballAgent,
            'hockey': HockeyAgent,
            'soccer': SoccerAgent,
            'csgo': CSGOAgent,
            'league_of_legends': LeagueOfLegendsAgent,
            'dota2': Dota2Agent,
            'valorant': VALORANTAgent,
            'overwatch': OverwatchAgent,
            'golf': GolfAgent
        }
        
        AgentClass = agent_map.get(sport_key, SportAgent)
        agent = AgentClass()
        
        # Generate picks using local agent
        picks = agent.make_picks(props_data=capped, log_to_ledger=False)
        return picks
        
    except Exception as e:
        st.error(f"Error loading {sport_key} picks: {e}")
        return []

def main():
    """Main app function"""
    
    # Custom full-width banner with header and hot picks
    st.markdown("""
    <style>
    .big-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #262730;
        color: #fff;
        padding: 1rem 2rem;
        border-radius: 7px;
        margin-bottom: 1.2rem;
    }
    .hot-picks {
        min-width: 300px;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

    # Compute dynamic values for banner (no sample data)
    last_updated_et = _get_last_updated_et()
    top_agents = _load_top_agents_from_ledger(os.path.join(os.getcwd(), 'picks_ledger.json'), top_n=3)

    # Build right-side hot picks summary lines without samples
    if top_agents:
        lines = []
        for ag in top_agents:
            summary = ag['best_hot_pick'] if ag['best_hot_pick'] else 'No active picks'
            lines.append(f"{ag['name']} ({ag['wins']} wins): {summary}")
        right_html = "<br>".join(lines)
    else:
        right_html = "No agent activity yet"


        # Unified top banner with inline Hot Picks and expanders (using Streamlit columns)
        banner = st.container()
        with banner:
            left, right = st.columns([7, 5])
            with left:
                st.markdown("# 🎯 BetFinder AI")
                st.markdown("<span style='color: #55ff55; font-weight: bold;'>🟢 Live Data Active</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 0.9em;'>Last updated: {last_updated_et}</span>", unsafe_allow_html=True)
            with right:
                st.markdown("### 🔥 Hot Picks by Top Agents")
                if top_agents:
                    for ag in top_agents:
                        inline_pick = ag['best_hot_pick'] or 'No active picks'
                        st.markdown(f"**{ag['name']} ({ag['wins']} wins):** {inline_pick}")
                        with st.expander("View pick history/results", expanded=False):
                            if ag['recent_picks']:
                                for rp in ag['recent_picks']:
                                    color = 'green' if rp['result'] == 'Win' else ('red' if rp['result'] == 'Loss' else '#9ca3af')
                                    emoji = '✅' if rp['result'] == 'Win' else ('❌' if rp['result'] == 'Loss' else '⏳')
                                    st.markdown(
                                        f"<span style='color:{color};font-weight:bold;'>{emoji} {rp['pick']} ({rp['result']}) • {rp['date']}</span>",
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("No recent history")
                else:
                    st.info("No agent activity yet")
    # Unified Hot Picks section once: inline + expander under banner
    if top_agents:
        st.markdown("### 🔥 Hot Picks by Top Agents")
        for ag in top_agents:
            inline_pick = ag['best_hot_pick'] or 'No active picks'
            st.markdown(f"**{ag['name']} ({ag['wins']} wins):** {inline_pick}")
            with st.expander("View pick history/results", expanded=False):
                if ag['recent_picks']:
                    for rp in ag['recent_picks']:
                        color = 'green' if rp['result'] == 'Win' else ('red' if rp['result'] == 'Loss' else '#9ca3af')
                        emoji = '✅' if rp['result'] == 'Win' else ('❌' if rp['result'] == 'Loss' else '⏳')
                        st.markdown(
                            f"<span style='color:{color};font-weight:bold;'>{emoji} {rp['pick']} ({rp['result']}) • {rp['date']}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No recent history")

    # Main content area (full width)
    st.markdown("### Today's Top Picks")

    # Option to switch between tab layouts
    layout_option = st.selectbox(
        "Layout Style:", 
        ["Compact Tabs", "Grouped Sports"], 
        index=0,
        help="Choose how to display sports categories"
    )

    if layout_option == "Compact Tabs":
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <small style="color: #9ca3af;">
                💡 Scroll horizontally to see all sports
            </small>
        </div>
        """, unsafe_allow_html=True)

        sport_tabs = st.tabs([
            "🏀 NBA", "🏈 NFL", "🏈 CFB", 
            "🎾 Tennis", "⚾ MLB", "🏒 NHL", "⚽ Soccer",
            "🎮 CS", "🎮 LoL", "🎮 Dota", "🎮 Val", 
            "🎮 OW", "⛳ Golf"
        ])

        sports_config = [
            ('basketball', 'NBA Basketball'),
            ('football', 'NFL Football'), 
            ('college_football', 'College Football'),
            ('tennis', 'Tennis'),
            ('baseball', 'MLB Baseball'),
            ('hockey', 'NHL Hockey'),
            ('soccer', 'Soccer'),
            ('csgo', 'CS:GO Esports'),
            ('league_of_legends', 'League of Legends'),
            ('dota2', 'Dota 2 Esports'),
            ('valorant', 'Valorant Esports'),
            ('overwatch', 'Overwatch Esports'),
            ('golf', 'PGA Golf')
        ]

        for i, (sport_key, sport_name) in enumerate(sports_config):
            with sport_tabs[i]:
                with st.spinner(f"Loading {sport_name} picks..."):
                    picks = load_sport_picks(sport_key, cap=20)

                if picks:
                    render_sport_section(
                        sport_name, 
                        picks, 
                        get_sport_emoji(sport_key)
                    )
                else:
                    st.info("No valid props available right now.")

    else:  # Grouped Sports layout
        with st.expander("🏟️ **Traditional Sports**", expanded=True):
            traditional_tabs = st.tabs(["🏀 NBA", "🏈 NFL", "🏈 CFB", "🎾 Tennis", "⚾ MLB", "🏒 NHL", "⚽ Soccer"])

            traditional_sports = [
                ('basketball', 'NBA Basketball'),
                ('football', 'NFL Football'), 
                ('college_football', 'College Football'),
                ('tennis', 'Tennis'),
                ('baseball', 'MLB Baseball'),
                ('hockey', 'NHL Hockey'),
                ('soccer', 'Soccer')
            ]

            for i, (sport_key, sport_name) in enumerate(traditional_sports):
                with traditional_tabs[i]:
                    with st.spinner(f"Loading {sport_name} picks..."):
                        picks = load_sport_picks(sport_key, cap=20)

                    if picks:
                        render_sport_section(
                            sport_name, 
                            picks, 
                            get_sport_emoji(sport_key)
                        )
                    else:
                        st.info("No valid props available right now.")

        with st.expander("🎮 **Esports**", expanded=False):
            esports_tabs = st.tabs(["🎮 CS", "🎮 LoL", "🎮 Dota", "🎮 Val", "🎮 OW"])

            esports_sports = [
                ('csgo', 'CS:GO Esports'),
                ('league_of_legends', 'League of Legends'),
                ('dota2', 'Dota 2 Esports'),
                ('valorant', 'Valorant Esports'),
                ('overwatch', 'Overwatch Esports')
            ]

            for i, (sport_key, sport_name) in enumerate(esports_sports):
                with esports_tabs[i]:
                    with st.spinner(f"Loading {sport_name} picks..."):
                        picks = load_sport_picks(sport_key, cap=20)

                    if picks:
                        render_sport_section(
                            sport_name, 
                            picks, 
                            get_sport_emoji(sport_key)
                        )
                    else:
                        st.info("No valid props available right now.")

        with st.expander("⛳ **Other Sports**", expanded=False):
            st.markdown("#### ⛳ PGA Golf")
            with st.spinner("Loading PGA Golf picks..."):
                picks = load_sport_picks('golf', cap=20)

            if picks:
                render_sport_section(
                    'PGA Golf', 
                    picks, 
                    get_sport_emoji('golf')
                )
            else:
                st.info("No valid props available right now.")

if __name__ == "__main__":
    main()