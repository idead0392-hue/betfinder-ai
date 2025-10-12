"""
BetFinder AI - Clean Redesigned Version
Professional dark theme with modern interface
"""

import streamlit as st
import pandas as pd
import os
import time
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
    
    # Render header
    render_header()
    
    # Live data status
    render_live_data_status()
    
    # Show routing status
    render_routing_status()
    
    # Sidebar stats and agent controls
    render_stats_sidebar()
    render_agent_controls()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Today's Top Picks")
        
        # Option to switch between tab layouts
        layout_option = st.selectbox(
            "Layout Style:", 
            ["Compact Tabs", "Grouped Sports"], 
            index=0,
            help="Choose how to display sports categories"
        )
        
        if layout_option == "Compact Tabs":
            # Responsive container for tabs
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <small style="color: #9ca3af;">
                    💡 Scroll horizontally to see all sports
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Sport selection tabs - shortened for better fit
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
            # Traditional Sports Group
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
            
            # Esports Group
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
            
            # Other Sports
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
    
    with col2:
        st.markdown("### ⚡ Quick Stats")
        
        # Load sample data for metrics
        try:
            csv_path = get_effective_csv_path()
            grouped = load_prizepicks_csv_grouped(csv_path)
            total_props = sum(len(props) for props in grouped.values())
            
            st.metric("Live Props", f"{total_props:,}")
            st.metric("Markets", len(grouped))
            
            # Top sports by volume
            st.markdown("**Top Markets:**")
            sorted_sports = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
            
            for sport, props in sorted_sports[:5]:
                emoji = get_sport_emoji(sport)
                st.markdown(f"{emoji} **{sport.title()}**: {len(props)} props")
                
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        
        # Recent activity
        st.markdown("---")
        st.markdown("### 📈 Recent Activity")
        st.markdown("🔥 **Hot Pick**: Bo Nix Over 1.5 TD")
        st.markdown("💰 **Value Bet**: Discount classification")
        st.markdown("📊 **ML Edge**: +15.3% expected value")

if __name__ == "__main__":
    main()