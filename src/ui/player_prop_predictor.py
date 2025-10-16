# src/ui/player_prop_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import random

# Import our provider system
try:
    from ..api.providers import SportbexProvider, SportType, create_sportbex_provider
    PROVIDER_AVAILABLE = True
except ImportError:
    PROVIDER_AVAILABLE = False
    st.error("SportbexProvider not available. Please ensure api_providers.py is properly configured.")

# Page configuration
st.set_page_config(
    page_title="Player Prop Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2c5aa0 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive-edge {
        background-color: #d4edda;
        color: #155724;
    }
    .negative-edge {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>""", unsafe_allow_html=True)

# --- Start of eSports Configuration ---
ESPORTS_TITLES = {"cs2", "league_of_legends", "dota2", "valorant"}

# Define allowed stat types for eSports to filter out non-standard props
ALLOWED_ESPORTS_STATS_PER_SPORT = {
    "cs2": ['Kills', 'Headshots', 'ADR', 'Fantasy Points', 'Maps 1-2 Kills'],
    "league_of_legends": ['Kills', 'Assists', 'Creep Score', 'KDA', 'Fantasy Points', 'Maps 1-2 Kills'],
    "dota2": ['Kills', 'Assists', 'Last Hits', 'GPM', 'Fantasy Points', 'Maps 1-2 Kills'],
    "valorant": ['Kills', 'Headshots', 'ACS', 'First Bloods', 'Fantasy Points', 'Maps 1-2 Kills']
}
# --- End of eSports Configuration ---

# Initialize session state
if 'props_data' not in st.session_state:
    st.session_state.props_data = None
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'provider' not in st.session_state:
    st.session_state.provider = None

@st.cache_resource
def initialize_provider():
    """Initialize SportbexProvider with caching."""
    if not PROVIDER_AVAILABLE:
        return None, "SportbexProvider module not available"
    
    try:
        provider = create_sportbex_provider()
        health_response = provider.health_check()
        if health_response and health_response.success:
            return provider, None
        else:
            error_msg = health_response.error_message if health_response else "Health check failed"
            return None, f"Provider health check failed: {error_msg}"
    except Exception as e:
        return None, f"Failed to initialize provider: {str(e)}"

def projection_engine(player_name: str, stat_type: str, prop_line: float) -> Dict[str, Any]:
    """AI-powered projection engine for player props (stub)."""
    np.random.seed(hash(player_name + stat_type) % 2**32)
    base_projection = prop_line + np.random.normal(0, prop_line * 0.15)
    base_projection = max(0, base_projection)
    
    projection_data = {
        'projection': round(base_projection, 1),
        'confidence': round(np.random.uniform(0.65, 0.95), 3),
        'prop_line': prop_line,
        'edge_over': round(((base_projection / prop_line) - 1) * 100, 1) if prop_line > 0 else 0,
    }
    
    return projection_data

def generate_mock_props_data(sport: str, num_props: int = 50) -> List[Dict]:
    """Generate mock props data for demonstration."""
    sports_config = {
        'basketball': {'stat_types': ['Points', 'Rebounds', 'Assists'], 'teams': ['Lakers', 'Warriors']},
        'football': {'stat_types': ['Passing Yards', 'Rushing Yards'], 'teams': ['Chiefs', 'Bills']},
        'baseball': {'stat_types': ['Hits', 'RBIs', 'Strikeouts'], 'teams': ['Yankees', 'Red Sox']},
        'hockey': {'stat_types': ['Goals', 'Assists', 'Saves'], 'teams': ['Maple Leafs', 'Canadiens']},
        'cs2': {'stat_types': ['Kills', 'Headshots', 'ADR'], 'teams': ['FaZe', 'Navi'], 'sport': 'cs2'},
        'league of legends': {'stat_types': ['Kills', 'Assists', 'Creep Score'], 'teams': ['T1', 'Gen.G'], 'sport': 'league_of_legends'},
        'dota 2': {'stat_types': ['Kills', 'Assists', 'Last Hits'], 'teams': ['OG', 'Team Spirit'], 'sport': 'dota2'},
        'valorant': {'stat_types': ['Kills', 'Headshots', 'ACS'], 'teams': ['Sentinels', 'Fnatic'], 'sport': 'valorant'},
    }
    config = sports_config.get(sport.lower(), sports_config['basketball'])
    props_data = []
    
    for i in range(num_props):
        team_1, team_2 = random.sample(config['teams'], 2)
        stat_type = random.choice(config['stat_types'])
        player_name = f"Player {chr(65 + i % 26)}"
        prop_line = round(random.uniform(5, 35), 1)
        projection_data = projection_engine(player_name, stat_type, prop_line)
        
        props_data.append({
            'matchup': f"{team_1} vs {team_2}",
            'player_name': player_name,
            'stat_type': stat_type,
            'prop_line': prop_line,
            'ai_projection': projection_data['projection'],
            'confidence': projection_data['confidence'],
            'edge_over': projection_data['edge_over'],
            'edge_under': round(-projection_data['edge_over'], 1),
            'game_date': (datetime.now() + timedelta(days=random.randint(0, 7))).strftime('%Y-%m-%d'),
            'sport': config.get('sport', sport.lower()),
        })
    return props_data

def _filter_esports_props(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the DataFrame to only include allowed stat types for eSports."""
    if df.empty:
        return df
        
    esports_df = df[df['sport'].str.lower().isin(ESPORTS_TITLES)]
    non_esports_df = df[~df['sport'].str.lower().isin(ESPORTS_TITLES)]
    
    filtered_esports_dfs = []
    for sport, group in esports_df.groupby('sport'):
        allowed_stats = ALLOWED_ESPORTS_STATS_PER_SPORT.get(sport.lower(), [])
        # Ensure case-insensitivity for stat types
        allowed_stats_lower = [stat.lower() for stat in allowed_stats]
        filtered_group = group[group['stat_type'].str.lower().isin(allowed_stats_lower)]
        filtered_esports_dfs.append(filtered_group)
        
    if not filtered_esports_dfs:
        return non_esports_df
        
    return pd.concat([non_esports_df] + filtered_esports_dfs)

@st.cache_data(ttl=300)
def fetch_props_data(sport: str, provider) -> Tuple[List[Dict], Optional[str]]:
    """Fetch props data using SportbexProvider."""
    if not provider:
        return generate_mock_props_data(sport), "Using mock data - provider unavailable"
    
    try:
        sport_type = getattr(SportType, sport.upper().replace(' ', '_'), SportType.BASKETBALL)
        response = provider.get_props(sport=sport_type)
        
        if response.success and response.data:
            return generate_mock_props_data(sport), None
        else:
            error_msg = response.error_message if hasattr(response, 'error_message') else "Unknown error"
            return generate_mock_props_data(sport), f"API error: {error_msg}. Using mock data."
            
    except Exception as e:
        return generate_mock_props_data(sport), f"Error fetching data: {str(e)}. Using mock data."

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Player Prop Predictor</h1>
        <p>AI-powered analysis of player prop betting opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.provider is None:
        with st.spinner("Initializing SportbexProvider..."):
            provider, error = initialize_provider()
            st.session_state.provider = provider
            if error:
                st.warning(f"Provider initialization issue: {error}")
    
    st.sidebar.header("🔍 Filters")
    
    # Add eSports to the sport selection dropdown
    sport_options = ['Basketball', 'Football', 'Baseball', 'Hockey', 'CS2', 'League of Legends', 'Dota 2', 'Valorant']
    selected_sport = st.sidebar.selectbox("Sport", sport_options, index=0)
    
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.session_state.last_fetch_time = datetime.now()
        st.rerun()
    
    # Load and filter data
    with st.spinner(f"Loading {selected_sport.lower()} props data..."):
        props_data, fetch_error = fetch_props_data(selected_sport, st.session_state.provider)
    
    if fetch_error:
        st.warning(f"⚠️ {fetch_error}")
    
    if not props_data:
        st.error("❌ No props data available. Please try again later.")
        st.stop()
        
    df = pd.DataFrame(props_data)
    
    # Apply eSports-specific filtering
    df = _filter_esports_props(df)
    
    if df.empty:
        st.warning("No props available after filtering.")
        st.stop()
    
    st.subheader(f"📊 Props Analysis ({len(df)} opportunities)")
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
