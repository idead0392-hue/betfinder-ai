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
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

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
        })
    return props_data

@st.cache_data(ttl=300)
def fetch_props_data(sport: str, provider) -> Tuple[List[Dict], Optional[str]]:
    """Fetch props data using SportbexProvider."""
    if not provider:
        return generate_mock_props_data(sport), "Using mock data - provider unavailable"
    
    try:
        sport_type = getattr(SportType, sport.upper(), SportType.BASKETBALL)
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
        <h1>Player Prop Predictor</h1>
        <p>AI-powered analysis of player prop betting opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.provider is None:
        with st.spinner("Initializing SportbexProvider..."):
            provider, error = initialize_provider()
            st.session_state.provider = provider
            if error:
                st.warning(f"Provider initialization issue: {error}")
    
    st.sidebar.header("Filters")
    selected_sport = st.sidebar.selectbox("Sport", ['Basketball', 'Football'])
    
    if st.sidebar.button("Refresh Data", type="primary"):
        st.cache_data.clear()
        st.session_state.last_fetch_time = datetime.now()
        st.rerun()
    
    with st.spinner(f"Loading {selected_sport.lower()} props data..."):
        props_data, fetch_error = fetch_props_data(selected_sport, st.session_state.provider)
    
    if fetch_error:
        st.warning(fetch_error)
    
    if not props_data:
        st.error("No props data available.")
        st.stop()
    
    df = pd.DataFrame(props_data)
    
    st.subheader(f"Props Analysis ({len(df)} opportunities)")
    st.dataframe(df)

if __name__ == "__main__":
    main()
