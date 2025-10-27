"""
Test script for render_props function with PrizePicksProvider integration.

This demonstrates how to fetch and render PrizePicks props data.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List

# Check if PrizePicksProvider is available
PROVIDER_AVAILABLE = False
PrizePicksProvider = None

try:
    from api_providers import PrizePicksProvider
    PROVIDER_AVAILABLE = True
except Exception as e:
    st.warning(f"PrizePicksProvider not available: {e}")

try:
    from page_utils import render_props
except Exception as e:
    st.error(f"Failed to import render_props: {e}")
    st.stop()


def main():
    """Main test function for render_props"""
    st.set_page_config(
        page_title="PrizePicks Props Test",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 PrizePicks Props Renderer Test")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    sport_filter = st.sidebar.selectbox(
        "Filter by Sport",
        ["All", "basketball", "football", "baseball", "hockey", "soccer"]
    )
    max_props = st.sidebar.slider("Max Props to Display", 5, 100, 25)
    
    # Fetch button
    if st.sidebar.button("🔄 Fetch Props", type="primary"):
        with st.spinner("Fetching PrizePicks props..."):
            fetch_and_render_props(sport_filter, max_props)
    else:
        st.info("👈 Click 'Fetch Props' in the sidebar to load data")
        
        # Show demo with mock data
        st.markdown("### Demo with Mock Data")
        demo_props = generate_demo_props()
        render_props(demo_props, top_n=10)


def fetch_and_render_props(sport_filter: str, max_props: int):
    """Fetch props from provider and render them"""
    
    props_response = None
    
    if PROVIDER_AVAILABLE and PrizePicksProvider is not None:
        try:
            provider = PrizePicksProvider()
            
            # Fetch props (optionally filtered by sport)
            sport_param = None if sport_filter == "All" else sport_filter
            resp = provider.get_props(sport=sport_param, max_props=max_props)
            
            if getattr(resp, "success", False):
                # Extract the data from response
                if isinstance(resp.data, dict) and "data" in resp.data:
                    props_response = resp.data.get("data")
                else:
                    props_response = resp.data
                
                st.success(f"✅ Successfully fetched {len(props_response) if props_response else 0} props")
            else:
                error_msg = getattr(resp, 'error_message', 'Unknown error')
                st.error(f"❌ Provider error: {error_msg}")
                st.info("Showing demo data instead")
                props_response = generate_demo_props()
        except Exception as e:
            st.error(f"❌ Exception while fetching: {str(e)}")
            st.info("Showing demo data instead")
            props_response = generate_demo_props()
    else:
        st.warning("⚠️ PrizePicksProvider not available - showing demo data")
        props_response = generate_demo_props()
    
    # Render the props
    if props_response:
        render_props(props_response, top_n=max_props)
    else:
        st.error("No props data available to display")


def generate_demo_props() -> List[Dict]:
    """Generate demo props data for testing"""
    return [
        {
            "player_name": "LeBron James",
            "team": "LAL",
            "matchup": "LAL vs BOS",
            "stat_type": "points",
            "line": 27.5,
            "league": "NBA",
            "odds": -110,
            "confidence": 75.0,
            "expected_value": 5.2,
            "sport": "basketball"
        },
        {
            "player_name": "Stephen Curry",
            "team": "GSW",
            "matchup": "GSW vs PHX",
            "stat_type": "3-pointers made",
            "line": 4.5,
            "league": "NBA",
            "odds": -110,
            "confidence": 72.0,
            "expected_value": 4.8,
            "sport": "basketball"
        },
        {
            "player_name": "Patrick Mahomes",
            "team": "KC",
            "matchup": "KC vs BUF",
            "stat_type": "passing yards",
            "line": 285.5,
            "league": "NFL",
            "odds": -110,
            "confidence": 78.0,
            "expected_value": 6.1,
            "sport": "football"
        },
        {
            "player_name": "Aaron Judge",
            "team": "NYY",
            "matchup": "NYY vs BOS",
            "stat_type": "hits",
            "line": 1.5,
            "league": "MLB",
            "odds": -110,
            "confidence": 68.0,
            "expected_value": 3.5,
            "sport": "baseball"
        },
        {
            "player_name": "Connor McDavid",
            "team": "EDM",
            "matchup": "EDM vs CGY",
            "stat_type": "points",
            "line": 1.5,
            "league": "NHL",
            "odds": -110,
            "confidence": 80.0,
            "expected_value": 7.2,
            "sport": "hockey"
        }
    ]


if __name__ == "__main__":
    main()
