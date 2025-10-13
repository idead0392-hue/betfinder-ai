"""
Demo script showing how to use PrizePicksProvider and render_props.

This is a standalone example demonstrating the integration pattern
described in the problem statement.

Run with: streamlit run demo_render_props.py
"""

import streamlit as st
from typing import List, Dict, Optional

# Import the provider and render function
try:
    from api_providers import PrizePicksProvider, APIResponse
    PROVIDER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import PrizePicksProvider: {e}")
    PROVIDER_AVAILABLE = False
    PrizePicksProvider = None

try:
    from page_utils import render_props
except ImportError as e:
    st.error(f"Failed to import render_props: {e}")
    st.stop()


def main():
    """Main demo application"""
    
    st.set_page_config(
        page_title="PrizePicks Props Demo",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 PrizePicks Props Renderer Demo")
    st.markdown("""
    This demo shows how to fetch and render PrizePicks props data using
    the `PrizePicksProvider` and `render_props` function.
    """)
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    sport_filter = st.sidebar.selectbox(
        "Sport Filter",
        ["All", "basketball", "football", "baseball", "hockey", "soccer", "csgo", "league_of_legends"],
        help="Filter props by sport (optional)"
    )
    
    max_props = st.sidebar.slider(
        "Maximum Props",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="Maximum number of props to fetch and display"
    )
    
    # Action buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fetch_button = st.button("🔄 Fetch Props", type="primary", use_container_width=True)
    with col2:
        demo_button = st.button("📊 Demo Data", use_container_width=True)
    
    # Main content area
    if fetch_button:
        fetch_and_display_props(sport_filter, max_props)
    elif demo_button:
        show_demo_data(max_props)
    else:
        st.info("👈 Use the sidebar to fetch props or view demo data")
        show_instructions()


def fetch_and_display_props(sport_filter: str, max_props: int):
    """
    Fetch props from PrizePicksProvider and render them.
    This follows the pattern from the problem statement.
    """
    
    # Example: fetch props and render them
    props_response = None
    
    if PROVIDER_AVAILABLE and PrizePicksProvider is not None:
        with st.spinner("Fetching props from PrizePicks..."):
            try:
                # Initialize provider
                provider = PrizePicksProvider()
                
                # Fetch props (with optional sport filter)
                sport_param = None if sport_filter == "All" else sport_filter
                resp = provider.get_props(sport=sport_param, max_props=max_props)
                
                # Check if successful
                if getattr(resp, "success", False):
                    # Extract data from response
                    if isinstance(resp.data, dict) and "data" in resp.data:
                        props_response = resp.data.get("data")
                    else:
                        props_response = resp.data
                    
                    st.success(f"✅ Successfully fetched {len(props_response)} props")
                    
                    # Display response metadata
                    with st.expander("📊 Response Details"):
                        st.write({
                            "Success": resp.success,
                            "Response Time": f"{resp.response_time:.2f}s" if resp.response_time else "N/A",
                            "Props Count": len(props_response) if props_response else 0,
                            "Status Code": resp.status_code
                        })
                else:
                    st.error(f"Provider error: {getattr(resp, 'error_message', 'Unknown')}")
                    st.info("Showing demo data instead")
                    props_response = get_demo_data()
                    
            except Exception as e:
                st.error(f"Exception occurred: {str(e)}")
                st.info("Showing demo data instead")
                props_response = get_demo_data()
    else:
        # Demo data if provider missing
        st.warning("⚠️ PrizePicksProvider not available")
        props_response = get_demo_data()
    
    # Render the props using the render_props function
    if props_response:
        st.markdown("---")
        render_props(props_response, top_n=max_props)
    else:
        st.error("No props data available to display")


def show_demo_data(max_props: int):
    """Show demo data without fetching from provider"""
    st.info("📊 Displaying demo data")
    demo_props = get_demo_data()
    render_props(demo_props, top_n=max_props)


def get_demo_data() -> List[Dict]:
    """
    Generate demo props data.
    This matches the structure from the problem statement.
    """
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
        },
        {
            "player_name": "Messi",
            "team": "MIA",
            "matchup": "MIA vs NYC",
            "stat_type": "goals",
            "line": 0.5,
            "league": "MLS",
            "odds": -110,
            "confidence": 65.0,
            "expected_value": 3.8,
            "sport": "soccer"
        },
        {
            "player_name": "s1mple",
            "team": "NAVI",
            "matchup": "NAVI vs FaZe",
            "stat_type": "kills",
            "line": 20.5,
            "league": "CS2",
            "odds": -110,
            "confidence": 73.0,
            "expected_value": 4.5,
            "sport": "csgo"
        },
        {
            "player_name": "Faker",
            "team": "T1",
            "matchup": "T1 vs Gen.G",
            "stat_type": "kills",
            "line": 3.5,
            "league": "LCK",
            "odds": -110,
            "confidence": 76.0,
            "expected_value": 5.8,
            "sport": "league_of_legends"
        }
    ]


def show_instructions():
    """Show usage instructions"""
    st.markdown("### 📖 How to Use")
    
    st.markdown("""
    #### Fetch Live Props
    1. Select a sport filter (optional) from the sidebar
    2. Adjust the maximum number of props to display
    3. Click the **🔄 Fetch Props** button to load live data
    
    #### View Demo Data
    - Click the **📊 Demo Data** button to see the interface with sample data
    
    #### Features
    - ✅ Real-time props fetching from PrizePicks
    - ✅ Clean, structured display with Pandas DataFrames
    - ✅ Sport filtering capability
    - ✅ Summary statistics (total props, leagues, players)
    - ✅ Error handling with fallback to demo data
    """)
    
    st.markdown("---")
    st.markdown("### 💻 Code Example")
    
    st.code("""
# Example: fetch props and render them
props_response = None
if PROVIDER_AVAILABLE and PrizePicksProvider is not None:
    provider = PrizePicksProvider()
    resp = provider.get_props(sport="basketball")
    if getattr(resp, "success", False):
        props_response = resp.data.get("data") if isinstance(resp.data, dict) and "data" in resp.data else resp.data
    else:
        st.error(f"Provider error: {getattr(resp, 'error_message', 'Unknown')}")
else:
    # Demo data if provider missing
    props_response = [
        {"player_name": "LeBron James", "team": "LAL", "matchup": "LAL vs BOS", "stat_type": "points", "line": 27.5}
    ]

# Render the props
render_props(props_response, top_n=25)
    """, language="python")


if __name__ == "__main__":
    main()
