"""
Example: Fetch and Render PrizePicks Props

This file demonstrates the exact usage pattern from the problem statement,
properly formatted and integrated with the BetFinder AI codebase.

This is a working example that can be used as a reference or template
for integrating PrizePicks props rendering in other parts of the application.
"""

import streamlit as st
from typing import List, Dict, Optional

# Check provider availability
PROVIDER_AVAILABLE = False
PrizePicksProvider = None

try:
    from api_providers import PrizePicksProvider
    PROVIDER_AVAILABLE = True
except ImportError:
    pass

# Import render function
try:
    from page_utils import render_props
except ImportError as e:
    st.error(f"Cannot import render_props: {e}")
    st.stop()


def fetch_and_render_props_example():
    """
    Example: fetch props and render them
    
    This follows the exact pattern from the problem statement,
    properly formatted and integrated.
    """
    
    # Initialize response variable
    props_response = None
    
    # Try to fetch from PrizePicks provider
    if PROVIDER_AVAILABLE and PrizePicksProvider is not None:
        # Initialize provider (optional: pass scraper_module name if needed)
        provider = PrizePicksProvider()
        
        # Fetch props with optional filters
        resp = provider.get_props(sport="basketball")  # pass filters as required
        
        # Check if successful
        if getattr(resp, "success", False):
            # Extract props data from response
            props_response = resp.data.get("data") if isinstance(resp.data, dict) and "data" in resp.data else resp.data
        else:
            # Handle error case
            st.error(f"Provider error: {getattr(resp, 'error_message', 'Unknown')}")
    else:
        # Demo data if provider missing
        props_response = [
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
            }
        ]
    
    # Render the props using the render_props function
    # (Optional: normalize in-place if your provider returns raw items)
    # If you have prizepicks_provider normalized output already, you can pass it directly
    if props_response:
        render_props(props_response, top_n=25)
    else:
        st.warning("No props data available")


def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="PrizePicks Example",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 PrizePicks Props - Usage Example")
    st.markdown("""
    This page demonstrates the exact pattern from the problem statement
    for fetching and rendering PrizePicks props data.
    """)
    st.markdown("---")
    
    # Show the example code
    with st.expander("📖 View Source Code", expanded=False):
        st.code('''
# Example: fetch props and render them
props_response = None
if PROVIDER_AVAILABLE and PrizePicksProvider is not None:
    provider = PrizePicksProvider()  # optional: pass scraper_module name if needed
    resp = provider.get_props(sport="basketball")  # pass filters as required
    if getattr(resp, "success", False):
        props_response = resp.data.get("data") if isinstance(resp.data, dict) and "data" in resp.data else resp.data
    else:
        st.error(f"Provider error: {getattr(resp, 'error_message', 'Unknown')}")
else:
    # Demo data if provider missing
    props_response = [
        {"player_name": "LeBron James", "team": "LAL", "matchup": "LAL vs BOS", "stat_type": "points", "line": 27.5}
    ]

# Normalize in-place if your provider returns raw items (optional)
# If you have prizepicks_provider normalized output already, you can pass it directly
render_props(props_response, top_n=25)
        ''', language='python')
    
    # Run the example
    st.markdown("### Live Demo")
    
    # Add controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("Click the button to fetch and render props using the example pattern")
    with col2:
        if st.button("▶️ Run Example", type="primary"):
            st.session_state.run_example = True
    
    # Execute the example if button was clicked
    if st.session_state.get('run_example', False):
        with st.spinner("Fetching and rendering props..."):
            fetch_and_render_props_example()
    
    # Show provider status
    st.sidebar.header("📊 Status")
    if PROVIDER_AVAILABLE:
        st.sidebar.success("✅ PrizePicksProvider Available")
    else:
        st.sidebar.warning("⚠️ Using Demo Data")
    
    # Additional information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Documentation")
    st.sidebar.markdown("""
    - `PRIZEPICKS_INTEGRATION.md` - Full integration guide
    - `demo_render_props.py` - Interactive demo
    - `test_prizepicks_integration.py` - Unit tests
    """)


if __name__ == "__main__":
    main()
