import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ============================================================================
# RAPIDAPI INTEGRATION - Pinnacle Odds API
# ============================================================================

def rapidapi_get(endpoint, params=None):
    """
    Efficient RapidAPI GET request handler for Pinnacle Odds API.
    
    Args:
        endpoint (str): API endpoint path
        params (dict): Query parameters
    
    Returns:
        dict: JSON response or error dict
    """
    url = f"https://pinnacle-odds.p.rapidapi.com/{endpoint}"
    headers = {
        "X-RapidAPI-Key": st.secrets.get("RAPIDAPI_KEY", "YOUR_API_KEY_HERE"),
        "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Example API Calls
# Get available markets
markets_data = rapidapi_get("markets")

# Function to extract props by sport/esport
def get_props_by_sport(markets_data):
    """
    Extract available props grouped by sport from markets data.
    
    Args:
        markets_data (dict): Markets API response
    
    Returns:
        dict: Dictionary mapping sport names to lists of available props
    """
    props_by_sport = {}
    
    if not markets_data or "error" in markets_data:
        return props_by_sport
    
    # Handle both list and dict responses
    markets_list = markets_data if isinstance(markets_data, list) else markets_data.get("markets", [])
    
    for market in markets_list:
        sport_name = market.get("sport", "Unknown")
        market_type = market.get("type", market.get("name", "Unknown"))
        
        if sport_name not in props_by_sport:
            props_by_sport[sport_name] = []
        
        if market_type not in props_by_sport[sport_name]:
            props_by_sport[sport_name].append(market_type)
    
    return props_by_sport

# Get props organized by sport
api_props_by_sport = get_props_by_sport(markets_data)

# Get betting status
betting_status = rapidapi_get("betting-status")

# ============================================================================
# END RAPIDAPI INTEGRATION
# ============================================================================

# Rest of the application code follows...
# Note: The following code assumes that nav_tabs, filtered_df, and value_candidates
# are defined earlier in the application. This is a partial file.

# TENNIS TAB
with nav_tabs[2]:
    st.markdown('<div class="section-title">Tennis<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    tennis_props = api_props_by_sport.get("Tennis", [])
    if tennis_props:
        st.write("**Available Props:**", ", ".join(tennis_props))
    else:
        st.info("No props available for Tennis at this time.")
    
    tennis_df = filtered_df[filtered_df["sport_title"] == "Tennis"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if tennis_df.empty:
        st.info("No tennis events available.")
    else:
        st.dataframe(tennis_df.head(100), use_container_width=True)

# BASKETBALL TAB
with nav_tabs[3]:
    st.markdown('<div class="section-title">Basketball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    basketball_props = api_props_by_sport.get("Basketball", [])
    if basketball_props:
        st.write("**Available Props:**", ", ".join(basketball_props))
    else:
        st.info("No props available for Basketball at this time.")
    
    basketball_df = filtered_df[filtered_df["sport_title"] == "Basketball"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if basketball_df.empty:
        st.info("No basketball events available.")
    else:
        st.dataframe(basketball_df.head(100), use_container_width=True)

# FOOTBALL TAB
with nav_tabs[4]:
    st.markdown('<div class="section-title">Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    football_props = api_props_by_sport.get("Football", [])
    if football_props:
        st.write("**Available Props:**", ", ".join(football_props))
    else:
        st.info("No props available for Football at this time.")
    
    football_df = filtered_df[filtered_df["sport_title"] == "Football"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if football_df.empty:
        st.info("No football events available.")
    else:
        st.dataframe(football_df.head(100), use_container_width=True)

# BASEBALL TAB
with nav_tabs[5]:
    st.markdown('<div class="section-title">Baseball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    baseball_props = api_props_by_sport.get("Baseball", [])
    if baseball_props:
        st.write("**Available Props:**", ", ".join(baseball_props))
    else:
        st.info("No props available for Baseball at this time.")
    
    baseball_df = filtered_df[filtered_df["sport_title"] == "Baseball"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if baseball_df.empty:
        st.info("No baseball events available.")
    else:
        st.dataframe(baseball_df.head(100), use_container_width=True)

# HOCKEY TAB
with nav_tabs[6]:
    st.markdown('<div class="section-title">Hockey<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    hockey_props = api_props_by_sport.get("Hockey", [])
    if hockey_props:
        st.write("**Available Props:**", ", ".join(hockey_props))
    else:
        st.info("No props available for Hockey at this time.")
    
    hockey_df = filtered_df[filtered_df["sport_title"] == "Hockey"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if hockey_df.empty:
        st.info("No hockey events available.")
    else:
        st.dataframe(hockey_df.head(100), use_container_width=True)

# SOCCER TAB
with nav_tabs[7]:
    st.markdown('<div class="section-title">Soccer<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    soccer_props = api_props_by_sport.get("Soccer", [])
    if soccer_props:
        st.write("**Available Props:**", ", ".join(soccer_props))
    else:
        st.info("No props available for Soccer at this time.")
    
    soccer_df = filtered_df[filtered_df["sport_title"] == "Soccer"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if soccer_df.empty:
        st.info("No soccer events available.")
    else:
        st.dataframe(soccer_df.head(100), use_container_width=True)

# GOLF TAB
with nav_tabs[8]:
    st.markdown('<div class="section-title">Golf<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    golf_props = api_props_by_sport.get("Golf", [])
    if golf_props:
        st.write("**Available Props:**", ", ".join(golf_props))
    else:
        st.info("No props available for Golf at this time.")
    
    golf_df = filtered_df[filtered_df["sport_title"] == "Golf"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if golf_df.empty:
        st.info("No golf events available.")
    else:
        st.dataframe(golf_df.head(100), use_container_width=True)

# MMA TAB
with nav_tabs[9]:
    st.markdown('<div class="section-title">MMA<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    mma_props = api_props_by_sport.get("MMA", [])
    if mma_props:
        st.write("**Available Props:**", ", ".join(mma_props))
    else:
        st.info("No props available for MMA at this time.")
    
    mma_df = filtered_df[filtered_df["sport_title"] == "MMA"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if mma_df.empty:
        st.info("No MMA events available.")
    else:
        st.dataframe(mma_df.head(100), use_container_width=True)

# CRICKET TAB
with nav_tabs[10]:
    st.markdown('<div class="section-title">Cricket<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Display available props
    cricket_props = api_props_by_sport.get("Cricket", [])
    if cricket_props:
        st.write("**Available Props:**", ", ".join(cricket_props))
    else:
        st.info("No props available for Cricket at this time.")
    
    cricket_df = filtered_df[filtered_df["sport_title"] == "Cricket"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if cricket_df.empty:
        st.info("No cricket events available.")
    else:
        st.dataframe(cricket_df.head(100), use_container_width=True)
