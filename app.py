# BetFinder AI - Instagram Sport Bar Template Revamp
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
st.write("### Available Markets")
st.write(markets_data)

# Get betting status
betting_status = rapidapi_get("betting-status")
st.write("### Betting Status")
st.write(betting_status)

# ============================================================================
# END RAPIDAPI INTEGRATION
# ============================================================================

# Rest of the application code follows...

st.warning("⚠️ No top picks available at this time.")
    
else:
    st.dataframe(value_candidates.sort_values("edge_pct", ascending=False).head(10), use_container_width=True)

# TENNIS TAB
with nav_tabs[2]:
    st.markdown('<div class="section-title">Tennis<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    tennis_df = filtered_df[filtered_df["sport_title"] == "Tennis"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if tennis_df.empty:
        st.info("No tennis events available.")
    else:
        st.dataframe(tennis_df.head(100), use_container_width=True)

# BASKETBALL TAB
with nav_tabs[3]:
    st.markdown('<div class="section-title">Basketball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    basketball_df = filtered_df[filtered_df["sport_title"] == "Basketball"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if basketball_df.empty:
        st.info("No basketball events available.")
    else:
        st.dataframe(basketball_df.head(100), use_container_width=True)

# FOOTBALL TAB
with nav_tabs[4]:
    st.markdown('<div class="section-title">Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    football_df = filtered_df[filtered_df["sport_title"] == "Football"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if football_df.empty:
        st.info("No football events available.")
    else:
        st.dataframe(football_df.head(100), use_container_width=True)

# BASEBALL TAB
with nav_tabs[5]:
    st.markdown('<div class="section-title">Baseball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    baseball_df = filtered_df[filtered_df["sport_title"] == "Baseball"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if baseball_df.empty:
        st.info("No baseball events available.")
    else:
        st.dataframe(baseball_df.head(100), use_container_width=True)

# HOCKEY TAB
with nav_tabs[6]:
    st.markdown('<div class="section-title">Hockey<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    hockey_df = filtered_df[filtered_df["sport_title"] == "Hockey"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if hockey_df.empty:
        st.info("No hockey events available.")
    else:
        st.dataframe(hockey_df.head(100), use_container_width=True)

# SOCCER TAB
with nav_tabs[7]:
    st.markdown('<div class="section-title">Soccer<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    soccer_df = filtered_df[filtered_df["sport_title"] == "Soccer"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if soccer_df.empty:
        st.info("No soccer events available.")
    else:
        st.dataframe(soccer_df.head(100), use_container_width=True)

# GOLF TAB
with nav_tabs[8]:
    st.markdown('<div class="section-title">Golf<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    golf_df = filtered_df[filtered_df["sport_title"] == "Golf"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if golf_df.empty:
        st.info("No golf events available.")
    else:
        st.dataframe(golf_df.head(100), use_container_width=True)

# MMA TAB
with nav_tabs[9]:
    st.markdown('<div class="section-title">MMA<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    mma_df = filtered_df[filtered_df["sport_title"] == "MMA"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if mma_df.empty:
        st.info("No MMA events available.")
    else:
        st.dataframe(mma_df.head(100), use_container_width=True)

# CRICKET TAB
with nav_tabs[10]:
    st.markdown('<div class="section-title">Cricket<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    cricket_df = filtered_df[filtered_df["sport_title"] == "Cricket"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if cricket_df.empty:
        st.info("No cricket events available.")
    else:
        st.dataframe(cricket_df.head(100), use_container_width=True)
