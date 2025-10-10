import streamlit as st
import pandas as pd
import requests
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from lxml import etree

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

def load_sports_data_with_agents():
    """Load all sports data using sport agents"""
    if st.session_state.data_loaded:
        return
    
    with st.spinner("Loading sports data using AI agents..."):
        try:
            # Import sport agents
            from sport_agents import (
                BasketballAgent, FootballAgent, TennisAgent, 
                BaseballAgent, HockeyAgent, SoccerAgent, EsportsAgent
            )
            
            # Load data using sport agents
            agents_and_sports = [
                (BasketballAgent(), "basketball"),
                (FootballAgent(), "football"), 
                (TennisAgent(), "tennis"),
                (BaseballAgent(), "baseball"),
                (HockeyAgent(), "hockey"),
                (SoccerAgent(), "soccer"),
                (EsportsAgent(), "esports")
            ]
            
            for agent, sport in agents_and_sports:
                try:
                    picks = agent.make_picks()
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

def display_sport_picks(sport_name, picks, sport_emoji):
    """Display picks for a specific sport"""
    st.markdown(f'<div class="section-title">{sport_name}<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    if picks:
        st.markdown(f"### {sport_emoji} {sport_name} Player Props")
        
        # Display picks in a nice format
        for i, pick in enumerate(picks[:10]):  # Show top 10 picks
            if isinstance(pick, dict):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    player_name = pick.get('player_name', pick.get('description', 'Unknown Player'))
                    stat_type = pick.get('stat_type', 'Unknown Stat')
                    line = pick.get('line', 'N/A')
                    bet_type = pick.get('bet_type', 'over')
                    
                    st.markdown(f"**{player_name}**")
                    st.markdown(f"{stat_type.title()} {bet_type.title()} {line}")
                    
                    if 'reasoning' in pick:
                        st.caption(f"💡 {pick['reasoning']}")
                
                with col2:
                    confidence = pick.get('confidence', 0)
                    if confidence >= 80:
                        conf_color = "🟢"
                    elif confidence >= 70:
                        conf_color = "🟡"
                    else:
                        conf_color = "🔴"
                    
                    st.metric("Confidence", f"{confidence}%", delta=None)
                    st.markdown(f"{conf_color} {confidence}%")
                
                with col3:
                    odds = pick.get('odds', -110)
                    if 'ml_prediction' in pick:
                        ml_pred = pick['ml_prediction']
                        edge = ml_pred.get('edge', 0)
                        st.metric("ML Edge", f"{edge:.1%}")
                        if edge > 0.05:
                            st.markdown("🚀 **High Value**")
                    else:
                        st.metric("Odds", f"{odds}")
                
                st.markdown("---")
        
        # Show additional statistics
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

# CSS
st.markdown("""
<style>
    .section-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .time {
        font-size: 14px;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1>🎯 BetFinder AI</h1>
    <p style="font-size: 18px; color: #666;">Advanced Sports Betting Analysis with ML Predictions</p>
</div>
""", unsafe_allow_html=True)

# Load data
load_sports_data_with_agents()

# Create tabs
tab_names = ["🏀 Basketball", "🏈 Football", "🎾 Tennis", "⚾ Baseball", "🏒 Hockey", "⚽ Soccer", "🎮 Esports"]
tabs = st.tabs(tab_names)

# Basketball Tab
with tabs[0]:
    picks = (get_cached_data("basketball_props") or {}).get('data', [])
    display_sport_picks("Basketball", picks, "🏀")

# Football Tab  
with tabs[1]:
    picks = (get_cached_data("football_props") or {}).get('data', [])
    display_sport_picks("Football", picks, "🏈")

# Tennis Tab
with tabs[2]:
    picks = (get_cached_data("tennis_props") or {}).get('data', [])
    display_sport_picks("Tennis", picks, "🎾")

# Baseball Tab
with tabs[3]:
    picks = (get_cached_data("baseball_props") or {}).get('data', [])
    display_sport_picks("Baseball", picks, "⚾")

# Hockey Tab
with tabs[4]:
    picks = (get_cached_data("hockey_props") or {}).get('data', [])
    display_sport_picks("Hockey", picks, "🏒")

# Soccer Tab
with tabs[5]:
    picks = (get_cached_data("soccer_props") or {}).get('data', [])
    display_sport_picks("Soccer", picks, "⚽")

# Esports Tab
with tabs[6]:
    picks = (get_cached_data("esports_props") or {}).get('data', [])
    display_sport_picks("Esports", picks, "🎮")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 30px;">
    <p>🤖 Powered by AI Sport Agents with ML Prediction Engine</p>
    <p>📊 Real-time prop analysis with confidence scoring and edge calculation</p>
</div>
""", unsafe_allow_html=True)