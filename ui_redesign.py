"""
Clean UI redesign for BetFinder AI
Modern dark theme with consistent styling
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

def apply_dark_theme():
    """Apply consistent dark theme styling"""
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #0a0e1a;
        color: #ffffff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1b69 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #2d3748;
    }
    
    /* Sport sections */
    .sport-section {
        background: #111827;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #374151;
    }
    
    /* Pick cards */
    .pick-card {
        background: #1f2937;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 3px solid #6366f1;
        transition: all 0.2s ease;
    }
    
    .pick-card:hover {
        background: #252e3f;
        border-left-color: #8b5cf6;
    }
    
    /* Badges */
    .badge-demon { 
        background: #dc2626; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 4px; 
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-discount { 
        background: #059669; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 4px; 
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-decent { 
        background: #0891b2; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 4px; 
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-goblin { 
        background: #7c2d12; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 4px; 
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Stats */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #374151;
    }
    
    .stat-label {
        color: #9ca3af;
        font-size: 0.875rem;
    }
    
    .stat-value {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Metrics */
    .metric-positive { color: #10b981; }
    .metric-negative { color: #ef4444; }
    .metric-neutral { color: #6b7280; }
    
    /* Tab styling for responsive design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        overflow-x: auto;
        padding: 0 4px;
        scrollbar-width: thin;
        scrollbar-color: #374151 #1f2937;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 6px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #1f2937;
        border-radius: 3px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 3px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        min-width: 55px;
        max-width: 100px;
        padding: 6px 10px;
        font-size: 0.82rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border-radius: 6px;
        background-color: #1f2937;
        border: 1px solid #374151;
        color: #9ca3af;
        transition: all 0.2s ease;
        flex-shrink: 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2d3748;
        color: #ffffff;
        border-color: #4b5563;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        border-color: #8b5cf6 !important;
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.2);
    }
    
    /* Mobile responsive tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            min-width: 45px;
            max-width: 70px;
            padding: 4px 6px;
            font-size: 0.75rem;
            height: 32px;
        }
    }
    
    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            min-width: 40px;
            max-width: 60px;
            padding: 3px 5px;
            font-size: 0.7rem;
            height: 30px;
        }
    }
    
    /* Expander styling for grouped layout */
    .streamlit-expanderHeader {
        background-color: #1f2937;
        border-radius: 8px;
        border: 1px solid #374151;
        padding: 0.75rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #2d3748;
        border-color: #4b5563;
    }
    
    .streamlit-expanderContent {
        background-color: #111827;
        border: 1px solid #374151;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render clean app header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; color: #ffffff; font-size: 2rem;">
            🎯 BetFinder AI
        </h1>
        <p style="margin: 0.5rem 0 0 0; color: #a78bfa; font-size: 1rem;">
            Professional Sports Betting Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_pick_card(pick, sport_emoji="🏈"):
    """Render individual pick as clean card with visual badges"""
    
    # Extract pick data
    player_name = pick.get('player_name', 'Unknown')
    stat_type = pick.get('stat_type', '')
    line = pick.get('line', 0)
    over_under = pick.get('over_under', '')
    confidence = pick.get('confidence', 0)
    expected_value = pick.get('expected_value', 0)
    odds = pick.get('odds', -110)
    matchup = pick.get('matchup', '')
    
    # Get PrizePicks classification
    prizepicks_class = pick.get('prizepicks_classification', '')
    if isinstance(prizepicks_class, dict):
        classification = prizepicks_class.get('classification', '')
    else:
        classification = str(prizepicks_class) if prizepicks_class else ''
    
    # Clean classification text
    classification = classification.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    
    # Use Streamlit columns for clean layout
    with st.container():
        # Header row with player and classification
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown(f"<div style='font-size: 1.2rem; text-align: center;'>{sport_emoji}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{player_name}**")
        
        with col3:
            # Visual badge using Streamlit
            if "DEMON" in classification.upper():
                st.markdown(f"<span style='background: #dc2626; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;'>👹 DEMON</span>", unsafe_allow_html=True)
            elif "DISCOUNT" in classification.upper():
                st.markdown(f"<span style='background: #059669; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;'>💰 DISCOUNT</span>", unsafe_allow_html=True)
            elif "DECENT" in classification.upper():
                st.markdown(f"<span style='background: #0891b2; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;'>✅ DECENT</span>", unsafe_allow_html=True)
            elif "GOBLIN" in classification.upper():
                st.markdown(f"<span style='background: #7c2d12; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;'>👺 GOBLIN</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='background: #6b7280; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;'>🎯 PICK</span>", unsafe_allow_html=True)
        
        # Bet details
        st.markdown(f"**{over_under} {line} {stat_type}** • {matchup}")
        
        # Metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            conf_color = "#10b981" if confidence >= 75 else "#6b7280"
            st.markdown(f"<div style='text-align: center;'><div style='color: #9ca3af; font-size: 0.8rem;'>Confidence</div><div style='color: {conf_color}; font-weight: 600;'>{confidence:.0f}%</div></div>", unsafe_allow_html=True)
            
        with metric_col2:
            ev_color = "#10b981" if expected_value > 0 else "#ef4444"
            ev_sign = "+" if expected_value > 0 else ""
            st.markdown(f"<div style='text-align: center;'><div style='color: #9ca3af; font-size: 0.8rem;'>Expected Value</div><div style='color: {ev_color}; font-weight: 600;'>{ev_sign}{expected_value:.1f}%</div></div>", unsafe_allow_html=True)
            
        with metric_col3:
            st.markdown(f"<div style='text-align: center;'><div style='color: #9ca3af; font-size: 0.8rem;'>Odds</div><div style='color: #1a73e8; font-weight: 600;'>{odds}</div></div>", unsafe_allow_html=True)
        
        # Add separator
        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #374151;'>", unsafe_allow_html=True)

def render_sport_section(sport_name, picks, sport_emoji="🏈"):
    """Render sport section with picks using clean Streamlit components"""
    
    if not picks:
        return
        
    # Section header
    st.markdown(f"### {sport_emoji} {sport_name} ({len(picks)} picks)")
    
    # Container for picks
    with st.container():
        # Render each pick
        for i, pick in enumerate(picks):
            render_pick_card(pick, sport_emoji)
            
            # Add spacing between picks except for the last one
            if i < len(picks) - 1:
                st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

def render_stats_sidebar():
    """Render statistics in sidebar"""
    with st.sidebar:
        st.markdown("### 📊 Today's Stats")
        
        # Mock stats for demo
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Picks", "47", "+12")
            st.metric("Win Rate", "68.2%", "+2.1%")
            
        with col2:
            st.metric("Avg Confidence", "74%", "+1%")
            st.metric("Expected ROI", "+15.3%", "+3.2%")

def render_live_data_status():
    """Show live data status"""
    eastern = ZoneInfo("America/New_York")
    current_time = datetime.now(eastern)
    
    st.markdown(f"""
    <div style="background: #065f46; border-radius: 6px; padding: 0.75rem; margin: 1rem 0; border: 1px solid #059669;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #10b981; font-size: 1rem;">🟢</span>
            <span style="color: #ffffff; font-weight: 600;">Live Data Active</span>
            <span style="color: #86efac; font-size: 0.875rem;">Last updated: {current_time.strftime('%I:%M %p ET')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sport emoji mapping
SPORT_EMOJIS = {
    'basketball': '🏀',
    'football': '🏈', 
    'college_football': '🏈',
    'tennis': '🎾',
    'baseball': '⚾',
    'hockey': '🏒',
    'soccer': '⚽',
    'csgo': '🎮',
    'league_of_legends': '🎮',
    'dota2': '🎮',
    'valorant': '🎮',
    'overwatch': '🎮',
    'rocket_league': '🚀',
    'golf': '⛳'
}

def get_sport_emoji(sport_key):
    """Get emoji for sport"""
    return SPORT_EMOJIS.get(sport_key, '🎯')