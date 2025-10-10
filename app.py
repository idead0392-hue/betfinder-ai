import streamlit as st
import pandas as pd
import requests
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# requests is used for custom API calls below
from lxml import etree

# Page configuration
st.set_page_config(
    page_title="BetFinder AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
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

def load_api_data(url, cache_key, method='GET', data=None):
    """Load data from API with caching"""
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached
    
    try:
        if method == 'POST':
            response = requests.post(url, json=data or {}, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            set_cached_data(cache_key, result)
            return result
    except:
        pass
    
    return None

# Sidebar - Bankroll Management
st.sidebar.header("💰 Bankroll Management")

# Initialize modules for sidebar
try:
    from picks_engine import PicksEngine
    from bankroll_manager import BankrollManager
    
    # Initialize systems
    picks_engine = PicksEngine()
    bankroll_manager = BankrollManager()
    
    # Initialize bankroll if needed
    if 'bankroll_initialized' not in st.session_state:
        st.session_state.bankroll_initialized = False
    
    if not st.session_state.bankroll_initialized:
        st.sidebar.info("💡 **First time setup:** Initialize your bankroll to start tracking")
        
        starting_bankroll = st.sidebar.number_input("Starting Bankroll ($)", 
                                              min_value=100.0, max_value=100000.0, 
                                              value=1000.0, step=50.0)
        unit_size = st.sidebar.number_input("Unit Size ($)", 
                                      min_value=1.0, max_value=500.0, 
                                      value=starting_bankroll * 0.01, step=1.0)
        
        if st.sidebar.button("🚀 Initialize Bankroll", type="primary"):
            bankroll_manager.initialize_bankroll(starting_bankroll, unit_size)
            st.session_state.bankroll_initialized = True
            st.sidebar.success(f"✅ Bankroll initialized: ${starting_bankroll:,.2f} | Unit: ${unit_size:.2f}")
            st.balloons()
            # Note: Removed st.rerun() to prevent tab redirect after initialization
    else:
        # Display current bankroll status
        performance = bankroll_manager.get_performance_metrics()
        risk_assessment = bankroll_manager.get_risk_assessment()
        
        # Bankroll overview (compact sidebar version)
        current_bankroll = performance.get('current_bankroll', 1000)
        bankroll_change = performance.get('bankroll_change', 0)
        roi = performance.get('roi_percentage', 0)
        win_rate = performance.get('win_rate', 0)
        risk_level = risk_assessment['risk_level']
        
        st.sidebar.metric("Current Bankroll", f"${current_bankroll:,.2f}", 
                 f"${bankroll_change:+.2f}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("ROI", f"{roi:.1f}%")
        with col2:
            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric("Risk", f"{risk_color.get(risk_level, '⚪')} {risk_level}")
        
        st.sidebar.metric("Win Rate", f"{win_rate:.1f}%", 
                 f"{performance.get('total_bets', 0)} bets")
        
        # Risk recommendations
        if risk_assessment['recommendations']:
            with st.sidebar.expander("🎯 Risk Recommendations", expanded=risk_assessment['risk_level'] == 'High'):
                for rec in risk_assessment['recommendations']:
                    st.markdown(f"• {rec}")
        
        # Daily risk tracking
        daily_risk_used = risk_assessment['daily_risk_used']
        daily_risk_limit = risk_assessment['daily_risk_limit']
        daily_risk_pct = risk_assessment['daily_risk_percentage']
        
        st.sidebar.progress(min(daily_risk_pct / 15, 1.0))  # 15% is high risk
        st.sidebar.caption(f"Daily Risk: ${daily_risk_used:.2f} / ${daily_risk_limit:.2f} ({daily_risk_pct:.1f}%)")
        
        # Quick actions (compact sidebar version)
        if st.sidebar.button("📊 View Bet History"):
            st.session_state.show_bet_history = True
        if st.sidebar.button("⚙️ Adjust Settings"):
            st.session_state.show_settings = True
        if st.sidebar.button("💾 Export Data"):
            filename = bankroll_manager.export_data("csv")
            st.sidebar.success(f"Data exported: {filename}")

except ImportError as e:
    st.sidebar.error(f"Failed to load bankroll system: {e}")

st.sidebar.markdown("---")

# Data visualization functions
def create_odds_trend_chart(odds_data, sport_name):
    """Create a trend chart for betting odds"""
    if not odds_data or not isinstance(odds_data, list):
        return None
    
    # Process odds data for visualization
    market_totals = {}
    for odds in odds_data:
        if isinstance(odds, dict) and 'marketIds' in odds:
            for market in odds['marketIds']:
                if isinstance(market, dict):
                    market_name = market.get('marketName', 'Unknown')
                    total_matched = market.get('totalMatched', 0)
                    if market_name not in market_totals:
                        market_totals[market_name] = []
                    market_totals[market_name].append(total_matched)
    
    if not market_totals:
        return None
    
    # Create bar chart
    fig = go.Figure()
    for market, totals in market_totals.items():
        avg_total = sum(totals) / len(totals) if totals else 0
        fig.add_trace(go.Bar(
            x=[market],
            y=[avg_total],
            name=market,
            text=f"${avg_total:,.0f}",
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f"{sport_name} Betting Market Activity",
        xaxis_title="Market Type",
        yaxis_title="Average Total Matched ($)",
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_competition_distribution_chart(competitions_data, sport_name):
    """Create a pie chart showing competition distribution"""
    if not competitions_data or not isinstance(competitions_data, list):
        return None
    
    # Count competitions by region
    region_counts = {}
    for comp in competitions_data:
        if isinstance(comp, dict):
            region = comp.get('competitionRegion', 'Unknown')
            region_counts[region] = region_counts.get(region, 0) + 1
    
    if not region_counts:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=list(region_counts.keys()),
        values=list(region_counts.values()),
        hole=0.3
    )])
    
    fig.update_layout(
        title=f"{sport_name} Competitions by Region",
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_market_activity_gauge(total_markets):
    """Create a gauge chart for market activity"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = total_markets,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Total Markets Available"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 200]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"},
                {'range': [100, 200], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150
            }
        }
    ))
    
    fig.update_layout(height=300, template="plotly_white")
    return fig


# Helper: safe rerun wrapper for Streamlit versions without experimental_rerun
def safe_rerun():
    """Try to rerun the Streamlit script. If not available, show a message and stop execution.

    This avoids AttributeError on Streamlit builds that don't expose experimental_rerun.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        # Fall through to graceful fallback
        pass

    # Graceful fallback: do nothing (allow the script to continue).
    # Many Streamlit versions will simply re-run on the next user interaction.
    return


def preload_sports_data():
    """Preload all sports data on app startup"""
    if st.session_state.data_loaded:
        return
    
    with st.spinner("Loading sports data..."):
        # Basketball data
        load_api_data("http://127.0.0.1:5001/api/basketball/props", "basketball_props")
        load_api_data("http://127.0.0.1:5001/api/basketball/odds", "basketball_odds", "POST")
        
        # Tennis data
        load_api_data("http://127.0.0.1:5001/api/tennis/competitions", "tennis_competitions")
        load_api_data("http://127.0.0.1:5001/api/tennis/odds", "tennis_odds", "POST")
        
        # Football data
        load_api_data("http://127.0.0.1:5001/api/football/competitions", "football_competitions")
        load_api_data("http://127.0.0.1:5001/api/football/odds", "football_odds", "POST")
        
        # Soccer data
        load_api_data("http://127.0.0.1:5001/api/soccer/competitions", "soccer_competitions")
        
        # Baseball data
        load_api_data("http://127.0.0.1:5001/api/baseball/competitions", "baseball_competitions")
        load_api_data("http://127.0.0.1:5001/api/baseball/odds", "baseball_odds", "POST")
        
        # Hockey data
        load_api_data("http://127.0.0.1:5001/api/hockey/competitions", "hockey_competitions")
        load_api_data("http://127.0.0.1:5001/api/hockey/odds", "hockey_odds", "POST")
        
        # Esports data
        load_api_data("http://127.0.0.1:5001/api/esports/competitions", "esports_competitions")
        load_api_data("http://127.0.0.1:5001/api/esports/odds", "esports_odds", "POST")
        
        # College Football data
        load_api_data("http://127.0.0.1:5001/api/college-football/competitions", "college_football_competitions")
        load_api_data("http://127.0.0.1:5001/api/college-football/odds", "college_football_odds", "POST")
        
        st.session_state.data_loaded = True

# Preload data on app start
preload_sports_data()

# Add refresh functionality
def refresh_all_data():
    """Clear cache and reload all sports data"""
    st.session_state.data_cache = {}
    st.session_state.cache_timestamp = {}
    st.session_state.data_loaded = False
    preload_sports_data()
    st.rerun()

# Header with enhanced controls
header_col1, header_col2, header_col3 = st.columns([3, 1, 1])

with header_col1:
    st.title("🎯 BetFinder AI")

with header_col2:
    # Theme toggle (starts in dark mode by default)
    if 'dark_theme' not in st.session_state:
        st.session_state.dark_theme = True  # Start in dark mode by default
    
    theme_label = "🌙 Dark" if not st.session_state.dark_theme else "☀️ Light"
    if st.button(theme_label, help="Toggle theme"):
        st.session_state.dark_theme = not st.session_state.dark_theme
        st.rerun()

with header_col3:
    if st.button("🔄 Refresh Data", help="Update all sports data"):
        refresh_all_data()

# Apply theme styling
if st.session_state.dark_theme:
    st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stExpander {
        background-color: #2d2d2d;
        border: 1px solid #404040;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)


# --- RESTORED MULTI-TAB UI ---
tab_names = [
    "Home", "Daily Picks", "Stats", "Props", "Tennis", "Basketball", "Football", "Baseball",
    "Hockey", "Soccer", "Esports", "College Football"
]
tabs = st.tabs(tab_names)


# --- Session-based Yahoo OAuth Token Storage (Streamlit Cloud compatible) ---
if 'yahoo_access_token' not in st.session_state:
    st.session_state['yahoo_access_token'] = None
    st.session_state['yahoo_refresh_token'] = None
    st.session_state['yahoo_oauth_state'] = None
    st.session_state['yahoo_oauth_step'] = 0





# Home Tab
with tabs[0]:
    st.header("🎯 Welcome to BetFinder AI")
    st.markdown("""
    **Your comprehensive sports betting analysis platform**
    
    Navigate through the tabs above to:
    - 🤖 **Daily Picks** - Get AI-generated betting recommendations 
    - 📊 **Stats** - View detailed statistics and analytics
    - 🎮 **Props** - Access player prop data and predictions
    - 🎾 **Sports Tabs** - Live odds and matchups for all major sports
    
    All data is sourced from professional APIs and updated in real-time.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sports", "8", "All major leagues")
    with col2:
        st.metric("Live Markets", "500+", "Real-time odds")
    with col3:
        st.metric("Data Sources", "Multiple", "Professional APIs")

# Daily Picks Tab
with tabs[1]:
    st.header("🤖 Daily AI Picks")
    st.markdown("**AI-generated betting recommendations with professional analysis**")
    
    # Import required modules
    try:
        from picks_engine import PicksEngine
        from bankroll_manager import BankrollManager
        
        # Initialize systems (reuse bankroll_manager from sidebar)
        picks_engine = PicksEngine()
        if 'bankroll_manager' not in locals():
            bankroll_manager = BankrollManager()
        
        # AI Picks Section - Always Available for Analysis
        st.subheader("🎯 Today's AI Picks")
        
        # Picks controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            max_picks = st.slider("Number of picks to generate", 1, 15, 8)
        
        with col2:
            min_confidence = st.slider("Min confidence %", 50, 90, 60)
        
        with col3:
            if st.button("🔄 Generate New Picks", type="primary"):
                st.session_state.picks_generated = False
        
        # Generate picks
        if 'picks_generated' not in st.session_state or not st.session_state.picks_generated:
            with st.spinner("🤖 AI analyzing markets and generating picks..."):
                daily_picks = picks_engine.get_daily_picks(max_picks)
                # Filter by confidence
                daily_picks = [pick for pick in daily_picks if pick['confidence'] >= min_confidence]
                
                # Add bankroll-adjusted bet sizing ONLY if bankroll is initialized
                if st.session_state.get('bankroll_initialized', False):
                    for pick in daily_picks:
                        recommended_bet, bet_reason = bankroll_manager.calculate_recommended_bet_size(
                            pick['confidence'], pick['expected_value']
                        )
                        pick['recommended_bet'] = recommended_bet
                        pick['bet_reason'] = bet_reason
                        pick['can_bet'] = recommended_bet > 0
                
                st.session_state.daily_picks = daily_picks
                st.session_state.picks_generated = True
        else:
            daily_picks = st.session_state.daily_picks
    
        if daily_picks:
            # Show picks summary
            summary = picks_engine.get_pick_summary(daily_picks)
            
            # Conditional metrics based on bankroll initialization
            if st.session_state.get('bankroll_initialized', False):
                # Enhanced summary with bankroll integration
                bettable_picks = [pick for pick in daily_picks if pick.get('can_bet', False)]
                total_potential_bet = sum(pick.get('recommended_bet', 0) for pick in bettable_picks)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Picks", len(daily_picks))
                with col2:
                    st.metric("Bettable Picks", len(bettable_picks))
                with col3:
                    st.metric("Avg Confidence", f"{summary['avg_confidence']}%")
                with col4:
                    st.metric("Total Risk", f"${total_potential_bet:.2f}")
                with col5:
                    potential_payout = sum([
                        bankroll_manager.calculate_payout(pick.get('recommended_bet', 0), pick['odds']) 
                        for pick in bettable_picks
                    ])
                    potential_profit = potential_payout - total_potential_bet
                    st.metric("Potential Profit", f"${potential_profit:.2f}")
            else:
                # Analysis-only summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Picks", len(daily_picks))
                with col2:
                    st.metric("Avg Confidence", f"{summary['avg_confidence']}%")
                with col3:
                    st.metric("Avg Expected Value", f"+{summary['avg_expected_value']}%")
                with col4:
                    st.metric("Sports Covered", summary['sports_covered'])
            
            st.markdown("---")
            
            # Display picks as cards in a horizontal layout
            if len(daily_picks) <= 4:
                # Single row for 4 or fewer picks
                cols = st.columns(len(daily_picks))
            else:
                # Multiple rows for more than 4 picks
                cols = st.columns(4)
            
            for i, pick in enumerate(daily_picks, 1):
                can_bet = pick.get('can_bet', False)
                recommended_bet = pick.get('recommended_bet', 0)
                
                # Calculate which column to use (wrap to new row after 4 cards)
                col_index = (i - 1) % 4
                
                with cols[col_index]:
                    # Enhanced visual indicators
                    confidence_emoji = "🔥" if pick['confidence'] >= 80 else "⚡" if pick['confidence'] >= 70 else "📈"
                    player_prop_indicator = " 👤" if pick.get('pick_type') in ['player_prop', 'player props'] else ""
                    
                    # Color-coded confidence
                    if pick['confidence'] >= 80:
                        confidence_color = "#28a745"  # Green
                        border_color = "#28a745"
                    elif pick['confidence'] >= 70:
                        confidence_color = "#ffc107"  # Yellow
                        border_color = "#ffc107"
                    else:
                        confidence_color = "#dc3545"  # Red
                        border_color = "#dc3545"
                    
                    # Card container with custom styling
                    st.markdown(f"""
                    <div style="
                        border: 2px solid {border_color};
                        border-radius: 10px;
                        padding: 15px;
                        margin: 5px 0;
                        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        height: 300px;
                        overflow-y: auto;
                    ">
                    <h4 style="color: {confidence_color}; margin: 0 0 10px 0; font-size: 16px;">
                        {confidence_emoji} Pick #{i}{player_prop_indicator}
                    </h4>
                    <p style="margin: 5px 0; font-size: 14px; font-weight: bold;">{pick['matchup']}</p>
                    <p style="margin: 5px 0; font-size: 12px;">{pick['sport'].title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Pick details
                    st.markdown(f"**🎯 {pick['pick']}**")
                    st.caption(f"💰 Odds: {pick['odds']:+d}")
                    st.caption(f"⚡ Confidence: {pick['confidence']}%")
                    st.caption(f"📊 EV: +{pick['expected_value']}%")
                    st.caption(f"📅 {pick['start_time']}")
                    
                    # Player prop specific details
                    if pick.get('pick_type') == 'player_prop' and pick.get('player_name'):
                        st.caption(f"👤 {pick['player_name']}")
                    
                    # Betting information
                    if st.session_state.get('bankroll_initialized', False):
                        if can_bet:
                            st.success(f"💵 ${recommended_bet:.2f}")
                            
                            # Risk assessment
                            risk_pct = (recommended_bet / bankroll_manager.data["bankroll"]) * 100
                            if risk_pct > 5:
                                st.caption("⚠️ 🔴 High Risk")
                            elif risk_pct > 2:
                                st.caption("⚠️ 🟡 Medium Risk")
                            else:
                                st.caption("⚠️ � Low Risk")
                            
                            # Unit sizing
                            units = recommended_bet / bankroll_manager.data["unit_size"]
                            st.caption(f"📐 {units:.1f}u")
                            
                            # Potential profit
                            potential_payout = bankroll_manager.calculate_payout(recommended_bet, pick['odds'])
                            potential_profit = potential_payout - recommended_bet
                            st.caption(f"� +${potential_profit:.2f}")
                            
                            # Bet button
                            bet_key = f"place_bet_{pick['game_id']}_{i}"
                            if st.button("� Bet", key=bet_key, type="primary", use_container_width=True):
                                success = bankroll_manager.add_bet(
                                    pick_id=pick['game_id'],
                                    sport=pick['sport'],
                                    matchup=pick['matchup'],
                                    pick_type=pick['pick_type'],
                                    odds=pick['odds'],
                                    bet_amount=recommended_bet,
                                    confidence=pick['confidence'],
                                    expected_value=pick['expected_value']
                                )
                                
                                if success:
                                    st.success("✅ Tracked!")
                                    st.balloons()
                                else:
                                    st.error("❌ Failed")
                        else:
                            st.warning("❌ Cannot Bet")
                            st.caption(pick['bet_reason'])
                    else:
                        st.info("💡 Analysis Mode")
                        st.caption("Enable bankroll for betting")
                    
                    # Expandable details
                    with st.expander("📋 Details", expanded=False):
                        st.markdown(f"**🧠 AI Analysis:**")
                        st.markdown(pick['reasoning'])
                        
                        # Market analysis
                        if pick['market_analysis']:
                            market = pick['market_analysis']
                            st.markdown("**📊 Market Intelligence:**")
                            st.caption(f"Line: {market['line_movement'].title()}")
                            st.caption(f"Public: {market['public_bias'].title()}")
                            st.caption(f"Sharp: {'Yes' if market['sharp_action'] else 'No'}")
                            st.caption(f"Reverse: {'Yes' if market['reverse_line_movement'] else 'No'}")
            
            # Create additional rows if more than 4 picks
            if len(daily_picks) > 4:
                remaining_picks = daily_picks[4:]
                for row_start in range(0, len(remaining_picks), 4):
                    row_picks = remaining_picks[row_start:row_start + 4]
                    cols = st.columns(len(row_picks))
                    
                    for j, pick in enumerate(row_picks):
                        pick_index = 5 + row_start + j  # Continue numbering from first row
                        
                        with cols[j]:
                            can_bet = pick.get('can_bet', False)
                            recommended_bet = pick.get('recommended_bet', 0)
                            
                            # Same card styling
                            confidence_emoji = "🔥" if pick['confidence'] >= 80 else "⚡" if pick['confidence'] >= 70 else "📈"
                            player_prop_indicator = " 👤" if pick.get('pick_type') in ['player_prop', 'player props'] else ""
                            
                            if pick['confidence'] >= 80:
                                confidence_color = "#28a745"
                                border_color = "#28a745"
                            elif pick['confidence'] >= 70:
                                confidence_color = "#ffc107"
                                border_color = "#ffc107"
                            else:
                                confidence_color = "#dc3545"
                                border_color = "#dc3545"
                            
                            st.markdown(f"""
                            <div style="
                                border: 2px solid {border_color};
                                border-radius: 10px;
                                padding: 15px;
                                margin: 5px 0;
                                background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                height: 300px;
                                overflow-y: auto;
                            ">
                            <h4 style="color: {confidence_color}; margin: 0 0 10px 0; font-size: 16px;">
                                {confidence_emoji} Pick #{pick_index}{player_prop_indicator}
                            </h4>
                            <p style="margin: 5px 0; font-size: 14px; font-weight: bold;">{pick['matchup']}</p>
                            <p style="margin: 5px 0; font-size: 12px;">{pick['sport'].title()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Same content as first row cards
                            st.markdown(f"**🎯 {pick['pick']}**")
                            st.caption(f"💰 Odds: {pick['odds']:+d}")
                            st.caption(f"⚡ Confidence: {pick['confidence']}%")
                            st.caption(f"📊 EV: +{pick['expected_value']}%")
                            st.caption(f"📅 {pick['start_time']}")
                            
                            if pick.get('pick_type') == 'player_prop' and pick.get('player_name'):
                                st.caption(f"👤 {pick['player_name']}")
                            
                            if st.session_state.get('bankroll_initialized', False):
                                if can_bet:
                                    st.success(f"💵 ${recommended_bet:.2f}")
                                    risk_pct = (recommended_bet / bankroll_manager.data["bankroll"]) * 100
                                    
                                    if risk_pct > 5:
                                        st.caption("⚠️ 🔴 High Risk")
                                    elif risk_pct > 2:
                                        st.caption("⚠️ 🟡 Medium Risk")
                                    else:
                                        st.caption("⚠️ � Low Risk")
                                    
                                    units = recommended_bet / bankroll_manager.data["unit_size"]
                                    st.caption(f"📐 {units:.1f}u")
                                    
                                    potential_payout = bankroll_manager.calculate_payout(recommended_bet, pick['odds'])
                                    potential_profit = potential_payout - recommended_bet
                                    st.caption(f"📈 +${potential_profit:.2f}")
                                    
                                    bet_key = f"place_bet_{pick['game_id']}_{pick_index}"
                                    if st.button("📝 Bet", key=bet_key, type="primary", use_container_width=True):
                                        success = bankroll_manager.add_bet(
                                            pick_id=pick['game_id'],
                                            sport=pick['sport'],
                                            matchup=pick['matchup'],
                                            pick_type=pick['pick_type'],
                                            odds=pick['odds'],
                                            bet_amount=recommended_bet,
                                            confidence=pick['confidence'],
                                            expected_value=pick['expected_value']
                                        )
                                        
                                        if success:
                                            st.success("✅ Tracked!")
                                            st.balloons()
                                        else:
                                            st.error("❌ Failed")
                                else:
                                    st.warning("❌ Cannot Bet")
                                    st.caption(pick['bet_reason'])
                            else:
                                st.info("💡 Analysis Mode")
                                st.caption("Enable bankroll for betting")
                            
                            with st.expander("📋 Details", expanded=False):
                                st.markdown(f"**🧠 AI Analysis:**")
                                st.markdown(pick['reasoning'])
                                
                                if pick['market_analysis']:
                                    market = pick['market_analysis']
                                    st.markdown("**📊 Market Intelligence:**")
                                    st.caption(f"Line: {market['line_movement'].title()}")
                                    st.caption(f"Public: {market['public_bias'].title()}")
                                    st.caption(f"Sharp: {'Yes' if market['sharp_action'] else 'No'}")
                                    st.caption(f"Reverse: {'Yes' if market['reverse_line_movement'] else 'No'}")
                
                # Summary action buttons - conditional on bankroll initialization
                if st.session_state.get('bankroll_initialized', False):
                    st.markdown("---")
                    col_x, col_y, col_z = st.columns(3)
                    
                    # Get bettable picks for this section
                    bettable_picks = [pick for pick in daily_picks if pick.get('can_bet', False)]
                    
                    with col_x:
                        if bettable_picks and st.button("📊 Place All Recommended Bets", type="secondary"):
                            placed_count = 0
                            for pick in bettable_picks:
                                success = bankroll_manager.add_bet(
                                    pick_id=pick['game_id'],
                                    sport=pick['sport'],
                                    matchup=pick['matchup'],
                                    pick_type=pick['pick_type'],
                                    odds=pick['odds'],
                                    bet_amount=pick['recommended_bet'],
                                    confidence=pick['confidence'],
                                    expected_value=pick['expected_value']
                                )
                                if success:
                                    placed_count += 1
                            
                            if placed_count > 0:
                                st.success(f"✅ Tracked {placed_count} bets!")
                                st.balloons()
                                # Removed st.rerun() to prevent tab redirect
                
                    with col_y:
                        if st.button("⚠️ Skip All Picks Today", type="secondary"):
                            st.session_state.picks_skipped = True
                            st.info("All picks skipped for today")
                
                    with col_z:
                        if st.button("🔄 Refresh Risk Assessment", type="secondary"):
                            # Force refresh of risk calculations
                            if 'daily_picks' in st.session_state:
                                # Clear cached picks to force regeneration
                                st.session_state.picks_generated = False
                            # Note: Removed st.rerun() to prevent tab redirect
                
                # Responsible gambling reminder
                st.markdown("---")
                st.warning("⚠️ **Responsible Gambling & Risk Management**")
                col_warn1, col_warn2 = st.columns(2)
                with col_warn1:
                    st.markdown("""
                    **Bankroll Protection:**
                    - All bet sizes calculated using Kelly Criterion
                    - Daily risk limits enforced automatically
                    - Stop-loss triggers at 20% drawdown
                    - Maximum 5% of bankroll per single bet
                    """)
                with col_warn2:
                    st.markdown("""
                    **Remember:**
                    - These are AI suggestions, not guarantees
                    - Past performance doesn't predict future results
                    - Only bet what you can afford to lose
                    - Seek help if gambling becomes a problem
                    """)
            
            else:
                st.info(f"🤖 No picks found meeting your criteria (min {min_confidence}% confidence)")
                st.markdown("""
                **Possible reasons:**
                - Market conditions don't meet our quality thresholds
                - All current picks fall below your confidence minimum
                - Daily risk limits have been reached
                - Limited live games available right now
                
                Try lowering the minimum confidence or check back later for new opportunities.
                """)
        
        else:
            st.info("💡 Initialize your bankroll above to start using AI picks with professional money management")
        
        # Bet History Modal
        if st.session_state.get('show_bet_history', False):
            st.markdown("---")
            st.subheader("📊 Betting History & Performance")
            
            bet_df = bankroll_manager.get_bet_history_dataframe()
            if not bet_df.empty:
                # Performance summary
                recent_7_days = bankroll_manager.get_recent_performance(7)
                recent_30_days = bankroll_manager.get_recent_performance(30)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("7-Day ROI", f"{recent_7_days['roi']:.1f}%")
                with col2:
                    st.metric("7-Day Win Rate", f"{recent_7_days['win_rate']:.1f}%")
                with col3:
                    st.metric("30-Day ROI", f"{recent_30_days['roi']:.1f}%")
                with col4:
                    st.metric("30-Day Win Rate", f"{recent_30_days['win_rate']:.1f}%")
                
                # Bet history table
                st.dataframe(bet_df[['date', 'sport', 'matchup', 'pick_type', 'odds', 
                                   'bet_amount', 'confidence', 'status', 'profit', 'roi']], 
                           width='stretch')
                
                # Settle pending bets
                pending_bets = bet_df[bet_df['status'] == 'pending']
                if not pending_bets.empty:
                    st.subheader("⏳ Settle Pending Bets")
                    for _, bet in pending_bets.iterrows():
                        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
                        with col_a:
                            st.write(f"{bet['matchup']} - {bet['pick_type']}")
                        with col_b:
                            if st.button("✅ Won", key=f"won_{bet['id']}"):
                                bankroll_manager.settle_bet(bet['id'], 'won')
                                st.success("Bet settled as Won!")
                        with col_c:
                            if st.button("❌ Lost", key=f"lost_{bet['id']}"):
                                bankroll_manager.settle_bet(bet['id'], 'lost')
                                st.error("Bet settled as Lost")
                        with col_d:
                            if st.button("➖ Push", key=f"push_{bet['id']}"):
                                bankroll_manager.settle_bet(bet['id'], 'pushed')
                                st.info("Bet settled as Push")
            else:
                st.info("No betting history yet. Start placing bets to see your performance!")
            
            if st.button("❌ Close History"):
                st.session_state.show_bet_history = False
                # Note: Removed st.rerun() to prevent tab redirect
        
        # Settings Modal
        if st.session_state.get('show_settings', False):
            st.markdown("---")
            st.subheader("⚙️ Bankroll Settings")
            
            current_settings = bankroll_manager.data['settings']
            
            col1, col2 = st.columns(2)
            with col1:
                new_max_daily_bets = st.number_input("Max Daily Bets", 
                                                   min_value=1, max_value=20, 
                                                   value=current_settings['max_daily_bets'])
                new_max_daily_risk = st.slider("Max Daily Risk %", 
                                              min_value=5.0, max_value=25.0, 
                                              value=current_settings['max_daily_risk'])
                new_conservative = st.checkbox("Conservative Mode", 
                                             value=current_settings['conservative_mode'])
            
            with col2:
                new_stop_loss = st.slider("Stop Loss %", 
                                        min_value=10.0, max_value=50.0, 
                                        value=current_settings['stop_loss_percentage'])
                new_profit_target = st.slider("Profit Target %", 
                                            min_value=20.0, max_value=200.0, 
                                            value=current_settings['profit_target_percentage'])
                new_max_bet_pct = st.slider("Max Bet % of Bankroll", 
                                          min_value=1.0, max_value=10.0, 
                                          value=bankroll_manager.data['max_bet_percentage'])
            
            if st.button("💾 Save Settings", type="primary"):
                bankroll_manager.data['settings'].update({
                    'max_daily_bets': new_max_daily_bets,
                    'max_daily_risk': new_max_daily_risk,
                    'conservative_mode': new_conservative,
                    'stop_loss_percentage': new_stop_loss,
                    'profit_target_percentage': new_profit_target
                })
                bankroll_manager.data['max_bet_percentage'] = new_max_bet_pct
                bankroll_manager.save_data()
                st.success("✅ Settings updated!")
                st.rerun()
            
            if st.button("❌ Close Settings"):
                st.session_state.show_settings = False
                st.rerun()
            
    except ImportError:
        st.error("❌ Picks engine not available")
        st.markdown("""
        The AI picks system requires the picks engine module.
        Please ensure all dependencies are properly installed.
        """)
    except Exception as e:
        st.error(f"❌ Error generating picks: {str(e)}")
        st.markdown("Please try refreshing the page or contact support if the issue persists.")

# Stats Tab
with tabs[2]:
    st.header("📊 Statistics & Analytics")
    st.markdown("**Comprehensive sports statistics and market analytics**")
    
    # API Statistics Section
    st.subheader("🔧 API Performance")
    try:
        api_stats_response = requests.get("http://127.0.0.1:5001/metrics", timeout=5)
        if api_stats_response.status_code == 200:
            api_stats = api_stats_response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                uptime = api_stats.get('uptime_seconds', 0)
                st.metric("API Uptime", f"{uptime/3600:.1f} hours")
            with col2:
                total_requests = api_stats.get('total_requests', 0)
                st.metric("Total Requests", f"{total_requests:,}")
            with col3:
                success_rate = api_stats.get('success_rate', 0)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col4:
                avg_response = api_stats.get('avg_response_time', 0)
                st.metric("Avg Response", f"{avg_response:.2f}s")
            
            # Endpoint statistics
            if 'endpoints' in api_stats:
                st.subheader("📈 Endpoint Performance")
                endpoints_data = []
                for endpoint, stats in api_stats['endpoints'].items():
                    endpoints_data.append({
                        'Endpoint': endpoint,
                        'Requests': stats.get('count', 0),
                        'Avg Time (s)': f"{stats.get('total_time', 0) / max(stats.get('count', 1), 1):.3f}",
                        'Errors': stats.get('errors', 0),
                        'Last Request': stats.get('last_request', 'Never')
                    })
                
                if endpoints_data:
                    import pandas as pd
                    df = pd.DataFrame(endpoints_data)
                    st.dataframe(df, width='stretch')
        else:
            st.warning("⚠️ Unable to connect to API metrics endpoint")
    except:
        st.error("❌ API metrics unavailable")
    
    # Sports Data Overview
    st.subheader("🏆 Sports Data Overview")
    
    # Count cached data
    cache_stats = {}
    for key in st.session_state.get('data_cache', {}):
        if '_competitions' in key:
            sport = key.replace('_competitions', '').replace('_', ' ').title()
            cache_stats[sport] = 'Loaded'
        elif '_odds' in key:
            sport = key.replace('_odds', '').replace('_', ' ').title()
            if sport in cache_stats:
                cache_stats[sport] += ' + Odds'
            else:
                cache_stats[sport] = 'Odds Only'
    
    if cache_stats:
        col1, col2 = st.columns(2)
        sports_list = list(cache_stats.items())
        mid_point = len(sports_list) // 2
        
        with col1:
            for sport, status in sports_list[:mid_point]:
                status_icon = "✅" if "Loaded" in status else "⚠️"
                st.write(f"{status_icon} **{sport}**: {status}")
        
        with col2:
            for sport, status in sports_list[mid_point:]:
                status_icon = "✅" if "Loaded" in status else "⚠️"
                st.write(f"{status_icon} **{sport}**: {status}")
    else:
        st.info("No sports data currently cached. Navigate to sports tabs to load data.")
    
    # Market Analysis
    st.subheader("💰 Market Analysis")
    
    # Get total markets across all sports
    total_markets = 0
    active_competitions = 0
    
    for cache_key, data in st.session_state.get('data_cache', {}).items():
        if 'competitions' in cache_key and isinstance(data, dict):
            competitions = data.get('data', [])
            for comp in competitions:
                if isinstance(comp, dict):
                    active_competitions += 1
                    total_markets += comp.get('marketCount', 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Competitions", active_competitions)
    with col2:
        st.metric("Total Markets", f"{total_markets:,}")
    with col3:
        avg_markets = total_markets / max(active_competitions, 1)
        st.metric("Avg Markets/Competition", f"{avg_markets:.1f}")
    
    # Data Freshness
    st.subheader("🕒 Data Freshness")
    cache_timestamps = st.session_state.get('cache_timestamp', {})
    
    if cache_timestamps:
        from datetime import datetime
        current_time = datetime.now()
        
        freshness_data = []
        for cache_key, timestamp in cache_timestamps.items():
            sport = cache_key.replace('_competitions', '').replace('_odds', '').replace('_matchups_', ' ').replace('_', ' ').title()
            age_seconds = (current_time - timestamp).total_seconds()
            age_minutes = age_seconds / 60
            
            if age_minutes < 5:
                freshness = "🟢 Fresh"
            elif age_minutes < 15:
                freshness = "🟡 Good"
            else:
                freshness = "🔴 Stale"
            
            freshness_data.append({
                'Data Source': sport,
                'Last Updated': f"{age_minutes:.1f} min ago",
                'Status': freshness
            })
        
        if freshness_data:
            df_fresh = pd.DataFrame(freshness_data)
            st.dataframe(df_fresh, width='stretch')
    else:
        st.info("No timestamp data available")

# Props Tab
with tabs[3]:
    st.header("Props")
    st.write("Use a custom API that accepts a Bearer token to provide props data.")
    st.markdown("**Security:** Do not commit your tokens. Add them to `.streamlit/secrets.toml` as `NEW_API_TOKEN` or paste temporarily below.")

    provider = st.selectbox("Provider", ["None", "Custom (Bearer token)", "PandaScore"])

    if provider == "Custom (Bearer token)":
        endpoint = st.text_input("API endpoint (full URL)", value="https://api.example.com/v1/props")
        use_secret = st.checkbox("Use token from st.secrets['NEW_API_TOKEN']", value=True)
        token = None
        if use_secret:
            token = st.secrets.get("NEW_API_TOKEN") if hasattr(st, 'secrets') else None
            if not token:
                st.warning("`NEW_API_TOKEN` not found in `st.secrets`. You can paste a token below for testing.")
        else:
            token = st.text_input("Paste token (will not be saved)", type="password")

        if st.button("Test API"):
            if not endpoint:
                st.error("Please provide an API endpoint to test.")
            elif not token:
                st.error("No token available. Add `NEW_API_TOKEN` to `.streamlit/secrets.toml` or paste a token.")
            else:
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                st.info(f"Calling {endpoint} with Bearer token (redacted in UI)")
                try:
                    resp = requests.get(endpoint, headers=headers, timeout=15)
                    st.write("Status:", resp.status_code)
                    ct = resp.headers.get('content-type','')
                    if 'application/json' in ct:
                        try:
                            st.json(resp.json())
                        except Exception:
                            st.code(resp.text)
                    else:
                        # Show text or XML
                        st.code(resp.text[:10000])
                except Exception as e:
                    st.error(f"Request failed: {e}")

    elif provider == "PandaScore":
        st.write("PandaScore API integration for esports data.")
        endpoint = st.selectbox("Endpoint", [
            "https://api.pandascore.co/videogames",
            "https://api.pandascore.co/matches",
            "https://api.pandascore.co/series",
            "https://api.pandascore.co/tournaments"
        ], index=0)
        use_secret = st.checkbox("Use token from st.secrets['NEW_API_TOKEN']", value=True)
        token = None
        if use_secret:
            token = st.secrets.get("NEW_API_TOKEN") if hasattr(st, 'secrets') else None
            if not token:
                st.warning("`NEW_API_TOKEN` not found in `st.secrets`. You can paste a token below for testing.")
        else:
            token = st.text_input("Paste token (will not be saved)", type="password")

        if st.button("Test PandaScore API"):
            if not token:
                st.error("No token available. Add `NEW_API_TOKEN` to `.streamlit/secrets.toml` or paste a token.")
            else:
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                st.info(f"Calling {endpoint} with Bearer token (redacted in UI)")
                try:
                    resp = requests.get(endpoint, headers=headers, timeout=15)
                    st.write("Status:", resp.status_code)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("API call successful!")
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                            st.dataframe(df)
                        else:
                            st.json(data)
                    else:
                        st.error(f"API returned status {resp.status_code}")
                        st.code(resp.text[:2000])
                except Exception as e:
                    st.error(f"Request failed: {e}")

        st.markdown("---")
        st.write("If you want me to wire a specific API (StatPal, RapidAPI, etc.), tell me the provider name and I will add a ready-to-use integration that reads the token from `st.secrets`.")
    else:
        st.info("Select a provider to configure props data source.")

with tabs[4]:
    # Initialize favorites system in session state
    if 'favorite_players' not in st.session_state:
        st.session_state.favorite_players = set()
    if 'favorite_tournaments' not in st.session_state:
        st.session_state.favorite_tournaments = set()
    
    # Use cached tennis data
    all_tennis_competitions = get_cached_data("tennis_competitions") or {"data": []}
    tennis_odds_data = get_cached_data("tennis_odds") or {"data": []}
    
    # Extract data arrays
    all_tennis_competitions = all_tennis_competitions.get('data', [])
    tennis_odds_data = tennis_odds_data.get('data', [])
    
    # Auto-load matchups for a specific tennis competition (with caching)
    def load_tennis_matchups(competition_id):
        cache_key = f"tennis_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/tennis/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific tennis game
    def find_tennis_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_tennis_competitions:
        # Create controls section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Tournament dropdown
            tournament_options = ["All Tournaments"] + [
                f"{comp['competition'].get('name', 'Unknown')} ({comp.get('competitionRegion', 'Int')})"
                for comp in all_tennis_competitions 
                if isinstance(comp, dict) and 'competition' in comp
            ]
            selected_tournament = st.selectbox(
                "🏆 Select Tournament",
                tournament_options,
                index=0
            )
        
        with col2:
            # Search bar for player names
            search_query = st.text_input(
                "🔍 Search Player",
                placeholder="Enter player name...",
                help="Search for matches by player name"
            )
        
        # Collect all matches from all tournaments for search
        all_matches = []
        tournaments_with_no_matches = []
        
        for comp in all_tennis_competitions:
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown Tournament')
                comp_region = comp.get('competitionRegion', 'International')
                markets_count = comp.get('marketCount', 0)
                
                # Load matchups for this competition
                comp_matchups = load_tennis_matchups(comp_info.get('id'))
                
                if comp_matchups:
                    for matchup in comp_matchups:
                        if isinstance(matchup, dict):
                            home_player = matchup.get('homeTeam', {}).get('name', 'Player 1')
                            away_player = matchup.get('awayTeam', {}).get('name', 'Player 2')
                            start_time = matchup.get('startTime', 'TBD')
                            game_id = matchup.get('bfid')
                            
                            match_data = {
                                'tournament': comp_name,
                                'region': comp_region,
                                'markets': markets_count,
                                'home_player': home_player,
                                'away_player': away_player,
                                'start_time': start_time,
                                'game_id': game_id,
                                'matchup': matchup
                            }
                            all_matches.append(match_data)
                else:
                    # Track tournaments with no matches
                    tournaments_with_no_matches.append({
                        'name': comp_name,
                        'region': comp_region,
                        'markets': markets_count
                    })
        
        # Show total matches and search suggestions
        if all_matches:
            st.info(f"📊 {len(all_matches)} total matches available across {len(all_tennis_competitions)} tournaments")
            
            # Show search suggestions if no search query
            if not search_query:
                sample_players = []
                for match in all_matches[:3]:  # Get first 3 matches
                    sample_players.extend([match['home_player'], match['away_player']])
                if sample_players:
                    st.caption(f"💡 Try searching for players like: {', '.join(sample_players[:3])}")
        else:
            # No live matches available
            st.warning("⚠️ No live tennis matches currently available")
            st.info("🎾 Tennis tournaments are scheduled but no active matches found. Check back during active tournament periods.")
        
        # Debug: Show available players for search
        if all_matches and search_query:
            all_players = []
            for match in all_matches:
                all_players.extend([match['home_player'], match['away_player']])
            unique_players = list(set(all_players))
            
            with st.expander(f"🔍 Debug: Available Players ({len(unique_players)} total)", expanded=False):
                col_debug1, col_debug2 = st.columns(2)
                mid_point = len(unique_players) // 2
                with col_debug1:
                    for player in unique_players[:mid_point]:
                        if search_query.lower() in player.lower():
                            st.success(f"✅ {player}")
                        else:
                            st.write(f"• {player}")
                with col_debug2:
                    for player in unique_players[mid_point:]:
                        if search_query.lower() in player.lower():
                            st.success(f"✅ {player}")
                        else:
                            st.write(f"• {player}")
        
        # Filter matches based on tournament selection and search query
        filtered_matches = all_matches
        
        # Filter by tournament
        if selected_tournament != "All Tournaments":
            tournament_name = selected_tournament.split(" (")[0]  # Extract tournament name without region
            filtered_matches = [m for m in filtered_matches if m['tournament'] == tournament_name]
        
        # Filter by player search
        if search_query:
            search_lower = search_query.lower()
            filtered_matches = [
                m for m in filtered_matches 
                if search_lower in m['home_player'].lower() or search_lower in m['away_player'].lower()
            ]
        
        # Debug information (can be removed later)
        if search_query:
            st.info(f"🔍 Searching for: '{search_query}' | Found {len(filtered_matches)} matches out of {len(all_matches)} total")
        
        # Display filtered matches
        if filtered_matches:
            st.write(f"� **{len(filtered_matches)} matches found**")
            
            # Group matches by tournament for better organization
            matches_by_tournament = {}
            for match in filtered_matches:
                tournament = match['tournament']
                if tournament not in matches_by_tournament:
                    matches_by_tournament[tournament] = []
                matches_by_tournament[tournament].append(match)
            
            # Display matches grouped by tournament
            for tournament, matches in matches_by_tournament.items():
                # Add tournament favorite button
                tourn_col1, tourn_col2 = st.columns([4, 1])
                with tourn_col1:
                    expander_title = f"🎾 {tournament} ({len(matches)} matches)"
                with tourn_col2:
                    is_fav_tournament = tournament in st.session_state.favorite_tournaments
                    if st.button("⭐" if is_fav_tournament else "☆", key=f"fav_tourn_{tournament}", 
                                help="Add/Remove from favorites"):
                        if is_fav_tournament:
                            st.session_state.favorite_tournaments.discard(tournament)
                        else:
                            st.session_state.favorite_tournaments.add(tournament)
                        st.rerun()
                
                with st.expander(expander_title, expanded=len(matches_by_tournament) <= 3):
                    for match in matches:
                        # Enhanced match display with favorites
                        col_a, col_b, col_c, col_d = st.columns([3, 3, 2, 1])
                        
                        with col_a:
                            st.markdown(f"**{match['away_player']}**")
                            st.caption("Player 1")
                            # Player 1 favorite button
                            is_fav1 = match['away_player'] in st.session_state.favorite_players
                            if st.button("⭐" if is_fav1 else "☆", key=f"fav1_{match['game_id']}_away", 
                                        help="Add/Remove from favorites"):
                                if is_fav1:
                                    st.session_state.favorite_players.discard(match['away_player'])
                                else:
                                    st.session_state.favorite_players.add(match['away_player'])
                                st.rerun()
                        
                        with col_b:
                            st.markdown(f"**{match['home_player']}**")
                            st.caption("Player 2")
                            # Player 2 favorite button
                            is_fav2 = match['home_player'] in st.session_state.favorite_players
                            if st.button("⭐" if is_fav2 else "☆", key=f"fav2_{match['game_id']}_home", 
                                        help="Add/Remove from favorites"):
                                if is_fav2:
                                    st.session_state.favorite_players.discard(match['home_player'])
                                else:
                                    st.session_state.favorite_players.add(match['home_player'])
                                st.rerun()
                        
                        with col_c:
                            st.caption(f"⏰ {match['start_time']}")
                            st.caption(f"🌍 {match['region']}")
                        
                        with col_d:
                            # Match quality indicator
                            has_favorites = (match['away_player'] in st.session_state.favorite_players or 
                                           match['home_player'] in st.session_state.favorite_players)
                            if has_favorites:
                                st.markdown("⭐")
                                st.caption("Favorite")
                            
                            st.caption(f"🏆 {match['markets']} markets")
                        
                        # Find odds for this match
                        match_odds = find_tennis_game_odds(match['game_id'], tennis_odds_data) if match['game_id'] else None
                        
                        if match_odds and 'marketIds' in match_odds:
                            st.markdown("**📊 Betting Markets:**")
                            for market in match_odds['marketIds'][:2]:  # Show max 2 markets to save space
                                if isinstance(market, dict):
                                    market_name = market.get('marketName', 'Match Winner')
                                    total_matched = market.get('totalMatched', 0)
                                    st.caption(f"💰 {market_name}: ${total_matched:,.2f}")
                        else:
                            st.caption("📊 Betting markets available")
                        
                        st.divider()
        
        else:
            if search_query:
                st.info(f"🔍 No matches found for player '{search_query}'" + 
                       (f" in {selected_tournament}" if selected_tournament != "All Tournaments" else ""))
                st.caption("Try a different search term or select 'All Tournaments'")
            else:
                # Show detailed information about tournament availability
                st.info("🎾 No live tennis matches currently scheduled")
                
                # Show available tournaments for context
                if tournaments_with_no_matches:
                    with st.expander("📋 Available Tournaments (No Current Matches)", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        
                        for i, tournament in enumerate(tournaments_with_no_matches):
                            col = [col_a, col_b, col_c][i % 3]
                            with col:
                                st.markdown(f"**{tournament['name']}**")
                                st.caption(f"📍 {tournament['region']}")
                                st.caption(f"📊 {tournament['markets']} markets")
                                st.markdown("---")
                
                st.markdown("""
                **Possible reasons:**
                - ⏰ Matches may be scheduled for different time zones
                - 🏆 Tournament may be between rounds or off-season
                - 📅 Check back during peak tennis season (Grand Slams, ATP/WTA tours)
                
                **Try these tournaments for active matches:**
                - 🎾 ATP Shanghai Masters (October)
                - 🎾 WTA Finals (October-November)
                - 🎾 Various Challenger events
                """)
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Tennis data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Tennis competitions data not available")
        st.info("Unable to load tennis competitions. Please check the API connection.")

with tabs[11]:
    st.markdown('<div class="section-title">Basketball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached basketball data
    all_competitions = get_cached_data("basketball_props") or {"data": []}
    odds_data = get_cached_data("basketball_odds") or {"data": []}
    
    # Extract data arrays
    all_competitions = all_competitions.get('data', [])
    odds_data = odds_data.get('data', [])
    
    # Auto-load matchups for a specific competition (with caching)
    def load_matchups(competition_id):
        cache_key = f"basketball_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/basketball/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Find NBA and WNBA competitions
    nba_comp = None
    wnba_comp = None
    
    for comp in all_competitions:
        if isinstance(comp, dict) and 'competition' in comp:
            comp_name = comp['competition'].get('name', '').upper()
            if comp_name == 'NBA':
                nba_comp = comp
            elif comp_name == 'WNBA':
                wnba_comp = comp
    
    # Helper function to find odds for a specific game
    def find_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    # Add data visualization section
    if all_competitions or odds_data:
        st.markdown("### 📊 Basketball Analytics Dashboard")
        
        # Create three columns for analytics
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            # Market activity gauge
            total_markets = sum(comp.get('marketCount', 0) for comp in all_competitions if isinstance(comp, dict))
            if total_markets > 0:
                gauge_fig = create_market_activity_gauge(total_markets)
                st.plotly_chart(gauge_fig, width='stretch')
        
        with viz_col2:
            # Competition distribution
            comp_chart = create_competition_distribution_chart(all_competitions, "Basketball")
            if comp_chart:
                st.plotly_chart(comp_chart, width='stretch')
        
        with viz_col3:
            # Odds trends
            odds_chart = create_odds_trend_chart(odds_data, "Basketball")
            if odds_chart:
                st.plotly_chart(odds_chart, width='stretch')
        
        st.markdown("---")  # Separator between analytics and games
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # NBA Column (Left)
    with col1:
        st.subheader("🏀 NBA")
        
        if nba_comp:
            comp_info = nba_comp['competition']
            markets_count = nba_comp.get('marketCount', 0)
            
            st.success(f"✅ {markets_count} betting markets available")
            
            # Load NBA matchups automatically
            nba_matchups = load_matchups(comp_info.get('id'))
            
            if nba_matchups:
                st.write(f"**{len(nba_matchups)} Upcoming Games:**")
                
                # Add a container for better organization when showing many games
                with st.container():
                    for i, matchup in enumerate(nba_matchups):  # Show all NBA games
                        if isinstance(matchup, dict):
                            home_team = matchup.get('homeTeam', {}).get('name', 'TBD')
                            away_team = matchup.get('awayTeam', {}).get('name', 'TBD')
                            start_time = matchup.get('startTime', 'TBD')
                            game_id = matchup.get('bfid')
                            
                            # Find odds for this game
                            game_odds = find_game_odds(game_id, odds_data) if game_id else None
                            
                            with st.expander(f"🏀 {away_team} @ {home_team}", expanded=False):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Away:** {away_team}")
                                    st.write(f"**Home:** {home_team}")
                                with col_b:
                                    st.write(f"**Time:** {start_time}")
                                    st.write(f"**Status:** {matchup.get('status', 'Scheduled')}")
                                
                                # Display odds if available
                                if game_odds and 'marketIds' in game_odds:
                                    st.markdown("**📊 Betting Odds:**")
                                    for market in game_odds['marketIds']:
                                        if isinstance(market, dict):
                                            market_name = market.get('marketName', 'Unknown')
                                            total_matched = market.get('totalMatched', 0)
                                            st.write(f"• **{market_name}**: ${total_matched:,.2f} total matched")
                                elif game_odds:
                                    st.info("📊 Odds data available - processing...")
                                else:
                                    st.caption("📊 No odds currently available")
                                
                                if 'markets' in matchup and matchup['markets']:
                                    st.write(f"**Markets:** {len(matchup['markets'])} available")
                
            else:
                st.info("📅 No games scheduled at this time")
                st.caption("Check back later for upcoming games or explore the betting markets available")
                
        else:
            st.warning("⚠️ NBA data not available")
            st.info("Unable to load NBA competition data")
    
    # WNBA Column (Right)
    with col2:
        st.subheader("🏀 WNBA")
        
        if wnba_comp:
            comp_info = wnba_comp['competition']
            markets_count = wnba_comp.get('marketCount', 0)
            
            st.success(f"✅ {markets_count} betting markets available")
            
            # Load WNBA matchups automatically
            wnba_matchups = load_matchups(comp_info.get('id'))
            
            if wnba_matchups:
                st.write(f"**{len(wnba_matchups)} Upcoming Games:**")
                
                # Add a container for better organization when showing many games
                with st.container():
                    for i, matchup in enumerate(wnba_matchups):  # Show all WNBA games
                        if isinstance(matchup, dict):
                            home_team = matchup.get('homeTeam', {}).get('name', 'TBD')
                            away_team = matchup.get('awayTeam', {}).get('name', 'TBD')
                            start_time = matchup.get('startTime', 'TBD')
                            game_id = matchup.get('bfid')
                            
                            # Find odds for this game
                            game_odds = find_game_odds(game_id, odds_data) if game_id else None
                            
                            with st.expander(f"🏀 {away_team} @ {home_team}", expanded=False):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Away:** {away_team}")
                                    st.write(f"**Home:** {home_team}")
                                with col_b:
                                    st.write(f"**Time:** {start_time}")
                                    st.write(f"**Status:** {matchup.get('status', 'Scheduled')}")
                                
                                # Display odds if available
                                if game_odds and 'marketIds' in game_odds:
                                    st.markdown("**📊 Betting Odds:**")
                                    for market in game_odds['marketIds']:
                                        if isinstance(market, dict):
                                            market_name = market.get('marketName', 'Unknown')
                                            total_matched = market.get('totalMatched', 0)
                                            st.write(f"• **{market_name}**: ${total_matched:,.2f} total matched")
                                elif game_odds:
                                    st.info("📊 Odds data available - processing...")
                                else:
                                    st.caption("📊 No odds currently available")
                                
                                if 'markets' in matchup and matchup['markets']:
                                    st.write(f"**Markets:** {len(matchup['markets'])} available")
                
            else:
                st.info("📅 No games scheduled at this time")
                st.caption("Check back later for upcoming games or explore the betting markets available")
                
        else:
            st.warning("⚠️ WNBA data not available")
            st.info("Unable to load WNBA competition data")
    
    # Footer
    st.markdown("---")
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    st.caption(f"Data automatically loaded from Sportbex API at {current_time} • Updates in real-time")

# FOOTBALL TAB
with tabs[6]:
    st.markdown('<div class="section-title">Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached football data
    all_football_competitions = get_cached_data("football_competitions") or {"data": []}
    football_odds_data = get_cached_data("football_odds") or {"data": []}
    
    # Extract data arrays
    all_football_competitions = all_football_competitions.get('data', [])
    football_odds_data = football_odds_data.get('data', [])
    
    # Auto-load matchups for a specific football competition (with caching)
    def load_football_matchups(competition_id):
        cache_key = f"football_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/football/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific football game
    def find_football_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_football_competitions:
        st.success(f"🏈 Found {len(all_football_competitions)} football competitions")
        
        # Display football competitions
        for i, comp in enumerate(all_football_competitions):
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown League')
                comp_region = comp.get('competitionRegion', 'USA')
                markets_count = comp.get('marketCount', 0)
                
                with st.expander(f"🏈 {comp_name}", expanded=i < 3):  # Expand first 3 leagues
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                    
                    # Load matchups for this competition
                    comp_matchups = load_football_matchups(comp_info.get('id'))
                    
                    if comp_matchups:
                        st.write(f"**{len(comp_matchups)} Upcoming Games:**")
                        
                        for matchup in comp_matchups:
                            if isinstance(matchup, dict):
                                home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                start_time = matchup.get('startTime', 'TBD')
                                game_id = matchup.get('bfid')
                                
                                # Find odds for this match
                                match_odds = find_football_game_odds(game_id, football_odds_data) if game_id else None
                                
                                with st.container():
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        st.markdown(f"**{away_team}**")
                                        st.caption("Away")
                                    with col_b:
                                        st.markdown(f"**{home_team}**")
                                        st.caption("Home")
                                    with col_c:
                                        st.caption(f"⏰ {start_time}")
                                    
                                    if match_odds and 'marketIds' in match_odds:
                                        st.markdown("**📊 Betting Markets:**")
                                        for market in match_odds['marketIds'][:3]:  # Show max 3 markets
                                            if isinstance(market, dict):
                                                market_name = market.get('marketName', 'Moneyline')
                                                total_matched = market.get('totalMatched', 0)
                                                st.caption(f"💰 {market_name}: ${total_matched:,.2f} matched")
                                    elif comp_matchups:
                                        st.caption("📊 Betting odds available")
                                    
                                    st.markdown("---")
                    else:
                        st.info("📅 No games scheduled currently")
                        st.caption("Check back during football season for live games")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Football data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Football competitions data not available")
        st.info("Football season may be off-season. Check back during NFL/College Football season.")

# BASEBALL TAB
with tabs[7]:
    st.markdown('<div class="section-title">Baseball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached baseball data
    all_baseball_competitions = get_cached_data("baseball_competitions") or {"data": []}
    baseball_odds_data = get_cached_data("baseball_odds") or {"data": []}
    
    # Extract data arrays
    all_baseball_competitions = all_baseball_competitions.get('data', [])
    baseball_odds_data = baseball_odds_data.get('data', [])
    
    # Auto-load matchups for a specific baseball competition (with caching)
    def load_baseball_matchups(competition_id):
        cache_key = f"baseball_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/baseball/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific baseball game
    def find_baseball_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_baseball_competitions:
        st.success(f"⚾ Found {len(all_baseball_competitions)} baseball competitions")
        
        # Display baseball competitions
        for i, comp in enumerate(all_baseball_competitions):
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown League')
                comp_region = comp.get('competitionRegion', 'USA')
                markets_count = comp.get('marketCount', 0)
                
                with st.expander(f"⚾ {comp_name}", expanded=i < 2):  # Expand first 2 leagues
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                    
                    # Load matchups for this competition
                    comp_matchups = load_baseball_matchups(comp_info.get('id'))
                    
                    if comp_matchups:
                        st.write(f"**{len(comp_matchups)} Upcoming Games:**")
                        
                        for matchup in comp_matchups:
                            if isinstance(matchup, dict):
                                home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                start_time = matchup.get('startTime', 'TBD')
                                game_id = matchup.get('bfid')
                                
                                # Find odds for this match
                                match_odds = find_baseball_game_odds(game_id, baseball_odds_data) if game_id else None
                                
                                with st.container():
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        st.markdown(f"**{away_team}**")
                                        st.caption("Away Team")
                                    with col_b:
                                        st.markdown(f"**{home_team}**")
                                        st.caption("Home Team")
                                    with col_c:
                                        st.caption(f"⏰ {start_time}")
                                    
                                    if match_odds and 'marketIds' in match_odds:
                                        st.markdown("**📊 Betting Markets:**")
                                        for market in match_odds['marketIds'][:3]:  # Show max 3 markets
                                            if isinstance(market, dict):
                                                market_name = market.get('marketName', 'Moneyline')
                                                total_matched = market.get('totalMatched', 0)
                                                st.caption(f"💰 {market_name}: ${total_matched:,.2f} matched")
                                    else:
                                        st.caption("📊 Betting odds available")
                                    
                                    st.markdown("---")
                    else:
                        st.info("📅 No games scheduled currently")
                        st.caption("Check back during baseball season for live games")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Baseball data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Baseball competitions data not available")
        st.info("Baseball season may be off-season. Check back during MLB season.")

# HOCKEY TAB
with tabs[8]:
    st.markdown('<div class="section-title">Hockey<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached hockey data
    all_hockey_competitions = get_cached_data("hockey_competitions") or {"data": []}
    hockey_odds_data = get_cached_data("hockey_odds") or {"data": []}
    
    # Extract data arrays
    all_hockey_competitions = all_hockey_competitions.get('data', [])
    hockey_odds_data = hockey_odds_data.get('data', [])
    
    # Auto-load matchups for a specific hockey competition (with caching)
    def load_hockey_matchups(competition_id):
        cache_key = f"hockey_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/hockey/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific hockey game
    def find_hockey_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_hockey_competitions:
        st.success(f"🏒 Found {len(all_hockey_competitions)} hockey competitions")
        
        # Display hockey competitions
        for i, comp in enumerate(all_hockey_competitions):
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown League')
                comp_region = comp.get('competitionRegion', 'International')
                markets_count = comp.get('marketCount', 0)
                
                with st.expander(f"🏒 {comp_name}", expanded=i < 2):  # Expand first 2 leagues
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                    
                    # Load matchups for this competition
                    comp_matchups = load_hockey_matchups(comp_info.get('id'))
                    
                    if comp_matchups:
                        st.write(f"**{len(comp_matchups)} Upcoming Games:**")
                        
                        for matchup in comp_matchups:
                            if isinstance(matchup, dict):
                                home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                start_time = matchup.get('startTime', 'TBD')
                                game_id = matchup.get('bfid')
                                
                                # Find odds for this match
                                match_odds = find_hockey_game_odds(game_id, hockey_odds_data) if game_id else None
                                
                                with st.container():
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        st.markdown(f"**{away_team}**")
                                        st.caption("Away Team")
                                    with col_b:
                                        st.markdown(f"**{home_team}**")
                                        st.caption("Home Team")
                                    with col_c:
                                        st.caption(f"⏰ {start_time}")
                                    
                                    if match_odds and 'marketIds' in match_odds:
                                        st.markdown("**📊 Betting Markets:**")
                                        for market in match_odds['marketIds'][:3]:  # Show max 3 markets
                                            if isinstance(market, dict):
                                                market_name = market.get('marketName', 'Moneyline')
                                                total_matched = market.get('totalMatched', 0)
                                                st.caption(f"💰 {market_name}: ${total_matched:,.2f} matched")
                                    else:
                                        st.caption("📊 Betting odds available")
                                    
                                    st.markdown("---")
                    else:
                        st.info("📅 No games scheduled currently")
                        st.caption("Check back during hockey season for live games")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Hockey data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Hockey competitions data not available")
        st.info("Hockey season may be off-season. Check back during NHL season.")

# SOCCER TAB
with tabs[9]:
    st.markdown('<div class="section-title">Soccer<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Auto-load soccer data on page load
    def load_soccer_data():
        try:
            api_url = "http://127.0.0.1:5001/api/soccer/competitions"
            response = requests.get(api_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                    return data['data']
            return []
        except:
            return []
    
    # Auto-load matchups for a specific soccer competition
    def load_soccer_matchups(competition_id):
        try:
            api_url = f"http://127.0.0.1:5001/api/soccer/matchups/{competition_id}"
            response = requests.get(api_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'data' in data:
                    return data['data']
            return []
        except Exception as e:
            return []
    
    # Load all soccer competitions
    all_soccer_competitions = load_soccer_data()
    
    if all_soccer_competitions:
        st.success(f"⚽ Found {len(all_soccer_competitions)} soccer competitions")
        
        # Create two columns for better organization
        col1, col2 = st.columns(2)
        mid_point = len(all_soccer_competitions) // 2
        
        with col1:
            st.subheader("⚽ Major Leagues (Part 1)")
            for i, comp in enumerate(all_soccer_competitions[:mid_point]):
                if isinstance(comp, dict) and 'competition' in comp:
                    comp_info = comp['competition']
                    comp_name = comp_info.get('name', 'Unknown League')
                    comp_region = comp.get('competitionRegion', 'International')
                    markets_count = comp.get('marketCount', 0)
                    
                    with st.expander(f"⚽ {comp_name}", expanded=False):
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                        
                        # Load matchups for this competition
                        comp_matchups = load_soccer_matchups(comp_info.get('id'))
                        
                        if comp_matchups:
                            st.write(f"**{len(comp_matchups)} Upcoming Matches:**")
                            for matchup in comp_matchups[:5]:  # Show max 5 matches
                                if isinstance(matchup, dict):
                                    home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                    away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                    start_time = matchup.get('startTime', 'TBD')
                                    
                                    st.markdown(f"**{away_team} vs {home_team}**")
                                    st.caption(f"⏰ {start_time}")
                                    
                                    if 'markets' in matchup and matchup['markets']:
                                        st.caption(f"📊 {len(matchup['markets'])} betting markets")
                                    
                                    st.markdown("---")
                        else:
                            st.info("📅 No matches scheduled currently")
        
        with col2:
            st.subheader("⚽ Major Leagues (Part 2)")
            for i, comp in enumerate(all_soccer_competitions[mid_point:]):
                if isinstance(comp, dict) and 'competition' in comp:
                    comp_info = comp['competition']
                    comp_name = comp_info.get('name', 'Unknown League')
                    comp_region = comp.get('competitionRegion', 'International')
                    markets_count = comp.get('marketCount', 0)
                    
                    with st.expander(f"⚽ {comp_name}", expanded=False):
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                        
                        # Load matchups for this competition
                        comp_matchups = load_soccer_matchups(comp_info.get('id'))
                        
                        if comp_matchups:
                            st.write(f"**{len(comp_matchups)} Upcoming Matches:**")
                            for matchup in comp_matchups[:5]:  # Show max 5 matches
                                if isinstance(matchup, dict):
                                    home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                    away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                    start_time = matchup.get('startTime', 'TBD')
                                    
                                    st.markdown(f"**{away_team} vs {home_team}**")
                                    st.caption(f"⏰ {start_time}")
                                    
                                    if 'markets' in matchup and matchup['markets']:
                                        st.caption(f"📊 {len(matchup['markets'])} betting markets")
                                    
                                    st.markdown("---")
                        else:
                            st.info("📅 No matches scheduled currently")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Soccer data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Soccer competitions data not available")
        st.info("Unable to load soccer competitions. Please check the API connection.")

# ESPORTS TAB
with tabs[10]:
    st.markdown('<div class="section-title">Esports<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached esports data
    all_esports_competitions = get_cached_data("esports_competitions") or {"data": []}
    esports_odds_data = get_cached_data("esports_odds") or {"data": []}
    
    # Extract data arrays
    all_esports_competitions = all_esports_competitions.get('data', [])
    esports_odds_data = esports_odds_data.get('data', [])
    
    # Auto-load matchups for a specific esports competition (with caching)
    def load_esports_matchups(competition_id):
        cache_key = f"esports_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/esports/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific esports match
    def find_esports_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_esports_competitions:
        st.success(f"🎮 Found {len(all_esports_competitions)} esports competitions")
        
        # Display esports competitions
        for i, comp in enumerate(all_esports_competitions):
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown Tournament')
                comp_region = comp.get('competitionRegion', 'Global')
                markets_count = comp.get('marketCount', 0)
                
                # Determine game type by name patterns
                game_icon = "🎮"
                if any(game in comp_name.upper() for game in ['LOL', 'LEAGUE']):
                    game_icon = "⚔️"
                elif any(game in comp_name.upper() for game in ['CS', 'COUNTER']):
                    game_icon = "🔫"
                elif 'DOTA' in comp_name.upper():
                    game_icon = "🐲"
                elif 'VALORANT' in comp_name.upper():
                    game_icon = "🎯"
                
                with st.expander(f"{game_icon} {comp_name}", expanded=i < 3):  # Expand first 3 tournaments
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                    
                    # Load matchups for this competition
                    comp_matchups = load_esports_matchups(comp_info.get('id'))
                    
                    if comp_matchups:
                        st.write(f"**{len(comp_matchups)} Upcoming Matches:**")
                        
                        for matchup in comp_matchups:
                            if isinstance(matchup, dict):
                                home_team = matchup.get('homeTeam', {}).get('name', 'Team 1')
                                away_team = matchup.get('awayTeam', {}).get('name', 'Team 2')
                                start_time = matchup.get('startTime', 'TBD')
                                game_id = matchup.get('bfid')
                                
                                # Find odds for this match
                                match_odds = find_esports_game_odds(game_id, esports_odds_data) if game_id else None
                                
                                with st.container():
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        st.markdown(f"**{away_team}**")
                                        st.caption("Team 1")
                                    with col_b:
                                        st.markdown(f"**{home_team}**")
                                        st.caption("Team 2")
                                    with col_c:
                                        st.caption(f"⏰ {start_time}")
                                    
                                    if match_odds and 'marketIds' in match_odds:
                                        st.markdown("**📊 Betting Markets:**")
                                        for market in match_odds['marketIds'][:3]:  # Show max 3 markets
                                            if isinstance(market, dict):
                                                market_name = market.get('marketName', 'Match Winner')
                                                total_matched = market.get('totalMatched', 0)
                                                st.caption(f"💰 {market_name}: ${total_matched:,.2f} matched")
                                    else:
                                        st.caption("📊 Betting odds available")
                                    
                                    st.markdown("---")
                    else:
                        st.info("📅 No matches scheduled currently")
                        st.caption("Check back for upcoming tournament matches")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Esports data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Esports competitions data not available")
        st.info("No active esports tournaments found. Check back for major events like Worlds, TI, or Majors.")

# COLLEGE FOOTBALL TAB
with tabs[11]:
    st.markdown('<div class="section-title">College Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached college football data
    all_college_football_competitions = get_cached_data("college_football_competitions") or {"data": []}
    college_football_odds_data = get_cached_data("college_football_odds") or {"data": []}
    
    # Extract data arrays
    all_college_football_competitions = all_college_football_competitions.get('data', [])
    college_football_odds_data = college_football_odds_data.get('data', [])
    
    # Auto-load matchups for a specific college football competition (with caching)
    def load_college_football_matchups(competition_id):
        cache_key = f"college_football_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/college-football/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific college football game
    def find_college_football_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_college_football_competitions:
        st.success(f"🏈 Found {len(all_college_football_competitions)} college football competitions")
        
        # Display college football competitions
        for i, comp in enumerate(all_college_football_competitions):
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown Conference')
                comp_region = comp.get('competitionRegion', 'USA')
                markets_count = comp.get('marketCount', 0)
                
                # Add conference icons
                conference_icon = "🏈"
                if any(conf in comp_name.upper() for conf in ['SEC', 'SOUTHEASTERN']):
                    conference_icon = "⭐"
                elif any(conf in comp_name.upper() for conf in ['BIG TEN', 'B1G']):
                    conference_icon = "🔟"
                elif 'ACC' in comp_name.upper():
                    conference_icon = "🅰️"
                elif 'BIG 12' in comp_name.upper():
                    conference_icon = "1️⃣2️⃣"
                elif 'PAC' in comp_name.upper():
                    conference_icon = "🌊"
                
                with st.expander(f"{conference_icon} {comp_name}", expanded=i < 2):  # Expand first 2 conferences
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                    
                    # Load matchups for this competition
                    comp_matchups = load_college_football_matchups(comp_info.get('id'))
                    
                    if comp_matchups:
                        st.write(f"**{len(comp_matchups)} Upcoming Games:**")
                        
                        for matchup in comp_matchups:
                            if isinstance(matchup, dict):
                                home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                start_time = matchup.get('startTime', 'TBD')
                                game_id = matchup.get('bfid')
                                
                                # Find odds for this match
                                match_odds = find_college_football_game_odds(game_id, college_football_odds_data) if game_id else None
                                
                                with st.container():
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        st.markdown(f"**{away_team}**")
                                        st.caption("Away Team")
                                    with col_b:
                                        st.markdown(f"**{home_team}**")
                                        st.caption("Home Team")
                                    with col_c:
                                        st.caption(f"⏰ {start_time}")
                                    
                                    if match_odds and 'marketIds' in match_odds:
                                        st.markdown("**📊 Betting Markets:**")
                                        for market in match_odds['marketIds'][:3]:  # Show max 3 markets
                                            if isinstance(market, dict):
                                                market_name = market.get('marketName', 'Spread')
                                                total_matched = market.get('totalMatched', 0)
                                                st.caption(f"💰 {market_name}: ${total_matched:,.2f} matched")
                                    else:
                                        st.caption("📊 Betting odds available")
                                    
                                    st.markdown("---")
                    else:
                        st.info("📅 No games scheduled currently")
                        st.caption("Check back during college football season for live games")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"College Football data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ College Football competitions data not available")
        st.info("College football season may be off-season. Check back during NCAA season.")




