import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="BetFinder AI - Sports Betting Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for StatKing.ai inspired design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    .odds-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .value-bet-badge {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎯 BetFinder AI</h1>', unsafe_allow_html=True)
st.markdown("**Advanced Sports Betting Analytics Platform** - Powered by AI")
st.divider()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=BetFinder+AI", use_column_width=True)
    st.markdown("### ⚙️ Filters & Settings")
    
    # Sport filter
    selected_sport = st.selectbox(
        "🏆 Select Sport",
        ["⚽ Soccer", "🏀 Basketball", "🏈 Football", "⚾ Baseball", "🎾 Tennis"]
    )
    
    # League filter
    leagues = {
        "⚽ Soccer": ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"],
        "🏀 Basketball": ["NBA", "EuroLeague", "NCAA"],
        "🏈 Football": ["NFL", "NCAA Football"],
        "⚾ Baseball": ["MLB", "NPB"],
        "🎾 Tennis": ["ATP", "WTA", "Grand Slams"]
    }
    selected_leagues = st.multiselect(
        "📊 Select Leagues",
        leagues.get(selected_sport, []),
        default=leagues.get(selected_sport, [])
    )
    
    # Bookmaker filter
    bookmakers = ["Bet365", "DraftKings", "FanDuel", "BetMGM", "Caesars", "PointsBet"]
    selected_bookmakers = st.multiselect(
        "🏪 Select Bookmakers",
        bookmakers,
        default=bookmakers
    )
    
    # Odds range
    st.markdown("### 💰 Odds Range")
    min_odds, max_odds = st.slider(
        "Select range",
        min_value=1.0,
        max_value=10.0,
        value=(1.5, 5.0),
        step=0.1
    )
    
    # Value bet threshold
    value_threshold = st.slider(
        "📈 Value Bet Threshold (%)",
        min_value=0,
        max_value=20,
        value=5,
        step=1
    )
    
    st.divider()
    
    # Quick stats in sidebar
    st.markdown("### 📊 Quick Stats")
    st.metric("Active Bets", "0")
    st.metric("Total ROI", "0%")
    st.metric("Win Rate", "0%")

# Main content area
# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <p class="metric-label">📈 Value Bets Found</p>
        <p class="metric-value">0</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <p class="metric-label">💵 Potential Value</p>
        <p class="metric-value">$0</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <p class="metric-label">🎲 Markets Analyzed</p>
        <p class="metric-value">0</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <p class="metric-label">⚡ Last Update</p>
        <p class="metric-value">Now</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Value Bets", "📊 Live Odds", "📈 Analytics", "💾 Bet Tracker"])

with tab1:
    st.markdown("### 🎯 Top Value Betting Opportunities")
    st.info("🔄 Click 'Analyze Odds' to find value bets based on your criteria")
    
    if st.button("🔍 Analyze Odds", type="primary", use_container_width=True):
        with st.spinner("Analyzing odds from multiple bookmakers..."):
            time.sleep(2)
            st.warning("⚠️ No value bets found matching your criteria. Try adjusting filters.")
    
    # Sample data structure for value bets
    st.markdown("""
    <div class="odds-table">
        <h4>Expected Format:</h4>
        <ul>
            <li><strong>Match:</strong> Team A vs Team B</li>
            <li><strong>Market:</strong> Match Winner / Over/Under / Both Teams to Score</li>
            <li><strong>Bookmaker:</strong> Best available odds</li>
            <li><strong>Odds:</strong> Decimal odds</li>
            <li><strong>Value %:</strong> Expected value percentage</li>
            <li><strong>Recommended Stake:</strong> Kelly criterion based</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Live Odds Board")
    st.info(f"Showing odds for {selected_sport} | Leagues: {', '.join(selected_leagues) if selected_leagues else 'None selected'}")
    
    # Sample odds dataframe
    sample_data = {
        "Time": [],
        "League": [],
        "Match": [],
        "Home Odds": [],
        "Draw Odds": [],
        "Away Odds": [],
        "Bookmaker": []
    }
    
    df_odds = pd.DataFrame(sample_data)
    
    if df_odds.empty:
        st.warning("📭 No live odds available. Adjust your filters or check back later.")
    else:
        st.dataframe(df_odds, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📈 Betting Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Performance Overview")
        st.line_chart(pd.DataFrame(
            {"ROI": [0], "Profit": [0]},
            index=[datetime.now()]
        ))
    
    with col2:
        st.markdown("#### 🎯 Win Rate by Sport")
        st.bar_chart(pd.DataFrame(
            {"Win Rate": []},
            index=[]
        ))
    
    st.markdown("#### 🔥 Hot Bookmakers")
    st.info("Track which bookmakers offer the best value over time")
    
    st.markdown("#### 📉 Odds Movement Tracker")
    st.info("Monitor how odds change leading up to events")

with tab4:
    st.markdown("### 💾 Bet Tracking & History")
    
    # Bet entry form
    with st.expander("➕ Add New Bet", expanded=False):
        with st.form("bet_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bet_sport = st.selectbox("Sport", ["Soccer", "Basketball", "Football", "Baseball", "Tennis"])
                bet_league = st.text_input("League")
                bet_event = st.text_input("Event/Match")
            
            with col2:
                bet_type = st.selectbox("Bet Type", ["Match Winner", "Over/Under", "Handicap", "Both Teams to Score", "Other"])
                bet_selection = st.text_input("Selection")
                bet_odds = st.number_input("Odds", min_value=1.01, value=2.0, step=0.01)
            
            with col3:
                bet_stake = st.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0)
                bet_bookmaker = st.selectbox("Bookmaker", bookmakers)
                bet_date = st.date_input("Date")
            
            submitted = st.form_submit_button("💾 Save Bet", use_container_width=True)
            if submitted:
                st.success("✅ Bet saved successfully!")
    
    # Bet history table
    st.markdown("#### 📋 Recent Bets")
    bet_history = pd.DataFrame({
        "Date": [],
        "Sport": [],
        "Event": [],
        "Bet Type": [],
        "Odds": [],
        "Stake": [],
        "Status": [],
        "P/L": []
    })
    
    if bet_history.empty:
        st.info("📭 No bets recorded yet. Add your first bet above!")
    else:
        st.dataframe(bet_history, use_container_width=True, hide_index=True)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Bets", "0")
        with col2:
            st.metric("Total Staked", "$0")
        with col3:
            st.metric("Total Profit", "$0", delta="0%")
        with col4:
            st.metric("Win Rate", "0%")

# Footer
st.divider()
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("**BetFinder AI** - Smart Sports Betting Analytics")
with col2:
    st.markdown("[📚 Documentation](#)")
with col3:
    st.markdown("[⚙️ Settings](#)")

st.caption("⚠️ Responsible Gambling: Please bet responsibly. This tool is for informational purposes only.")
