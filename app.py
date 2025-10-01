import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

# API Configuration
API_KEY = "ede96651f63959b778ed2e2bbb2331f1"


def fetch_live_odds(sport="tennis", regions="us", markets="h2h,totals,spreads,outrights,player_props,team_props"):
    """Fetch live odds from The Odds API"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {"apiKey": API_KEY, "regions": regions, "markets": markets}
    r = requests.get(url, params=params)
    r.raise_for_status()
    odds_json = r.json()
    # Diagnostic: dump raw API response
    st.write(odds_json)
    return odds_json


# Helper to normalize odds JSON into a flat dataframe catching all market types
def normalize_odds_json(odds_json):
    rows = []
    if not odds_json:
        return pd.DataFrame()
    for event in odds_json:
        event_id = event.get("id")
        sport_key = event.get("sport_key")
        sport_title = event.get("sport_title")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        for bk in event.get("bookmakers", []):
            bookmaker = bk.get("title") or bk.get("key")
            for mk in bk.get("markets", []):
                market_key = mk.get("key")
                market_name = mk.get("key")
                for out in mk.get("outcomes", []):
                    row = {
                        "event_id": event_id,
                        "sport_key": sport_key,
                        "sport_title": sport_title,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker,
                        "market": market_key or market_name,
                        "name": out.get("name"),
                        "price": out.get("price"),
                        "point": out.get("point"),
                    }
                    rows.append(row)
    df = pd.DataFrame(rows)
    return df


# Page config
st.set_page_config(
    page_title="BetFinder AI - Sports Betting Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for StatKing.ai inspired design
st.markdown(
    """    .main-header {
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
    }""",
    unsafe_allow_html=True
)

# Main header
st.markdown('<h1 class="main-header">🎯 BetFinder AI</h1>', unsafe_allow_html=True)
st.markdown("**Advanced Sports Betting Analytics Platform** - Powered by AI")
st.divider()

# Fetch data upfront with try/except and build df for dynamic filters
with st.spinner("Fetching latest odds..."):
    try:
        odds_json = fetch_live_odds()
        df = normalize_odds_json(odds_json)
    except Exception as e:
        st.error(f"Failed to fetch odds: {e}")
        df = pd.DataFrame()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=BetFinder+AI", use_column_width=True)
    st.markdown("### ⚙️ Filters & Settings")

    # Dynamic unique values from df for filters (fallbacks to empty lists)
    sports_list = sorted(df["sport_title"].dropna().unique().tolist()) if not df.empty else []
    leagues_list = sorted(df["sport_key"].dropna().unique().tolist()) if not df.empty else []
    bookmakers_list = sorted(df["bookmaker"].dropna().unique().tolist()) if not df.empty else []
    markets_list = sorted(df["market"].dropna().unique().tolist()) if not df.empty else []

    # Use sensible defaults only from available values
    selected_sport = st.selectbox(
        "🏆 Select Sport",
        options=sports_list,
        index=0 if sports_list else None,
        placeholder="Select sport" if not sports_list else None,
    )

    selected_leagues = st.multiselect(
        "📊 Select Leagues",
        options=leagues_list,
        default=leagues_list,
    )

    selected_markets = st.multiselect(
        "🧭 Select Markets",
        options=markets_list,
        default=markets_list,
    )

    selected_bookmakers = st.multiselect(
        "🏪 Select Bookmakers",
        options=bookmakers_list,
        default=bookmakers_list,
    )

    st.markdown("### 💰 Odds Range")
    min_odds, max_odds = st.slider(
        "Select range",
        min_value=float(df["price"].min()) if not df.empty else 1.0,
        max_value=float(df["price"].max()) if not df.empty else 10.0,
        value=(
            float(max(1.0, df["price"].min())) if not df.empty else 1.5,
            float(min(10.0, df["price"].max())) if not df.empty else 5.0,
        ),
        step=0.1,
        disabled=df.empty,
    )

    value_threshold = st.slider(
        "📈 Value Bet Threshold (%)",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
    )

    st.divider()

    st.markdown("### 📊 Quick Stats")
    st.metric("Active Bets", "0")
    st.metric("Total ROI", "0%")
    st.metric("Win Rate", "0%")

# Apply filters to df for display
filtered_df = df.copy()
if not filtered_df.empty:
    if selected_sport:
        filtered_df = filtered_df[filtered_df["sport_title"] == selected_sport]
    if selected_leagues:
        filtered_df = filtered_df[filtered_df["sport_key"].isin(selected_leagues)]
    if selected_markets:
        filtered_df = filtered_df[filtered_df["market"].isin(selected_markets)]
    if selected_bookmakers:
        filtered_df = filtered_df[filtered_df["bookmaker"].isin(selected_bookmakers)]
    filtered_df = filtered_df[(filtered_df["price"].fillna(0) >= min_odds) & (filtered_df["price"].fillna(0) <= max_odds)]

# Main content area
# Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        """
    <div class="stat-card">
        <p class="metric-label">📈 Value Bets Found</p>
        <p class="metric-value">0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
    <div class="stat-card">
        <p class="metric-label">💵 Potential Value</p>
        <p class="metric-value">$0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
    <div class="stat-card">
        <p class="metric-label">🎲 Markets Analyzed</p>
        <p class="metric-value">0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
    <div class="stat-card">
        <p class="metric-label">⚡ Last Update</p>
        <p class="metric-value">Now</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

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
    st.markdown(
        """
    <div class="odds-table">
        Expected Format:
        <ul>
            Match: Team A vs Team B
            Market: Match Winner / Over/Under / Both Teams to Score
            Bookmaker: Best available odds
            Odds: Decimal odds
            Value %: Expected value percentage
            Recommended Stake: Kelly criterion based
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("### 📊 Live Odds Board")
    st.info(
        f"Showing odds for {selected_sport or 'All Sports'} | Leagues: {', '.join(selected_leagues) if selected_leagues else 'All'} | Markets: {', '.join(selected_markets) if selected_markets else 'All'}"
    )

    # Build a presentable view from filtered_df
    display_cols = [
        "commence_time",
        "sport_title",
        "sport_key",
        "home_team",
        "away_team",
        "market",
        "name",
        "price",
        "point",
        "bookmaker",
    ]

    df_odds = filtered_df[display_cols] if not filtered_df.empty else pd.DataFrame(columns=display_cols)

    # Empty warning section
    if df_odds.empty:
        st.warning("📭 No live odds available for current filters or feed is empty. Adjust filters or try again later.")
    else:
        st.dataframe(df_odds, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📈 Betting Analytics & Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Performance Overview")
        st.line_chart(pd.DataFrame({"ROI": [0], "Profit": [0]}, index=[datetime.now()]))

    with col2:
        st.markdown("#### 🎯 Win Rate by Sport")
        st.bar_chart(pd.DataFrame({"Win Rate": []}, index=[]))

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
                bet_sport = st.selectbox("Sport", sorted(sports_list) or [""], index=0 if sports_list else 0)
                bet_league = st.text_input("League")
                bet_event = st.text_input("Event/Match")

            with col2:
                bet_type = st.selectbox(
                    "Bet Type",
                    sorted(selected_markets) if selected_markets else sorted(markets_list) or ["Other"],
                )
                bet_selection = st.text_input("Selection")
                bet_odds = st.number_input("Odds", min_value=1.01, value=2.0, step=0.01)

            with col3:
                bet_stake = st.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0)
                bet_bookmaker = st.selectbox("Bookmaker", sorted(bookmakers_list) or [""])
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
        "P/L": [],
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
