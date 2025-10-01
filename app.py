# BetFinder AI - Instagram Sport Bar Template Revamp
import streamlit as st
import pandas as pd
from datetime import datetime
# Helper functions
def implied_prob_from_decimal_odds(odds):
    if pd.isna(odds) or odds <= 0:
        return None
    return 1.0 / odds
def price_improvement(avg_prob, best_prob):
    if pd.isna(avg_prob) or pd.isna(best_prob) or avg_prob == 0:
        return None
    return ((avg_prob - best_prob) / avg_prob) * 100
# Page config
st.set_page_config(page_title="BetFinder AI", layout="wide")
# Load data (placeholder - replace with actual data loading)
df = pd.DataFrame()  # Replace with actual data loading
# Filter parameters (no sidebar - moved to main area if needed)
selected_sport = None
selected_leagues = []
selected_markets = []
selected_bookmakers = []
min_odds = 1.0
max_odds = 10.0
value_threshold = 5.0
# Filter data
filtered_df = df.copy()
if not filtered_df.empty:
    if selected_sport and selected_sport != "All":
        filtered_df = filtered_df[filtered_df["sport_title"] == selected_sport]
    if selected_leagues:
        filtered_df = filtered_df[filtered_df["sport_key"].isin(selected_leagues)]
    if selected_markets:
        filtered_df = filtered_df[filtered_df["market"].isin(selected_markets)]
    if selected_bookmakers:
        filtered_df = filtered_df[filtered_df["bookmaker"].isin(selected_bookmakers)]
    filtered_df = filtered_df[(filtered_df["price"].fillna(0) >= min_odds) & (filtered_df["price"].fillna(0) <= max_odds)]
# NAV TABS - Updated with Value Picks, Top 10, and Sports Categories
nav_tabs = st.tabs(["Value Picks", "BFAI Top 10", "Tennis", "Basketball", "Football", "Baseball", "Hockey", "Soccer", "Golf", "MMA", "Cricket", "Rugby"])
# Compute value bets
value_candidates = pd.DataFrame()
if not filtered_df.empty:
    dfw = filtered_df.copy()
    dfw["implied_prob"] = dfw["price"].apply(implied_prob_from_decimal_odds)
    grp = dfw.groupby(["event_id", "market", "name"], dropna=False)
    agg = grp.agg(avg_market_prob=("implied_prob", "mean"), best_price=("price", "max")).reset_index()
    agg["best_implied_prob"] = agg["best_price"].apply(implied_prob_from_decimal_odds)
    agg["edge_pct"] = agg.apply(lambda r: price_improvement(r["avg_market_prob"], r["best_implied_prob"]), axis=1)
    value_candidates = agg[(agg["edge_pct"].notna()) & (agg["edge_pct"] > value_threshold)].copy()
# VALUE PICKS TAB
with nav_tabs[0]:
    st.markdown('<div class="section-title">Value Picks<span class="time">Auto-screened across books</span></div>', unsafe_allow_html=True)
    if value_candidates.empty:
        st.warning("⚠️ No value bets found matching your criteria. Try adjusting filters.")
    else:
        st.dataframe(value_candidates.sort_values("edge_pct", ascending=False).head(50), use_container_width=True)
# TOP 10 PICKS TAB
with nav_tabs[1]:
    st.markdown('<div class="section-title">BetFinder AI Top 10 Picks of the Hour<span class="time">Updated hourly</span></div>', unsafe_allow_html=True)
    if value_candidates.empty:
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
# RUGBY TAB
with nav_tabs[11]:
    st.markdown('<div class="section-title">Rugby<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    rugby_df = filtered_df[filtered_df["sport_title"] == "Rugby"] if not filtered_df.empty and "sport_title" in filtered_df.columns else pd.DataFrame()
    if rugby_df.empty:
        st.info("No rugby events available.")
    else:
        st.dataframe(rugby_df.head(100), use_container_width=True)
