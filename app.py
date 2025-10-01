# BetFinder AI - Instagram Sport Bar Template Revamp
import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
from typing import Dict, Any, List, Optional

st.set_page_config(
    page_title="BetFinder AI - Sport Bar Vibes",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API key securely: env var first, then Streamlit secrets
API_KEY = (
    os.environ.get("SPORTBEX_API_KEY")
    or os.environ.get("ODDS_API_KEY")
    or st.secrets.get("SPORTBEX_API_KEY", None)
    or st.secrets.get("ODDS_API_KEY", None)
)
API_BASE = "https://api.the-odds-api.com/v4"

st.markdown("<div class='app-bg'>", unsafe_allow_html=True)

# Robust odds fetcher with retries and clearer errors
def fetch_live_odds(
    sport: str = "tennis",
    regions: str = "us",
    markets: str = "h2h,totals,spreads,outrights,player_props,team_props",
    retries: int = 2,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    if not API_KEY:
        raise RuntimeError(
            "Missing API key. Set SPORTBEX_API_KEY/ODDS_API_KEY env var or st.secrets."
        )
    url = f"{API_BASE}/sports/{sport}/odds/"
    params = {"apiKey": API_KEY, "regions": regions, "markets": markets}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("Unauthorized: Check API key or plan limits (401).")
            if r.status_code == 403:
                raise RuntimeError("Forbidden: Key lacks access to resource (403).")
            if r.status_code == 429:
                raise RuntimeError("Rate limited (429): Too many requests.")
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise ValueError("Unexpected response format: expected a list of events.")
            return data
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
        except requests.HTTPError as e:
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            raise RuntimeError(f"HTTP error {r.status_code}: {msg}") from e
        except Exception as e:
            last_err = e
            break
    if last_err:
        raise RuntimeError(f"Failed to fetch odds after retries: {last_err}")
    return []


def normalize_odds_json(odds_json: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not odds_json:
        return pd.DataFrame()
    for event in odds_json:
        event_id = event.get("id")
        sport_key = event.get("sport_key")
        sport_title = event.get("sport_title")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        for bk in event.get("bookmakers", []) if isinstance(event.get("bookmakers", []), list) else []:
            bookmaker = (bk.get("title") or bk.get("key") or "") if isinstance(bk, dict) else ""
            for mk in bk.get("markets", []) if isinstance(bk, dict) else []:
                market_key = mk.get("key") if isinstance(mk, dict) else None
                for out in mk.get("outcomes", []) if isinstance(mk, dict) else []:
                    rows.append(
                        {
                            "event_id": event_id,
                            "sport_key": sport_key,
                            "sport_title": sport_title,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker,
                            "market": market_key,
                            "name": out.get("name") if isinstance(out, dict) else None,
                            "price": out.get("price") if isinstance(out, dict) else None,
                            "point": out.get("point") if isinstance(out, dict) else None,
                        }
                    )
    df = pd.DataFrame(rows)
    for col in ["price", "point"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def implied_prob_from_decimal_odds(odds: float) -> Optional[float]:
    try:
        if odds and odds > 1.0:
            return 1.0 / odds
    except Exception:
        pass
    return None


def price_improvement(consensus_prob: Optional[float], price_prob: Optional[float]) -> Optional[float]:
    try:
        if consensus_prob is None or price_prob is None:
            return None
        return (consensus_prob - price_prob) * 100.0
    except Exception:
        return None


# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-brand">', unsafe_allow_html=True)
    st.image(
        "https://via.placeholder.com/300x90/10b981/04120a?text=BetFinder+AI",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Filters & Settings")

with st.spinner("Fetching latest odds..."):
    try:
        odds_json = fetch_live_odds()
        df = normalize_odds_json(odds_json)
    except Exception as e:
        st.error(f"Failed to fetch odds: {e}")
        df = pd.DataFrame()

sports_list = sorted(df["sport_title"].dropna().unique().tolist()) if not df.empty else []
leagues_list = sorted(df["sport_key"].dropna().unique().tolist()) if not df.empty else []
bookmakers_list = sorted(df["bookmaker"].dropna().unique().tolist()) if not df.empty else []
markets_list = sorted(df["market"].dropna().unique().tolist()) if not df.empty else []

with st.sidebar:
    selected_sport = st.selectbox("🏆 Select Sport", options=(sports_list or ["All"]))
    selected_leagues = st.multiselect("📊 Select Leagues", options=leagues_list, default=leagues_list)
    selected_markets = st.multiselect("🧭 Select Markets", options=markets_list, default=markets_list)
    selected_bookmakers = st.multiselect("🏪 Select Bookmakers", options=bookmakers_list, default=bookmakers_list)
    min_odds, max_odds = st.slider(
        "💰 Odds Range",
        min_value=float(df["price"].min()) if not df.empty else 1.0,
        max_value=float(df["price"].max()) if not df.empty else 10.0,
        value=(
            float(max(1.0, df["price"].min())) if not df.empty else 1.5,
            float(min(10.0, df["price"].max())) if not df.empty else 5.0,
        ),
        step=0.1,
        disabled=df.empty,
    )
    value_threshold = st.slider("📈 Value Bet Threshold (%)", 0, 20, 5, 1)

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

# NAV TABS
nav_tabs = st.tabs(["🎯 Value Bets", "📺 Live Events", "📅 Upcoming", "📊 Analytics", "📚 Odds Board", "💾 Tracker"])

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

# VALUE BETS TAB
with nav_tabs[0]:
    st.markdown('<div class="section-title">Top Value Opportunities<span class="time">Auto-screened across books</span></div>', unsafe_allow_html=True)
    if value_candidates.empty:
        st.warning("⚠️ No value bets found matching your criteria. Try adjusting filters.")
    else:
        st.dataframe(
            value_candidates.sort_values("edge_pct", ascending=False).head(50),
            use_container_width=True,
        )

# LIVE EVENTS TAB
with nav_tabs[1]:
    st.markdown('<div class="section-title">Live Events<span class="time">Right now</span></div>', unsafe_allow_html=True)
    display_cols = ["commence_time", "sport_title", "home_team", "away_team", "market", "name", "price", "point", "bookmaker"]
    df_live = filtered_df[display_cols] if not filtered_df.empty else pd.DataFrame(columns=display_cols)
    if df_live.empty:
        st.warning("📭 No live events available. Adjust filters or try later.")
    else:
        st.dataframe(df_live.head(200), use_container_width=True)

# UPCOMING TAB (simple placeholder cards)
with nav_tabs[2]:
    st.markdown('<div class="section-title">Upcoming Matches<span class="time">Next 48 hours</span></div>', unsafe_allow_html=True)
    st.write("Coming soon: curated upcoming matches view.")
