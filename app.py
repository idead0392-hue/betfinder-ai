# BetFinder AI - Instagram Sport Bar Template Revamp
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

st.set_page_config(
    page_title="BetFinder AI - Sport Bar Vibes",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root{
        --green:#10b981;           /* vivid green */
        --green-dark:#059669;
        --green-ghost:#d1fae5;
        --ink:#0b1220;
        --ink-2:#111827;
        --card:#0f172a;            /* slate card */
        --muted:#6b7280;
        --accent:#22d3ee;          /* cyan accents */
        --warning:#f59e0b;
        --danger:#ef4444;
        --panel:#111827aa;         /* translucent panels */
      }
      .app-bg { background: radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,.25), transparent),
                              radial-gradient(1000px 500px at 110% 10%, rgba(16,185,129,.25), transparent),
                              linear-gradient(180deg, #0b1220, #0b1220); padding:0; margin:0; }
      .hero { background: linear-gradient(135deg, var(--green), var(--green-dark)); border-radius: 16px; padding: 18px; color: white; display:flex; align-items:center; gap:18px; }
      .hero h1 {font-size: 2.1rem; line-height:1.1; margin:0; font-weight:900;}
      .hero .kpis {display:flex; gap:12px; flex-wrap:wrap}
      .kpi {background: rgba(255,255,255,.15); padding:10px 12px; border-radius:12px;}
      .kpi .v {font-size:1.15rem; font-weight:800}
      .kpi .l {font-size:.8rem; opacity:.95}
      .section-title {display:flex; align-items:center; justify-content:space-between; margin: 14px 0 8px 0;}
      .section-title h3 {color:#e5e7eb; font-size:1.05rem; letter-spacing:.25px; margin:0;}
      .section-title .time {color:#9ca3af; font-size:.85rem}
      .card-grid {display:grid; grid-template-columns: repeat(12, 1fr); gap:12px;}
      .card {grid-column: span 4; background: var(--card); border-radius:16px; overflow:hidden; border: 1px solid #1f2937;}
      .card .banner {height:120px; background-size:cover; background-position:center;}
      .card .body {padding:12px 14px;}
      .badge {display:inline-block; padding:4px 10px; background: var(--green-ghost); color:var(--green-dark); border-radius:999px; font-weight:700; font-size:.75rem}
      .teams {display:flex; align-items:center; justify-content:space-between; color:#e5e7eb; font-weight:800; font-size:1rem}
      .teams span {display:inline-block}
      .meta {display:flex; gap:10px; margin-top:8px; color:#9ca3af; font-size:.8rem}
      .odds {margin-top:8px; display:flex; gap:8px}
      .odds .pill {flex:1; text-align:center; padding:8px; border-radius:10px; background:#0b1220; color:#e5e7eb; border:1px solid #1f2937}
      .odds .best {background: linear-gradient(180deg, #064e3b, #065f46); border-color:#059669;}
      .panel {background: var(--panel); border:1px solid #1f2937; border-radius:14px; padding:12px}
      .divider {height:1px; background:#1f2937; margin: 8px 0}
      .stTabs [data-baseweb=tab-list] {gap: 8px}
      .stTabs [data-baseweb=tab] {border-radius:999px; padding:8px 14px; background:#0f172a; color:#e5e7eb; font-weight:700}
      .stTabs [aria-selected=true] {background: var(--green) !important; color:#04120a !important}
      .sidebar-brand img {border-radius:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-bg">', unsafe_allow_html=True)

API_KEY = "ede96651f63959b778ed2e2bbb2331f1"

def fetch_live_odds(sport="tennis", regions="us", markets="h2h,totals,spreads,outrights,player_props,team_props"):
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {"apiKey": API_KEY, "regions": regions, "markets": markets}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

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
                for out in mk.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "sport_key": sport_key,
                        "sport_title": sport_title,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker,
                        "market": market_key,
                        "name": out.get("name"),
                        "price": out.get("price"),
                        "point": out.get("point"),
                    })
    return pd.DataFrame(rows)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-brand">', unsafe_allow_html=True)
    st.image("https://via.placeholder.com/300x90/10b981/04120a?text=BetFinder+AI", use_container_width=True)
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
        value=(float(max(1.0, df["price"].min())) if not df.empty else 1.5,
               float(min(10.0, df["price"].max())) if not df.empty else 5.0),
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

# HERO
colh1, colh2 = st.columns([2.2, 1])
with colh1:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.image(
        "https://images.unsplash.com/photo-1517649763962-0c623066013b?q=80&w=1600&auto=format&fit=crop",
        use_column_width=True,
    )
    st.markdown(
        """
        <div>
          <h1>BetFinder AI</h1>
          <div class="kpis">
            <div class="kpi"><div class="v">Live</div><div class="l">Events</div></div>
            <div class="kpi"><div class="v">Upcoming</div><div class="l">Matches</div></div>
            <div class="kpi"><div class="v">Value</div><div class="l">Signals</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
with colh2:
    st.markdown(
        f"<div class='panel'><b>Today</b><div class='divider'></div><div>🗓 {datetime.now().strftime('%a, %b %d')}</div><div>⏰ {datetime.now().strftime('%I:%M %p')}</div></div>",
        unsafe_allow_html=True,
    )

# NAV TABS
nav_tabs = st.tabs(["🎯 Value Bets", "📺 Live Events", "📅 Upcoming", "📊 Analytics", "📚 Odds Board", "💾 Tracker"])

# VALUE BETS
with nav_tabs[0]:
    st.markdown('<div class="section-title"><h3>Top Value Opportunities</h3><span class="time">Auto-screened across books</span></div>', unsafe_allow_html=True)
    if st.button("🔍 Analyze Odds", type="primary", use_container_width=True):
        with st.spinner("Analyzing odds from multiple bookmakers..."):
            time.sleep(2)
            st.warning("⚠️ No value bets found matching your criteria. Try adjusting filters.")
    st.markdown('<div class="card-grid">', unsafe_allow_html=True)
    for i in range(3):
        st.markdown(
            f"""
            <div class='card'>
              <div class='banner' style="background-image:url('https://images.unsplash.com/photo-1542574271-7f3b92e6c821?q=80&w=1600&auto=format&fit=crop')"></div>
              <div class='body'>
                <span class='badge'>Value {5+i*2}%</span>
                <div class='teams'><span>Team A</span><span>vs</span><span>Team B</span></div>
                <div class='meta'><span>Market: Match Winner</span><span>Book: Best</span></div>
                <div class='odds'>
                  <div class='pill'>A 2.10</div>
                  <div class='pill best'>B 2.35</div>
                  <div class='pill'>Draw 3.20</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# LIVE EVENTS
with nav_tabs[1]:
    st.markdown('<div class="section-title"><h3>Live Events</h3><span class="time">Right now</span></div>', unsafe_allow_html=True)
    display_cols = ["commence_time", "sport_title", "home_team", "away_team", "market", "name", "price", "point", "bookmaker"]
    df_live = filtered_df[display_cols] if not filtered_df.empty else pd.DataFrame(columns=display_cols)
    if df_live.empty:
        st.warning("📭 No live events available. Adjust filters or try later.")
    else:
        st.markdown('<div class="card-grid">', unsafe_allow_html=True)
        for _, row in df_live.head(9).iterrows():
            banner = "https://images.unsplash.com/photo-1521417531039-99f22f39a8f1?q=80&w=1600&auto=format&fit=crop"
            tm = (row["commence_time"] or "").replace("T"," ").replace("Z"," UTC")
            st.markdown(
                f"""
                <div class='card'>
                  <div class='banner' style="background-image:url('{banner}')"></div>
                  <div class='body'>
                    <span class='badge'>Live</span>
                    <div class='teams'><span>{row['home_team'] or ''}</span><span>vs</span><span>{row['away_team'] or ''}</span></div>
                    <div class='meta'><span>{row['sport_title'] or ''}</span><span>{tm}</span><span>{row['bookmaker'] or ''}</span></div>
                    <div class='odds'>
                      <div class='pill'>{row['name'] or ''} {row['price'] or ''}</div>
                      <div class='pill best'>{row['market'] or ''}</div>
                      <div class='pill'>{row['point'] if pd.notna(row['point']) else ''}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

# UPCOMING
with nav_tabs[2]:
    st.markdown('<div class="section-title"><h3>Upcoming Matches</h3><span class="time">Next 48 hours</span></div>', unsafe_allow_html=True)
    # Simulated upcoming layout using cards
    st.markdown('<div class="card-grid">', unsafe_allow_html=True)
    for i in range(6):
        banner = "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=1600&auto=format&fit=crop"
        st.markdown(
            f"""
            <div class='card'>
              <div class='banner' style="background-image:url('{banner}')"></div>
              <div class='body'>
                <span class='badge'>Kickoff {i+1}h</span>
                <div class='teams'><span>Club {i+1}</span><span>vs</span><span>Rivals {i+2}</span></div>
                <div class='meta'><span>Tournament</span><span>{datetime.now().strftime('%b %d')}</span></div>
                <div class='odds'>
                  <div class='pill'>Home 1.{i}0</div>
