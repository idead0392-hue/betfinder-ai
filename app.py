

import streamlit as st
import pandas as pd
import requests

# --- RESTORED MULTI-TAB UI ---
tab_names = [
    "Home", "Stats", "Props", "Tennis", "Basketball", "Football", "Baseball",
    "Hockey", "Soccer", "Esports", "College Football"
]
tabs = st.tabs(tab_names)

# Demo sports news DataFrame
demo_news = pd.DataFrame([
    {"headline": "LeBron James Scores 50 Points in Lakers Win", "sport": "Basketball", "summary": "LeBron James led the Lakers to a thrilling victory with a 50-point performance.", "date": "2025-10-03"},
    {"headline": "Serena Williams Announces Retirement", "sport": "Tennis", "summary": "Tennis legend Serena Williams has announced her retirement after the US Open.", "date": "2025-10-02"},
    {"headline": "Patrick Mahomes Sets Passing Record", "sport": "Football", "summary": "Chiefs QB Patrick Mahomes set a new NFL record for passing yards in a single game.", "date": "2025-10-01"},
    {"headline": "Faker Wins Another Esports Title", "sport": "Esports", "summary": "Faker continues his dominance in League of Legends with another world championship.", "date": "2025-09-30"},
    {"headline": "Lionel Messi Scores Hat-Trick for Inter Miami", "sport": "Soccer", "summary": "Messi dazzled fans with a hat-trick in last night's match.", "date": "2025-09-29"}
])

# Home Tab
with tabs[0]:
    st.header("🏟️ Sports News & Highlights")
    st.write("Stay up to date with the latest sports news, highlights, and trending stories across all major leagues.")
    st.dataframe(demo_news, width='stretch', hide_index=True)
    st.markdown("---")
    st.write("Welcome to BetFinder AI! Use the tabs above to explore player props, stats, and more.")

# Stats Tab (show /api/players table as demo and /api/player for Scottie Scheffler)
with tabs[1]:
    # --- Modern Dashboard UI ---
    st.markdown("""
        <style>
        .search-bar input {
            width: 400px;
            font-size: 1.2em;
            padding: 0.5em 1em;
            border-radius: 8px;
            border: 1px solid #333;
            background: #18181b;
            color: #fff;
        }
        .filter-bar {
            display: flex;
            gap: 1.2em;
            margin-bottom: 1.2em;
            margin-top: 0.5em;
            font-size: 1.1em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='margin-bottom:0.2em;'>Projections</h2>", unsafe_allow_html=True)
    # Search bar
    player_name = st.text_input("", value="", placeholder="Search players or teams...", key="player_search", help="Type a player or team name", label_visibility="collapsed")

    # Interactive filter bar
    leagues = ["ALL", "WNBA", "NFL", "CFB", "MLB", "TENNIS", "SOCCER", "LOL", "CS2", "VAL", "COD"]
    league_map = {
        "WNBA": ["Lakers"],
        "NFL": ["Chiefs"],
        "CFB": ["USC", "Michigan"],
        "MLB": ["Yankees", "Red Sox", "Dodgers"],
        "TENNIS": ["-"],
        "SOCCER": ["Inter Miami"],
        "LOL": ["T1"],
        "CS2": [],
        "VAL": [],
        "COD": []
    }
    selected_league = st.radio(
        label="",
        options=leagues,
        index=0,
        horizontal=True,
        key="league_filter",
        help="Filter by league or sport"
    )

    # Demo player stats DataFrame for instant loading
    demo_stats = pd.DataFrame([
        {"player": "LeBron James", "team": "Lakers", "games": 5, "points": 32.4, "assists": 8.1, "rebounds": 7.2, "hit_rate": 85},
        {"player": "Patrick Mahomes", "team": "Chiefs", "games": 4, "points": 0, "assists": 0, "rebounds": 0, "hit_rate": 78},
        {"player": "Serena Williams", "team": "-", "games": 3, "points": 0, "assists": 0, "rebounds": 0, "hit_rate": 92},
        {"player": "Lionel Messi", "team": "Inter Miami", "games": 6, "points": 5, "assists": 3, "rebounds": 0, "hit_rate": 88},
        {"player": "Faker", "team": "T1", "games": 7, "points": 0, "assists": 0, "rebounds": 0, "hit_rate": 95}
    ])
    df = demo_stats
    # Filter by league
    if selected_league != "ALL":
        teams = league_map.get(selected_league, [])
        if teams:
            df = df[df["team"].isin(teams)]
        else:
            df = df.iloc[0:0]  # Empty DataFrame for leagues with no demo data
    # Filter by search
    if player_name:
        df = df[df.apply(lambda row: player_name.lower() in str(row).lower(), axis=1)]
    def color_hit(val):
        try:
            v = float(val)
            if v >= 80:
                color = '#22c55e'  # green
            elif v >= 60:
                color = '#fde047'  # yellow
            else:
                color = '#ef4444'  # red
            return f'background-color: {color}; color: #18181b;'
        except:
            return ''
    styled = df.style.applymap(color_hit, subset=[c for c in df.columns if 'hit' in c.lower() or 'rate' in c.lower()])
    st.dataframe(styled, width='stretch', hide_index=True)

# Demo DataFrames (defined once for instant tab switching)
demo_props = pd.DataFrame([
    {"player": "John Doe", "team": "Yankees", "stat": "Hits", "line": 1.5, "over_odds": "+120", "under_odds": "-140", "event_time": "2025-10-04 19:00"},
    {"player": "Jane Smith", "team": "Red Sox", "stat": "Hits", "line": 2.5, "over_odds": "+150", "under_odds": "-110", "event_time": "2025-10-04 20:00"},
    {"player": "Alex Lee", "team": "Dodgers", "stat": "Hits", "line": 1.0, "over_odds": "+100", "under_odds": "-120", "event_time": "2025-10-04 21:00"}
])
demo_tennis = pd.DataFrame([
    {"player": "Serena Williams", "event": "US Open", "stat": "Aces", "line": 5.5, "over_odds": "+110", "under_odds": "-130", "event_time": "2025-10-05 15:00"},
    {"player": "Rafael Nadal", "event": "French Open", "stat": "Double Faults", "line": 2.5, "over_odds": "+120", "under_odds": "-140", "event_time": "2025-10-06 17:00"}
])
demo_basketball = pd.DataFrame([
    {"player": "LeBron James", "team": "Lakers", "stat": "Points", "line": 28.5, "over_odds": "+105", "under_odds": "-125", "event_time": "2025-10-07 19:30"},
    {"player": "Stephen Curry", "team": "Warriors", "stat": "3PT Made", "line": 4.5, "over_odds": "+115", "under_odds": "-135", "event_time": "2025-10-07 21:00"}
])
demo_football = pd.DataFrame([
    {"player": "Patrick Mahomes", "team": "Chiefs", "stat": "Passing Yards", "line": 305.5, "over_odds": "+100", "under_odds": "-120", "event_time": "2025-10-08 20:20"},
    {"player": "Derrick Henry", "team": "Titans", "stat": "Rushing Yards", "line": 98.5, "over_odds": "+110", "under_odds": "-130", "event_time": "2025-10-08 13:00"}
])
demo_baseball = pd.DataFrame([
    {"player": "Aaron Judge", "team": "Yankees", "stat": "Home Runs", "line": 1.5, "over_odds": "+150", "under_odds": "-110", "event_time": "2025-10-09 19:00"},
    {"player": "Mookie Betts", "team": "Dodgers", "stat": "Runs", "line": 2.5, "over_odds": "+120", "under_odds": "-140", "event_time": "2025-10-09 21:00"}
])
demo_hockey = pd.DataFrame([
    {"player": "Connor McDavid", "team": "Oilers", "stat": "Points", "line": 2.5, "over_odds": "+130", "under_odds": "-150", "event_time": "2025-10-10 19:00"},
    {"player": "Auston Matthews", "team": "Maple Leafs", "stat": "Goals", "line": 1.5, "over_odds": "+140", "under_odds": "-120", "event_time": "2025-10-10 20:00"}
])
demo_soccer = pd.DataFrame([
    {"player": "Lionel Messi", "team": "Inter Miami", "stat": "Goals", "line": 1.5, "over_odds": "+110", "under_odds": "-130", "event_time": "2025-10-11 16:00"},
    {"player": "Cristiano Ronaldo", "team": "Al Nassr", "stat": "Shots", "line": 4.5, "over_odds": "+120", "under_odds": "-140", "event_time": "2025-10-11 18:00"}
])
demo_esports = pd.DataFrame([
    {"player": "Faker", "team": "T1", "stat": "Kills", "line": 8.5, "over_odds": "+125", "under_odds": "-145", "event_time": "2025-10-12 13:00"},
    {"player": "s1mple", "team": "NAVI", "stat": "Headshots", "line": 10.5, "over_odds": "+135", "under_odds": "-115", "event_time": "2025-10-12 15:00"}
])
demo_cfb = pd.DataFrame([
    {"player": "Caleb Williams", "team": "USC", "stat": "Passing TDs", "line": 3.5, "over_odds": "+110", "under_odds": "-130", "event_time": "2025-10-13 19:00"},
    {"player": "Blake Corum", "team": "Michigan", "stat": "Rushing Yards", "line": 120.5, "over_odds": "+120", "under_odds": "-140", "event_time": "2025-10-13 21:00"}
])

# Props Tab
with tabs[2]:
    st.header("Props")
    st.subheader("AI Machine-Learned Picks")
    ai_picks = pd.DataFrame([
        {"player": "LeBron James", "team": "Lakers", "prop": "Points Over 28.5", "ai_confidence": 92, "ai_rationale": "Consistent high scoring vs. weak defense"},
        {"player": "Patrick Mahomes", "team": "Chiefs", "prop": "Passing Yards Over 305.5", "ai_confidence": 88, "ai_rationale": "Facing bottom-5 pass defense, high volume expected"},
        {"player": "Serena Williams", "team": "-", "prop": "Aces Over 5.5", "ai_confidence": 85, "ai_rationale": "Strong serve, opponent weak on return"},
        {"player": "Lionel Messi", "team": "Inter Miami", "prop": "Goals Over 1.5", "ai_confidence": 90, "ai_rationale": "Recent form, favorable matchup"},
        {"player": "Faker", "team": "T1", "prop": "Kills Over 8.5", "ai_confidence": 95, "ai_rationale": "Dominant in recent matches, high kill participation"}
    ])
    def color_conf(val):
        try:
            v = float(val)
            if v >= 90:
                color = '#22c55e'  # green
            elif v >= 80:
                color = '#fde047'  # yellow
            else:
                color = '#ef4444'  # red
            return f'background-color: {color}; color: #18181b;'
        except:
            return ''
    ai_picks_sorted = ai_picks.sort_values(by="ai_confidence", ascending=False)
    styled = ai_picks_sorted.style.applymap(color_conf, subset=["ai_confidence"])
    st.dataframe(styled, width='stretch', hide_index=True)

# Tennis Tab
with tabs[3]:
    st.header("Tennis")
    st.dataframe(demo_tennis, width='stretch', hide_index=True)

# Basketball Tab
with tabs[4]:
    st.header("Basketball")
    st.dataframe(demo_basketball, width='stretch', hide_index=True)

# Football Tab
with tabs[5]:
    st.header("Football")
    st.dataframe(demo_football, width='stretch', hide_index=True)

# Baseball Tab
with tabs[6]:
    st.header("Baseball")
    st.dataframe(demo_baseball, width='stretch', hide_index=True)

# Hockey Tab
with tabs[7]:
    st.header("Hockey")
    st.dataframe(demo_hockey, width='stretch', hide_index=True)

# Soccer Tab
with tabs[8]:
    st.header("Soccer")
    st.dataframe(demo_soccer, width='stretch', hide_index=True)

# Esports Tab
with tabs[9]:
    st.header("Esports")
    st.dataframe(demo_esports, width='stretch', hide_index=True)

# College Football Tab
with tabs[10]:
    st.header("College Football")
    st.dataframe(demo_cfb, width='stretch', hide_index=True)


    
# COLLEGE FOOTBALL TAB
with tabs[10]:
    st.write("**College Football Data (API Demo):**")
    st.info("Replace this section with actual College Football API data display.")






# TENNIS TAB
with tabs[3]:
    st.markdown('<div class="section-title">Tennis<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Tennis Data (API Demo):**")
    st.info("Replace this section with actual Tennis API data display.")

# BASKETBALL TAB
with tabs[4]:
    st.markdown('<div class="section-title">Basketball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Basketball Data (API Demo):**")
    st.info("Replace this section with actual Basketball API data display.")

# FOOTBALL TAB
with tabs[5]:
    st.markdown('<div class="section-title">Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Football Data (API Demo):**")
    st.info("Replace this section with actual Football API data display.")

# BASEBALL TAB
with tabs[6]:
    st.markdown('<div class="section-title">Baseball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Baseball Data (API Demo):**")
    st.info("Replace this section with actual Baseball API data display.")

# HOCKEY TAB
with tabs[7]:
    st.markdown('<div class="section-title">Hockey<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Hockey Data (API Demo):**")
    st.info("Replace this section with actual Hockey API data display.")

# SOCCER TAB
with tabs[8]:
    st.markdown('<div class="section-title">Soccer<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Soccer Data (API Demo):**")
    st.info("Replace this section with actual Soccer API data display.")




