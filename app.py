import streamlit as st

# Tab-to-CSV value mapping for filtering
tab_csv_map = {
    "Football": "NFL",
    "Basketball": "NBA",
    "CSGO": "Counter-Strike",
    "League of Legends": "LoL",
    "Dota2": "Dota 2",
    # Add more mappings as needed based on your diagnostics
}
import pandas as pd
import os
import datetime
import logging
from typing import Dict, List, Any
import hashlib
import streamlit as st
import pandas as pd
import os

# Tab Names = Agent Names/Sports
tab_names = [
    "Home", "Football", "Basketball", "Tennis", "Baseball", "Hockey",
    "Soccer", "College Football", "CSGO", "League of Legends",
    "Dota2", "Valorant", "Apex Legends", "Rocket League"
]

csv_mapping = {
    "Football": "fanduel_nfl_props.csv",
    "Basketball": "prizepicks_props.csv",
    "Tennis": "prizepicks_props.csv",
    "Baseball": "prizepicks_props.csv",
    "Hockey": "prizepicks_props.csv",
    "Soccer": "prizepicks_props.csv",
    "College Football": "prizepicks_props.csv",
    "CSGO": "prizepicks_props.csv",
    "League of Legends": "prizepicks_props.csv",
    "Dota2": "prizepicks_props.csv",
    "Valorant": "prizepicks_props.csv",
    "Apex Legends": "prizepicks_props.csv",
    "Rocket League": "prizepicks_props.csv",
}

tabs = st.tabs(tab_names)

with tabs[0]:
    st.title("🏠 Home - BetFinder AI")
    st.markdown("Monitor real-time prop betting analytics across all sports.")
    st.markdown("Select a tab for your favorite sport or game.")

def render_fanduel_card(row, sport):
    # Fanduel style: white panel, colored accent bar, horizontal layout.
    # Conditional extras for NFL, CSGO shown as badge.
    player, team, matchup, prop_type, line, odds, extra, logo = '', '', '', '', '', '', '', ''

    if sport == "Football":
        player = row.get('player', row.get('team', 'N/A'))
        matchup = row.get('teams', f"{row.get('team', 'N/A')} vs {row.get('opponent', '')}")
        position = row.get('position', '')
        prop_type = row.get('prop_type', row.get('prop_name', ''))
        line = row.get('line', row.get('value', ''))
        odds = row.get('odds', '')
        extra = f"<span class='badge'>{position}</span>" if position else ""

    elif sport == "CSGO":
        player = row.get('player', row.get('team', 'N/A'))
        matchup = row.get('match', row.get('teams', ''))
        prop_type = row.get('prop_type', row.get('stat_type', ''))
        line = row.get('line', row.get('value', ''))
        odds = row.get('odds', '')
        map_name = row.get('map', '')
        extra = f"<span class='badge'>{map_name}</span>" if map_name else ""

    else: # Default for other sports/agents
        player = row.get('player', row.get('team', 'N/A'))
        matchup = row.get('teams', row.get('match', 'N/A'))
        prop_type = row.get('prop_type', row.get('prop_name', ''))
        line = row.get('line', row.get('value', ''))
        odds = row.get('odds', '')

    odds_color = "#25C16F" if str(odds).startswith('+') else "#E35252"  # green: positive, red: negative

    st.markdown(
        f"""
        <div style="
            background:white;
            border-radius:14px;
            box-shadow:0 2px 6px #E8E8E8;
            margin-bottom:18px;
            border:1px solid #E8E8E8;
            padding:20px;
            display:flex;
            align-items:center;
            justify-content:space-between;
        ">
            <div style='flex:2'>
              <div style="font-weight:600;font-size:19px">{player}</div>
              <div style="color:#555;font-size:15px">{matchup} {extra}</div>
              <div style="margin-top:7px;font-size:14px;color:#777">
                <b>{prop_type}</b> &nbsp; <span style="color:#333;font-weight:600"> {line} </span>
              </div>
            </div>
            <div style='flex:0.7;text-align:right'>
                <div style="
                    padding:7px 17px;
                    background:{odds_color};
                    color:white;
                    border-radius:12px;
                    display:inline-block;
                    font-weight:600;
                    font-size:18px;
                  ">
                  {odds}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

def universal_prop_card(row):
    name = row.get("Name", row.get("Player", row.get("Team", "N/A")))
    team = row.get("Team", "")
    matchup = row.get("Matchup", row.get("matchup", ""))
    prop = row.get("Prop", row.get("Prop Type", row.get("prop_type", "")))
    points = row.get("Points", row.get("Value", row.get("line", "")))
    home_team = row.get("Home_Team", "")
    away_team = row.get("Away_Team", "")
    game_date = row.get("Game_Date", "")
    game_time = row.get("Game_Time", "")
    last_updated = row.get("Last_Updated", "")
    allow_under = row.get("Allow_Under", False)
    under_badge = (
        "<span style='background: #48BB78; color: #fff; border-radius: 4px; padding: 2px 8px; font-size: 12px;'>UNDER ALLOWED</span>"
        if str(allow_under).lower() in ("true", "yes", "1") else ""
    )
    st.markdown(
        f"""
        <div style=\"background:#fff; border-radius:10px; box-shadow:0 2px 8px #e3e9ee; margin-bottom:20px; border:1px solid #e0e7ef; padding:18px 20px; max-width:520px;\">
            <div style=\"display:flex;align-items:center;justify-content:space-between;\">
                <div>
                    <div style=\"font-weight:600;font-size:21px;color:#364A63;\">{name}</div>
                    <div style=\"color:#7A93B2;font-size:14px;margin-bottom:4px;\">{team}</div>
                    <div style=\"color:#7A93B2;font-size:13px;\">{matchup}</div>
                </div>
                <div style=\"text-align:right;\">
                    <div style=\"background:#F1F3F8;border-radius:7px;padding:7px 17px;font-size:17px;font-weight:600;color:#536485;\">
                        {points} {prop}
                    </div>
                </div>
            </div>
            <div style=\"margin-top:7px;display:flex;flex-direction:row;gap:12px;font-size:13px;color:#645A9F;\">
                <span>🏠 {home_team}</span>
                <span>🆚</span>
                <span>🏃‍♂️ {away_team}</span>
            </div>
            <div style=\"margin-top:10px;font-size:12px;color:#97A0B4;\">
                <span>📅 {game_date}</span>
                <span style=\"margin-left:15px;\">⏰ {game_time}</span>
                <span style=\"margin-left:15px;\">{under_badge}</span>
                <span style=\"margin-left:15px;color:#C3C3C3;\">Last updated:</span>
                <span style=\"margin-left:5px;color:#A3AEC7;\">{last_updated}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

for idx, sport in enumerate(tab_names[1:], 1):
    with tabs[idx]:
        st.header(f"{sport} Props")
        csv_path = csv_mapping[sport]
        if os.path.exists(csv_path):
            props_df = pd.read_csv(csv_path)

            # Filtering logic using tab_csv_map
            col = None
            for _col in ["sport", "league"]:
                if _col in props_df.columns:
                    col = _col
                    break

            key_value = tab_csv_map.get(sport, sport)
            if col and col in props_df.columns:
                filtered = props_df[props_df[col].str.lower() == key_value.lower()]
            else:
                filtered = props_df

            st.markdown(f"**Source:** `{csv_path}`")
            st.metric("Total Props", len(props_df))
            st.metric("Last Updated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))

            display_count = 20
            num_columns = 2  # You can change this to 3 or more for smaller cards

            with st.spinner("Loading props..."):
                if not filtered.empty:
                    cols = st.columns(num_columns)
                    for i, (_, row) in enumerate(filtered.head(display_count).iterrows()):
                        current_col = cols[i % num_columns]
                        with current_col:
                            universal_prop_card(row)
                    if len(filtered) > display_count:
                        st.info(f"Showing first {display_count} props out of {len(filtered)} total.")
                else:
                    st.info(f"No {sport} props available at this time.")

        else:
            st.error(f"CSV file '{csv_path}' not found for {sport} agent.")

st.markdown("---")
st.caption("© 2025 BetFinder AI | All agents mapped. Tabs are auto-updating in real time.")
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_unique_key(player: str, stat: str, idx: int, sport_key: str) -> str:
    """
    Generate a unique key for Streamlit widgets using all prop attributes.
    This ensures no duplicate keys even if the same player appears multiple times.
    """
    # Create a unique string using all identifying information
    full_str = f"{sport_key}_{player}_{stat}_{idx}"
    # Hash it to create a short, unique key
    hash_suffix = hashlib.md5(full_str.encode()).hexdigest()[:12]
    return f"btn_{hash_suffix}"

# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_compact_prop_row(player: str, stat: str, line: float, league: str, idx: int, sport_key: str = ""):
    """
    Render a single prop row with proper unique keys.
    """
    cols = st.columns([3, 2, 1, 1])
    
    with cols[0]:
        st.write(f"**{player}**")
    
    with cols[1]:
        st.write(f"{stat}: {line}")
    
    with cols[2]:
        st.write(league)
    
    with cols[3]:
        # Generate unique key using all prop attributes
        unique_key = make_unique_key(player, stat, idx, sport_key)
        if st.button("📋", key=unique_key, help=f"Copy: {player} - {stat} {line}"):
            st.success(f"✅ Copied: {player} - {stat} {line}")

def display_sport_picks(sport_name: str, picks: List[Dict], sport_emoji: str, sport_key: str = None):
    """
    Display picks for a specific sport with unique button keys.
    """
    if not sport_key:
        sport_key = sport_name.lower().replace(" ", "_")
    
    st.subheader(f"{sport_emoji} {sport_name}")
    
    if not picks:
        st.info(f"No {sport_name} picks available")
        return
    
    # Display each pick
    for idx, pick in enumerate(picks):
        player = pick.get('player_name', 'Unknown')
        stat_type = pick.get('stat_type', '')
        line = pick.get('line_score', 0)
        league = pick.get('league', sport_name)
        
        # Render row with unique key incorporating all attributes
        render_compact_prop_row(player, stat_type, line, league, idx, sport_key=sport_key)

# ============================================================================
# MAIN APP
# ============================================================================

st.set_page_config(
    page_title="BetFinder AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 BetFinder AI - Prop Finder")
st.markdown("### Real-time props from prizepicks_props.csv")

# Check if we have data
if props_df.empty:
    st.error(
        "❌ **No props data available**\n\n"
        "Please check that `prizepicks_props.csv` exists and is valid."
    )
    st.stop()

# ============================================================================
# DISPLAY DATA STATUS
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Props", len(props_df))

with col2:
    if 'league' in props_df.columns:
        st.metric("Sports", props_df['league'].nunique())
    elif 'sport' in props_df.columns:
        st.metric("Sports", props_df['sport'].nunique())
    else:
        st.metric("Sports", "N/A")

with col3:
    if os.path.exists(PROPS_CSV_PATH):
        mtime = os.path.getmtime(PROPS_CSV_PATH)
        last_update = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        st.metric("Last Updated", last_update)

# ============================================================================
# DISPLAY PROPS BY SPORT
# ============================================================================

st.markdown("---")

league_col = 'league' if 'league' in props_df.columns else ('sport' if 'sport' in props_df.columns else None)
if league_col:
    for sport in sorted(props_df[league_col].unique()):
        # For NFL/Football, show only FanDuel props
        if sport.lower() in ["football", "nfl"] and not fanduel_nfl_df.empty:
            st.markdown("## 🏈 NFL Props (FanDuel)")
            fanduel_props = fanduel_nfl_df.to_dict('records')
            display_sport_picks("NFL (FanDuel)", fanduel_props, "🏈", sport_key="nfl_fanduel")
            st.markdown("---")
        else:
            # Filter props for this sport
            sport_props = props_df[props_df[league_col] == sport].to_dict('records')
            sport_key = sport.lower().replace(" ", "_").replace("-", "_")
            display_sport_picks(sport, sport_props, "🏆", sport_key=sport_key)
            st.markdown("---")
else:
    st.warning("No 'league' or 'sport' column found in props data. Unable to display sports tabs.")

print("[DIAGNOSTIC] ✅ app.py rendering complete")
