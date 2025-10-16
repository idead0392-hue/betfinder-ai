# src/ui/app.py
import streamlit as st
import pandas as pd
import os
import datetime
import logging
from typing import Dict, List, Any
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROPS_CSV_PATH = "data/prizepicks_props.csv"

def load_props_data():
    """
    Load props from CSV file with proper error handling.
    """
    logger.info(f"Loading props from {PROPS_CSV_PATH}")
    
    if not os.path.exists(PROPS_CSV_PATH):
        error_msg = f"Props file not found: {PROPS_CSV_PATH}"
        logger.error(error_msg)
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(PROPS_CSV_PATH)
        logger.info(f"Loaded {len(df)} props from {PROPS_CSV_PATH}")
        
        required_cols = ['player_name', 'stat_type', 'line_score', 'league']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            error_msg = f"Missing required columns: {missing}"
            logger.error(error_msg)
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        error_msg = f"Error reading props CSV: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame()

def make_unique_key(player: str, stat: str, idx: int, sport_key: str) -> str:
    """
    Generate a unique key for Streamlit widgets.
    """
    full_str = f"{sport_key}_{player}_{stat}_{idx}"
    hash_suffix = hashlib.md5(full_str.encode()).hexdigest()[:12]
    return f"btn_{hash_suffix}"

def render_compact_prop_row(player: str, stat: str, line: float, league: str, idx: int, sport_key: str = ""):
    """
    Render a single prop row.
    """
    cols = st.columns([3, 2, 1, 1])
    
    with cols[0]:
        st.write(f"**{player}**")
    
    with cols[1]:
        st.write(f"{stat}: {line}")
    
    with cols[2]:
        st.write(league)
    
    with cols[3]:
        unique_key = make_unique_key(player, stat, idx, sport_key)
        if st.button("📋", key=unique_key, help=f"Copy: {player} - {stat} {line}"):
            st.success(f"Copied: {player} - {stat} {line}")

def display_sport_picks(sport_name: str, picks: List[Dict], sport_emoji: str, sport_key: str = None):
    """
    Display picks for a specific sport.
    """
    if not sport_key:
        sport_key = sport_name.lower().replace(" ", "_")
    
    st.subheader(f"{sport_emoji} {sport_name}")
    
    if not picks:
        st.info(f"No {sport_name} picks available")
        return
    
    for idx, pick in enumerate(picks):
        player = pick.get('player_name', 'Unknown')
        stat_type = pick.get('stat_type', '')
        line = pick.get('line_score', 0)
        league = pick.get('league', sport_name)
        
        render_compact_prop_row(player, stat_type, line, league, idx, sport_key=sport_key)

def main():
    st.set_page_config(
        page_title="BetFinder AI",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 BetFinder AI - Prop Finder")
    st.markdown("### Real-time props from prizepicks_props.csv")

    props_df = load_props_data()

    if props_df.empty:
        st.error(
            "**No props data available**\n\n"
            "Please check that `data/prizepicks_props.csv` exists and is valid."
        )
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Props", len(props_df))

    with col2:
        st.metric("Sports", props_df['league'].nunique())

    with col3:
        if os.path.exists(PROPS_CSV_PATH):
            mtime = os.path.getmtime(PROPS_CSV_PATH)
            last_update = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            st.metric("Last Updated", last_update)

    st.markdown("---")

    for sport in sorted(props_df['league'].unique()):
        sport_props = props_df[props_df['league'] == sport].to_dict('records')
        sport_key = sport.lower().replace(" ", "_").replace("-", "_")
        display_sport_picks(sport, sport_props, "🏆", sport_key=sport_key)
        st.markdown("---")

if __name__ == "__main__":
    main()
