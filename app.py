import streamlit as st
import pandas as pd
import os
import datetime
import logging
from typing import Dict, List, Any
import hashlib

# ============================================================================
# DIAGNOSTIC: Print at module load
# ============================================================================
print("[DIAGNOSTIC] app.py module loading...")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROPS_CSV_PATH = "prizepicks_props.csv"

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_props_data():
    """
    Load props from CSV file with proper error handling.
    Returns real data only - no sample fallback.
    This is the ONLY place props should be loaded from.
    """
    print(f"[DIAGNOSTIC] Attempting to load props from: {PROPS_CSV_PATH}")
    logger.info(f"Loading props from {PROPS_CSV_PATH}")
    
    # Check if file exists
    if not os.path.exists(PROPS_CSV_PATH):
        error_msg = f"❌ Props file not found: {PROPS_CSV_PATH}"
        print(f"[ERROR] {error_msg}")
        logger.error(error_msg)
        return pd.DataFrame()  # Return empty DataFrame
    
    try:
        # Load CSV
        df = pd.read_csv(PROPS_CSV_PATH)
        print(f"[DIAGNOSTIC] ✅ Successfully loaded {len(df)} props from CSV")
        logger.info(f"Loaded {len(df)} props from {PROPS_CSV_PATH}")
        
        # Basic validation
        required_cols = ['player_name', 'stat_type', 'line_score', 'league']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            error_msg = f"❌ Missing required columns: {missing}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            return pd.DataFrame()
        
        print(f"[DIAGNOSTIC] ✅ CSV validation passed")
        return df
        
    except Exception as e:
        error_msg = f"❌ Error reading props CSV: {str(e)}"
        print(f"[ERROR] {error_msg}")
        logger.error(error_msg)
        return pd.DataFrame()

# ============================================================================
# LOAD PROPS AT MODULE INITIALIZATION
# ============================================================================

print("[DIAGNOSTIC] Loading props data at module init")
props_df = load_props_data()
print(f"[DIAGNOSTIC] Loaded {len(props_df)} props total")

# Show breakdown by sport
if not props_df.empty:
    sport_groups = props_df.groupby('league')
    print("[DIAGNOSTIC] Props by sport:")
    for sport, group in sport_groups:
        print(f"  {sport}: {len(group)} props")
else:
    print("[DIAGNOSTIC] ⚠️  No props loaded - empty DataFrame")

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
    st.metric("Sports", props_df['league'].nunique())

with col3:
    if os.path.exists(PROPS_CSV_PATH):
        mtime = os.path.getmtime(PROPS_CSV_PATH)
        last_update = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        st.metric("Last Updated", last_update)

# ============================================================================
# DISPLAY PROPS BY SPORT
# ============================================================================

st.markdown("---")

# Get unique sports and display each
for sport in sorted(props_df['league'].unique()):
    # Filter props for this sport
    sport_props = props_df[props_df['league'] == sport].to_dict('records')
    
    # Create a clean sport key for unique button IDs
    sport_key = sport.lower().replace(" ", "_").replace("-", "_")
    
    # Display the sport's picks
    display_sport_picks(sport, sport_props, "🏆", sport_key=sport_key)
    st.markdown("---")

print("[DIAGNOSTIC] ✅ app.py rendering complete")
