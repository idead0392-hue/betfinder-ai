#!/usr/bin/env python3
"""
Test script to verify render_props function works correctly
"""

import sys
import pandas as pd
from props_data_fetcher import props_fetcher

def test_render_props_logic():
    """Test the render_props logic without Streamlit"""
    print("=" * 60)
    print("Testing render_props logic")
    print("=" * 60)
    
    # Fetch props
    print("\n1. Fetching props from PrizePicks...")
    props = props_fetcher.fetch_prizepicks_props(max_items=50)
    print(f"   ✓ Fetched {len(props)} props")
    
    if not props:
        print("   ✗ No props available")
        return False
    
    # Test data structure conversion (what render_props does)
    print("\n2. Converting to DataFrame format...")
    df_data = []
    for prop in props[:10]:  # Test with first 10
        row = {
            'Player': prop.get('player_name', 'N/A'),
            'Sport': prop.get('sport', 'N/A'),
            'League': prop.get('league', 'N/A'),
            'Stat': prop.get('stat_type', 'N/A'),
            'Line': prop.get('line', 0),
            'Team': prop.get('team', 'N/A'),
            'Matchup': prop.get('matchup', 'N/A'),
            'Over/Under': prop.get('over_under', 'N/A'),
            'Confidence': f"{prop.get('confidence', 0):.1f}%",
            'Odds': prop.get('odds', -110),
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print(f"   ✓ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Display sample data
    print("\n3. Sample DataFrame structure:")
    print(df.head(3).to_string())
    
    # Test sports breakdown logic
    print("\n4. Sports breakdown:")
    sports_count = {}
    for prop in props:
        sport = prop.get('sport', 'Unknown')
        if not sport:  # Handle empty sport field
            league = prop.get('league', 'Unknown')
            sport = league if league else 'Unknown'
        sports_count[sport] = sports_count.get(sport, 0) + 1
    
    for sport, count in sorted(sports_count.items()):
        print(f"   - {sport}: {count} props")
    
    # Test filtering logic
    print("\n5. Testing filter logic...")
    
    # Sport filter
    basketball_props = [p for p in props if 'basketball' in str(p.get('sport', '')).lower() or 'nba' in str(p.get('league', '')).lower()]
    print(f"   - Basketball/NBA props: {len(basketball_props)}")
    
    # Stat filter
    points_props = [p for p in props if 'points' in str(p.get('stat_type', '')).lower()]
    print(f"   - Points props: {len(points_props)}")
    
    # Player search
    player_name = 'garland'
    player_props = [p for p in props if player_name.lower() in str(p.get('player_name', '')).lower()]
    print(f"   - Props for '{player_name}': {len(player_props)}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        success = test_render_props_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
