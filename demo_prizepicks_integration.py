#!/usr/bin/env python3
"""
Demo script showing PrizePicks integration in action
This simulates what the Streamlit app would display
"""

from props_data_fetcher import props_fetcher
import pandas as pd

def demo_prizepicks_integration():
    print("=" * 70)
    print("PrizePicks Integration Demo for player_prop_predictor.py")
    print("=" * 70)
    
    # Fetch props
    print("\n📊 Fetching props from PrizePicks CSV...")
    props = props_fetcher.fetch_prizepicks_props(max_items=100)
    print(f"✓ Successfully fetched {len(props)} props")
    
    if not props:
        print("✗ No props available")
        return
    
    # Display sports breakdown
    print("\n📈 Sports Breakdown:")
    sports_count = {}
    for prop in props:
        sport = prop.get('sport', '')
        league = prop.get('league', 'Unknown')
        # Use league if sport is empty
        display_sport = sport if sport else league
        sports_count[display_sport] = sports_count.get(display_sport, 0) + 1
    
    for sport, count in sorted(sports_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {sport}: {count} props")
    
    # Create DataFrame
    print(f"\n📋 Sample Props (showing 20 of {len(props)}):")
    df_data = []
    for prop in props[:20]:
        row = {
            'Player': prop.get('player_name', 'N/A')[:25],
            'Sport/League': prop.get('league', prop.get('sport', 'N/A')),
            'Stat': prop.get('stat_type', 'N/A')[:20],
            'Line': prop.get('line', 0),
            'Matchup': prop.get('matchup', 'N/A')[:30],
            'Confidence': f"{prop.get('confidence', 0):.0f}%",
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Filter examples
    print("\n🔍 Filter Examples:")
    
    # Filter by league/sport
    tennis_props = [p for p in props if 'tennis' in str(p.get('league', '')).lower()]
    print(f"  • TENNIS props: {len(tennis_props)}")
    
    nba_props = [p for p in props if 'nba' in str(p.get('league', '')).lower()]
    print(f"  • NBA props: {len(nba_props)}")
    
    nfl_props = [p for p in props if 'nfl' in str(p.get('league', '')).lower()]
    print(f"  • NFL props: {len(nfl_props)}")
    
    # Filter by stat type
    points_props = [p for p in props if 'points' in str(p.get('stat_type', '')).lower()]
    print(f"  • Points props: {len(points_props)}")
    
    rebounds_props = [p for p in props if 'rebound' in str(p.get('stat_type', '')).lower()]
    print(f"  • Rebounds props: {len(rebounds_props)}")
    
    # Sample detailed prop
    if props:
        print("\n📌 Sample Detailed Prop:")
        sample = props[0]
        print(f"  Player: {sample.get('player_name')}")
        print(f"  League: {sample.get('league')}")
        print(f"  Stat: {sample.get('stat_type')}")
        print(f"  Line: {sample.get('line')}")
        print(f"  Team: {sample.get('team')}")
        print(f"  Matchup: {sample.get('matchup')}")
        print(f"  Confidence: {sample.get('confidence')}%")
        print(f"  Odds: {sample.get('odds')}")
    
    print("\n" + "=" * 70)
    print("✓ Demo completed - Integration working correctly!")
    print("=" * 70)
    print("\nTo use in Streamlit:")
    print("1. Run: streamlit run player_prop_predictor.py")
    print("2. Select 'PrizePicks' from the Data Source dropdown")
    print("3. View live props with interactive filters")

if __name__ == '__main__':
    demo_prizepicks_integration()
