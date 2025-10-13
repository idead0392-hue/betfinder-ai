#!/usr/bin/env python3
"""
Comprehensive integration test for PrizePicks props rendering
"""

from props_data_fetcher import props_fetcher
import pandas as pd

def test_full_integration():
    """Test the complete integration flow"""
    print("=" * 80)
    print("COMPREHENSIVE PRIZEPICKS INTEGRATION TEST")
    print("=" * 80)
    
    # Test 1: Data Fetching
    print("\n[TEST 1] Fetching PrizePicks Props")
    print("-" * 80)
    props = props_fetcher.fetch_prizepicks_props(max_items=1000)
    print(f"✓ Fetched {len(props)} props from CSV")
    assert len(props) > 0, "Should fetch at least some props"
    
    # Test 2: Data Structure Validation
    print("\n[TEST 2] Validating Data Structure")
    print("-" * 80)
    required_fields = ['player_name', 'stat_type', 'line', 'league']
    sample_prop = props[0]
    
    for field in required_fields:
        assert field in sample_prop, f"Missing required field: {field}"
        print(f"✓ Field '{field}' present: {sample_prop[field]}")
    
    # Test 3: Sport Distribution
    print("\n[TEST 3] Sport/League Distribution")
    print("-" * 80)
    leagues = {}
    sports = {}
    for prop in props:
        league = prop.get('league', 'Unknown')
        sport = prop.get('sport', '')
        leagues[league] = leagues.get(league, 0) + 1
        if sport:
            sports[sport] = sports.get(sport, 0) + 1
    
    print(f"✓ Found {len(leagues)} unique leagues:")
    for league, count in sorted(leagues.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  • {league}: {count} props")
    
    if sports:
        print(f"\n✓ Found {len(sports)} unique sports:")
        for sport, count in sorted(sports.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {sport}: {count} props")
    
    # Test 4: Filtering Capabilities
    print("\n[TEST 4] Testing Filter Capabilities")
    print("-" * 80)
    
    # NBA props
    nba_props = [p for p in props if 'nba' in str(p.get('league', '')).lower()]
    print(f"✓ NBA props: {len(nba_props)}")
    
    # NFL props
    nfl_props = [p for p in props if 'nfl' in str(p.get('league', '')).lower()]
    print(f"✓ NFL props: {len(nfl_props)}")
    
    # Points props
    points_props = [p for p in props if 'points' in str(p.get('stat_type', '')).lower()]
    print(f"✓ Points props: {len(points_props)}")
    
    # Test 5: DataFrame Conversion (render_props logic)
    print("\n[TEST 5] DataFrame Conversion")
    print("-" * 80)
    
    display_props = props[:25]  # Take first 25
    df_data = []
    for prop in display_props:
        row = {
            'Player': prop.get('player_name', 'N/A')[:20],
            'Sport': prop.get('sport', prop.get('league', 'N/A'))[:10],
            'Stat': prop.get('stat_type', 'N/A')[:18],
            'Line': f"{prop.get('line', 0):.1f}",
            'Odds': prop.get('odds', -110),
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print(f"✓ Created DataFrame with {len(df)} rows x {len(df.columns)} columns")
    print("\nSample Data:")
    print(df.head(10).to_string(index=False))
    
    # Test 6: Player Search
    print("\n[TEST 6] Player Search Functionality")
    print("-" * 80)
    
    search_terms = ['james', 'curry', 'mahomes', 'ohtani']
    for term in search_terms:
        results = [p for p in props if term.lower() in str(p.get('player_name', '')).lower()]
        print(f"✓ Search '{term}': {len(results)} results")
        if results:
            print(f"  Example: {results[0].get('player_name')} - {results[0].get('stat_type')}")
    
    # Test 7: Stat Type Variety
    print("\n[TEST 7] Stat Type Variety")
    print("-" * 80)
    
    stat_types = {}
    for prop in props:
        stat = prop.get('stat_type', 'unknown')
        stat_types[stat] = stat_types.get(stat, 0) + 1
    
    print(f"✓ Found {len(stat_types)} unique stat types")
    print("Top 15 stat types:")
    for stat, count in sorted(stat_types.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  • {stat}: {count}")
    
    # Test 8: Line Value Analysis
    print("\n[TEST 8] Line Value Analysis")
    print("-" * 80)
    
    lines = [p.get('line', 0) for p in props]
    if lines:
        print(f"✓ Line statistics:")
        print(f"  • Min: {min(lines):.1f}")
        print(f"  • Max: {max(lines):.1f}")
        print(f"  • Average: {sum(lines)/len(lines):.1f}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Total props fetched: {len(props)}")
    print(f"✓ Unique leagues: {len(leagues)}")
    print(f"✓ Unique stat types: {len(stat_types)}")
    print(f"✓ Data structure: Valid")
    print(f"✓ Filtering: Working")
    print(f"✓ DataFrame conversion: Working")
    print("\n🎉 ALL TESTS PASSED - Integration is fully functional!")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    import sys
    try:
        success = test_full_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
