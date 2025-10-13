#!/usr/bin/env python3
"""
Test script for PrizePicks integration with player_prop_predictor.py
"""

import sys
from props_data_fetcher import PropsDataFetcher, props_fetcher

def test_props_fetcher():
    """Test fetching props from PrizePicks"""
    print("=" * 60)
    print("Testing PropsDataFetcher")
    print("=" * 60)
    
    # Fetch props
    print("\n1. Fetching props from PrizePicks...")
    props = props_fetcher.fetch_prizepicks_props(max_items=10)
    
    print(f"   ✓ Fetched {len(props)} props")
    
    if props:
        print("\n2. Sample prop data:")
        sample = props[0]
        for key, value in sample.items():
            print(f"   - {key}: {value}")
        
        # Check required fields
        print("\n3. Validating required fields...")
        required_fields = ['player_name', 'stat_type', 'line', 'sport', 'league']
        for field in required_fields:
            if field in sample:
                print(f"   ✓ {field}: present")
            else:
                print(f"   ✗ {field}: MISSING")
        
        # Group by sport
        print("\n4. Props by sport:")
        sports_count = {}
        for prop in props:
            sport = prop.get('sport', 'Unknown')
            sports_count[sport] = sports_count.get(sport, 0) + 1
        
        for sport, count in sorted(sports_count.items()):
            print(f"   - {sport}: {count}")
    else:
        print("   ⚠ No props fetched")
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
    
    return len(props) > 0

if __name__ == '__main__':
    success = test_props_fetcher()
    sys.exit(0 if success else 1)
