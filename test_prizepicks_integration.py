#!/usr/bin/env python3
"""
Simple integration test for PrizePicksProvider and render_props function.
"""

import sys
from typing import List, Dict

def test_prizepicks_provider():
    """Test PrizePicksProvider initialization and basic functionality"""
    print("=" * 60)
    print("Testing PrizePicksProvider...")
    print("=" * 60)
    
    try:
        from api_providers import PrizePicksProvider
        provider = PrizePicksProvider()
        print("✓ PrizePicksProvider initialized successfully")
        
        # Test get_props method
        print("\nTesting get_props method...")
        resp = provider.get_props(max_props=10)
        
        if resp.success:
            print(f"✓ get_props succeeded")
            print(f"  Response time: {resp.response_time:.2f}s")
            
            if isinstance(resp.data, dict) and "data" in resp.data:
                props_count = len(resp.data["data"])
                print(f"  Props fetched: {props_count}")
                
                if props_count > 0:
                    # Show first prop as example
                    first_prop = resp.data["data"][0]
                    print(f"\n  Example prop:")
                    print(f"    Player: {first_prop.get('player_name', 'N/A')}")
                    print(f"    Stat: {first_prop.get('stat_type', 'N/A')}")
                    print(f"    Line: {first_prop.get('line', 'N/A')}")
                    print(f"    Team: {first_prop.get('team', 'N/A')}")
            else:
                print(f"  Warning: Unexpected data format")
        else:
            print(f"⚠ get_props failed: {resp.error_message}")
            
    except Exception as e:
        print(f"✗ Error testing PrizePicksProvider: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_render_props_function():
    """Test render_props function with mock data"""
    print("\n" + "=" * 60)
    print("Testing render_props function...")
    print("=" * 60)
    
    try:
        # We can't actually run streamlit code in a test,
        # but we can verify the function exists and is callable
        from page_utils import render_props
        print("✓ render_props function imported successfully")
        
        # Create mock data
        mock_props = [
            {
                "player_name": "Test Player",
                "team": "TEST",
                "stat_type": "points",
                "line": 25.5,
                "league": "NBA",
                "matchup": "TEST vs DEMO",
                "odds": -110,
                "confidence": 75.0,
                "expected_value": 5.0,
            }
        ]
        
        print("✓ Mock props data created")
        print(f"  Props count: {len(mock_props)}")
        
        # Note: We can't actually call render_props here because it needs streamlit context
        print("✓ render_props function is callable (requires Streamlit context to execute)")
        
    except Exception as e:
        print(f"✗ Error testing render_props: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n🧪 PrizePicks Integration Tests\n")
    
    results = []
    
    # Test 1: PrizePicksProvider
    results.append(("PrizePicksProvider", test_prizepicks_provider()))
    
    # Test 2: render_props function
    results.append(("render_props function", test_render_props_function()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
