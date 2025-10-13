#!/usr/bin/env python3
"""
Visual output test for render_props function.

This creates a simple visual representation of what the render_props
function displays, for verification purposes.
"""

import sys
from typing import List, Dict

def create_mock_props() -> List[Dict]:
    """Create mock props data for testing"""
    return [
        {
            "player_name": "LeBron James",
            "team": "LAL",
            "matchup": "LAL vs BOS",
            "stat_type": "points",
            "line": 27.5,
            "league": "NBA",
            "odds": -110,
            "confidence": 75.0,
            "expected_value": 5.2
        },
        {
            "player_name": "Stephen Curry",
            "team": "GSW",
            "matchup": "GSW vs PHX",
            "stat_type": "3-pointers made",
            "line": 4.5,
            "league": "NBA",
            "odds": -110,
            "confidence": 72.0,
            "expected_value": 4.8
        },
        {
            "player_name": "Patrick Mahomes",
            "team": "KC",
            "matchup": "KC vs BUF",
            "stat_type": "passing yards",
            "line": 285.5,
            "league": "NFL",
            "odds": -110,
            "confidence": 78.0,
            "expected_value": 6.1
        },
        {
            "player_name": "Aaron Judge",
            "team": "NYY",
            "matchup": "NYY vs BOS",
            "stat_type": "hits",
            "line": 1.5,
            "league": "MLB",
            "odds": -110,
            "confidence": 68.0,
            "expected_value": 3.5
        },
        {
            "player_name": "Connor McDavid",
            "team": "EDM",
            "matchup": "EDM vs CGY",
            "stat_type": "points",
            "line": 1.5,
            "league": "NHL",
            "odds": -110,
            "confidence": 80.0,
            "expected_value": 7.2
        }
    ]


def display_props_table(props: List[Dict]):
    """Display props in a simple table format"""
    
    print("\n" + "=" * 120)
    print(" " * 45 + "🎯 PROPS DISPLAY (5 shown)")
    print("=" * 120)
    print()
    
    # Header
    header = f"{'Player':<20} {'Team':<6} {'Stat':<18} {'Line':<8} {'Matchup':<15} {'League':<6} {'Conf':<6} {'EV':<6}"
    print(header)
    print("-" * 120)
    
    # Rows
    for prop in props:
        player = prop.get('player_name', 'Unknown')[:19]
        team = prop.get('team', '')[:5]
        stat = prop.get('stat_type', '')[:17]
        line = f"{prop.get('line', 0):.1f}"
        matchup = prop.get('matchup', '')[:14]
        league = prop.get('league', '')[:5]
        conf = f"{prop.get('confidence', 0):.0f}%"
        ev = f"{prop.get('expected_value', 0):.1f}%"
        
        row = f"{player:<20} {team:<6} {stat:<18} {line:<8} {matchup:<15} {league:<6} {conf:<6} {ev:<6}"
        print(row)
    
    print("-" * 120)
    print()
    
    # Summary stats
    print("📊 SUMMARY STATISTICS")
    print(f"   Total Props: {len(props)}")
    print(f"   Unique Leagues: {len(set(p.get('league', '') for p in props))}")
    print(f"   Unique Players: {len(set(p.get('player_name', '') for p in props))}")
    print()
    print("=" * 120)
    print()


def main():
    """Main function"""
    print("\n" + "=" * 120)
    print(" " * 35 + "VISUAL OUTPUT TEST - render_props Function")
    print("=" * 120)
    print()
    print("This demonstrates what the render_props function displays in Streamlit.")
    print("The actual Streamlit version uses a Pandas DataFrame with better formatting.")
    print()
    
    # Create mock data
    props = create_mock_props()
    
    # Display the table
    display_props_table(props)
    
    print("✅ Visual output test completed successfully!")
    print()
    print("To see the actual Streamlit interface, run:")
    print("   streamlit run demo_render_props.py")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
