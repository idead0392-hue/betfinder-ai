#!/usr/bin/env python3
"""
Diagnostic script to validate prop data and show filtering results
"""

import pandas as pd
import sys
sys.path.append('/workspaces/betfinder-ai')

from page_utils import _validate_prop_consistency, _validate_sport_category_match, _map_to_sport

def main():
    # Load CSV
    df = pd.read_csv('/workspaces/betfinder-ai/prizepicks_props.csv')
    print(f"📊 Total props in CSV: {len(df)}")
    
    # Convert to prop format and validate
    valid_props = []
    invalid_props = []
    sport_mismatches = []
    
    for _, row in df.iterrows():
        prop = {
            'player_name': row.get('Name', ''),
            'team': row.get('Team', ''),
            'stat_type': str(row.get('Prop', '')).lower(),
            'league': row.get('League', ''),
            'matchup': row.get('Matchup', ''),
            'home_team': row.get('Home_Team', ''),
            'away_team': row.get('Away_Team', '')
        }
        
        # Test consistency validation
        if _validate_prop_consistency(prop):
            # Test sport mapping and category validation
            sport = _map_to_sport(prop)
            if sport and _validate_sport_category_match(prop, sport):
                valid_props.append(prop)
            else:
                sport_mismatches.append((prop, sport))
        else:
            invalid_props.append(prop)
    
    print(f"✅ Valid props: {len(valid_props)}")
    print(f"❌ Invalid props (consistency): {len(invalid_props)}")
    print(f"🔄 Sport mismatches: {len(sport_mismatches)}")
    
    # Show examples of invalid props
    print("\n🔍 Sample invalid props (consistency issues):")
    for i, prop in enumerate(invalid_props[:5]):
        print(f"  {i+1}. {prop['player_name']} - {prop['stat_type']} - {prop['team']} vs {prop['matchup']}")
    
    # Show examples of sport mismatches  
    print("\n🔍 Sample sport mismatches:")
    for i, (prop, sport) in enumerate(sport_mismatches[:5]):
        print(f"  {i+1}. {prop['player_name']} - {prop['stat_type']} - League: {prop['league']} - Assigned: {sport}")
    
    # Show league distribution of valid props
    print("\n📈 League distribution of valid props:")
    league_counts = {}
    for prop in valid_props:
        league = prop['league']
        league_counts[league] = league_counts.get(league, 0) + 1
    
    for league, count in sorted(league_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {league}: {count}")

if __name__ == "__main__":
    main()