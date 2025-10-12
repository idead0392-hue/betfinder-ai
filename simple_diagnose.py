#!/usr/bin/env python3
"""
Simple diagnostic script to check prop data quality
"""

import pandas as pd

def validate_prop_consistency(prop_data: dict) -> bool:
    """Simple validation without Streamlit dependencies"""
    player_name = str(prop_data.get('player_name', '')).lower()
    stat_type = str(prop_data.get('stat_type', '')).lower()
    team = str(prop_data.get('team', '')).lower()
    matchup = str(prop_data.get('matchup', '')).lower()
    league = str(prop_data.get('league', '')).lower()
    
    # Skip props with missing essential data
    if not player_name or not stat_type:
        return False
    
    # Skip if team is completely inconsistent with matchup
    if team and matchup and '@' in matchup:
        parts = matchup.split('@')
        if len(parts) == 2:
            away_team = parts[0].strip().lower()
            home_team = parts[1].strip().lower()
            
            if team not in away_team and team not in home_team and away_team not in team and home_team not in team:
                # Check abbreviation mappings
                team_abbrevs = {
                    'new england patriots': 'ne', 'new orleans saints': 'no',
                    'chicago bears': 'chi', 'washington commanders': 'was',
                    'carolina panthers': 'car', 'arizona cardinals': 'ari',
                    'indianapolis colts': 'ind', 'los angeles chargers': 'lac',
                    'miami dolphins': 'mia'
                }
                
                team_full_name = None
                for full, abbrev in team_abbrevs.items():
                    if abbrev == team or team in full:
                        team_full_name = full
                        break
                
                if team_full_name and team_full_name not in matchup:
                    return False
                elif not team_full_name:
                    return False
    
    return True

def main():
    # Load CSV
    df = pd.read_csv('/workspaces/betfinder-ai/prizepicks_props.csv')
    print(f"📊 Total props in CSV: {len(df)}")
    
    # Convert to prop format and validate
    valid_props = []
    invalid_props = []
    
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
        
        if validate_prop_consistency(prop):
            valid_props.append(prop)
        else:
            invalid_props.append(prop)
    
    print(f"✅ Valid props: {len(valid_props)}")
    print(f"❌ Invalid props: {len(invalid_props)}")
    
    # Show examples of invalid props
    print("\n🔍 Sample invalid props:")
    for i, prop in enumerate(invalid_props[:5]):
        print(f"  {i+1}. {prop['player_name']} - {prop['stat_type']} - Team: {prop['team']} - Matchup: {prop['matchup']}")
    
    # Show league distribution
    print("\n📈 League distribution:")
    league_counts = {}
    for prop in valid_props:
        league = prop['league']
        league_counts[league] = league_counts.get(league, 0) + 1
    
    for league, count in sorted(league_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {league}: {count}")

if __name__ == "__main__":
    main()