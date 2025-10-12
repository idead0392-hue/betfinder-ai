#!/usr/bin/env python3
"""
Examine PrizePicks API for team/matchup data structure
"""
import requests
import json
import time
import random

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def examine_full_structure():
    session = requests.Session()
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'Origin': 'https://app.prizepicks.com',
        'Referer': 'https://app.prizepicks.com/'
    }
    
    try:
        response = session.get('https://partner-api.prizepicks.com/projections', headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Build lookup for included objects
            included = {obj['id']: obj for obj in data.get('included', []) if isinstance(obj, dict) and obj.get('type')}
            
            print('=== SAMPLE TEAM OBJECT ===')
            team_objects = [obj for obj in data.get('included', []) if obj.get('type') == 'team']
            if team_objects:
                print(json.dumps(team_objects[0], indent=2))
            
            print('\n=== LOOKING FOR GAME OBJECTS ===')
            game_objects = [obj for obj in data.get('included', []) if obj.get('type') == 'game']
            if game_objects:
                print(f"Found {len(game_objects)} game objects")
                print("Sample game object:")
                print(json.dumps(game_objects[0], indent=2)[:1000])
            else:
                print("No game objects found in included data")
            
            print('\n=== SAMPLE PROJECTION WITH RELATIONSHIPS ===')
            if data.get('data'):
                sample_proj = data['data'][0]
                print(json.dumps(sample_proj, indent=2)[:1000])
                
                # Check game relationship
                game_rel = sample_proj.get('relationships', {}).get('game', {}).get('data', {})
                if game_rel and game_rel.get('id'):
                    game_id = game_rel['id']
                    print(f"\nLooking for game ID: {game_id}")
                    if game_id in included:
                        print("Found corresponding game object!")
                        print(json.dumps(included[game_id], indent=2))
                    else:
                        print("Game ID not found in included objects")
                        
        else:
            print(f"API request failed with status: {response.status_code}")
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    examine_full_structure()