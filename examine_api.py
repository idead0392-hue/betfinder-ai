#!/usr/bin/env python3
"""
Quick script to examine PrizePicks API structure for team/matchup data
"""
import requests
import json
import random

def examine_api_structure():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]

    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }

    try:
        print("📡 Examining PrizePicks API structure...")
        response = requests.get('https://api.prizepicks.com/projections', headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Success! Status: {response.status_code}")
            
            # Check what types of included objects we have
            if 'included' in data:
                types_found = {}
                for obj in data['included']:
                    obj_type = obj.get('type', 'unknown')
                    if obj_type not in types_found:
                        types_found[obj_type] = []
                    types_found[obj_type].append(obj)
                
                print(f"\n🔍 Found {len(types_found)} different included object types:")
                for obj_type, objects in types_found.items():
                    print(f"  - {obj_type}: {len(objects)} objects")
                
                # Look for league/game/matchup related objects
                potential_matchup_types = ['league', 'game', 'match', 'fixture', 'event', 'contest']
                for ptype in potential_matchup_types:
                    if ptype in types_found:
                        print(f"\n🎯 Found potential matchup data in '{ptype}' objects:")
                        sample = types_found[ptype][0]
                        print(json.dumps(sample, indent=2)[:800] + "...")
                
                # Look at new_player objects for team/opponent info
                if 'new_player' in types_found:
                    print("\n👤 Sample new_player object:")
                    sample_player = types_found['new_player'][0]
                    print(json.dumps(sample_player, indent=2)[:600] + "...")
            
            # Look at a sample projection
            if 'data' in data and data['data']:
                print("\n📊 Sample projection structure:")
                sample_proj = data['data'][0]
                print(json.dumps(sample_proj, indent=2)[:600] + "...")
                
                # Check if projections have relationships to games/matches
                relationships = sample_proj.get('relationships', {})
                print(f"\n🔗 Projection relationships: {list(relationships.keys())}")
                
        else:
            print(f"❌ API Failed with status: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    examine_api_structure()