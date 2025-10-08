#!/usr/bin/env python3
"""
Example usage of SportbexProvider with environment variables.

This demonstrates how to use the SportbexProvider class with proper
environment variable configuration for production use.
"""

import os
from api_providers import SportbexProvider, SportType, create_sportbex_provider

def main():
    """Example usage of SportbexProvider."""
    
    # Method 1: Set environment variable (recommended for production)
    # os.environ['SPORTBEX_API_KEY'] = 'your_api_key_here'
    # provider = create_sportbex_provider()
    
    # Method 2: Pass API key directly (for testing)
    api_key = os.getenv('SPORTBEX_API_KEY', 'NZLDw8ZXFv0O8elaPq0wjbP4zxb2gCwJDsArWQUF')
    provider = SportbexProvider(api_key=api_key)
    
    # Example 1: Get tennis competitions
    print("🎾 Getting tennis competitions...")
    response = provider.get_competitions(sport=SportType.TENNIS)
    if response.success:
        competitions = response.data.get('data', []) if isinstance(response.data, dict) else response.data
        print(f"Found {len(competitions)} tennis competitions")
        
        # Show first few competitions
        for comp in competitions[:3]:
            if isinstance(comp, dict):
                name = comp.get('competition', {}).get('name', 'Unknown')
                region = comp.get('competitionRegion', 'Unknown')
                markets = comp.get('marketCount', 0)
                print(f"  - {name} ({region}) - {markets} markets")
    else:
        print(f"Error: {response.error_message}")
    
    # Example 2: Get basketball props
    print("\n🏀 Getting basketball props...")
    response = provider.get_props(sport=SportType.BASKETBALL)
    if response.success:
        props = response.data.get('data', []) if isinstance(response.data, dict) else response.data
        print(f"Found {len(props)} basketball competitions")
    else:
        print(f"Error: {response.error_message}")
    
    # Example 3: Get matchups for a specific competition
    print("\n🏆 Getting matchups for tennis competitions...")
    tennis_comps = provider.get_competitions(sport=SportType.TENNIS)
    if tennis_comps.success:
        competitions = tennis_comps.data.get('data', []) if isinstance(tennis_comps.data, dict) else tennis_comps.data
        if competitions:
            # Get matchups for first competition
            comp_id = competitions[0].get('competition', {}).get('id')
            if comp_id:
                matchups_response = provider.get_matchups(sport=SportType.TENNIS, competition_id=comp_id)
                if matchups_response.success:
                    matchups = matchups_response.data.get('data', []) if isinstance(matchups_response.data, dict) else matchups_response.data
                    print(f"Found {len(matchups)} matchups for competition {comp_id}")
                else:
                    print(f"Error getting matchups: {matchups_response.error_message}")
    
    # Example 4: Health check
    print("\n🔍 Running health check...")
    health = provider.health_check()
    if health.success:
        print(f"✅ API is healthy (response time: {health.data.get('response_time', 'N/A')}s)")
    else:
        print(f"❌ API health check failed: {health.error_message}")

if __name__ == "__main__":
    main()