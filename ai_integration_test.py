import http.client
import os
from dotenv import load_dotenv

#!/usr/bin/env python3
"""
Azure AI Inference SDK Integration Test
This script tests the Azure AI Inference SDK integration with GitHub Models.
"""

# Load environment variables from .env file
load_dotenv()

# RapidAPI Configuration from environment variables
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST', 'esportapi1.p.rapidapi.com')

if not RAPIDAPI_KEY:
    raise ValueError("RAPIDAPI_KEY environment variable is required. Please add it to your .env file.")

def get_esport_game_statistics(game_id, api_key, api_host):
    """
    Get esport game statistics using RapidAPI.
    
    Args:
        game_id: The esport game ID
        api_key: RapidAPI key
        api_host: RapidAPI host
    
    Returns:
        str: JSON response data
    """
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': api_host
    }
    conn.request("GET", f"/api/esport/game/{game_id}/statistics", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def get_esport_team(team_id, api_key, api_host):
    """
    Get esport team information using RapidAPI.
    
    Args:
        team_id: The esport team ID
        api_key: RapidAPI key
        api_host: RapidAPI host
    
    Returns:
        str: JSON response data
    """
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': api_host
    }
    conn.request("GET", f"/api/esport/team/{team_id}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def get_esport_tournament(tournament_id, api_key, api_host):
    """
    Get esport tournament information using RapidAPI.
    
    Args:
        tournament_id: The esport tournament ID
        api_key: RapidAPI key
        api_host: RapidAPI host
    
    Returns:
        str: JSON response data
    """
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': api_host
    }
    conn.request("GET", f"/api/esport/tournament/{tournament_id}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def get_esport_season_info(tournament_id, season_id, api_key, api_host):
    """
    Get esport season information using RapidAPI.
    
    Args:
        tournament_id: The esport tournament ID
        season_id: The season ID
        api_key: RapidAPI key
        api_host: RapidAPI host
    
    Returns:
        str: JSON response data
    """
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': api_host
    }
    conn.request("GET", f"/api/esport/tournament/{tournament_id}/season/{season_id}/info", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def get_esport_season_last_matches(tournament_id, season_id, page, api_key, api_host):
    """
    Get esport season last matches using RapidAPI.
    
    Args:
        tournament_id: The esport tournament ID
        season_id: The season ID
        page: The page number
        api_key: RapidAPI key
        api_host: RapidAPI host
    
    Returns:
        str: JSON response data
    """
    conn = http.client.HTTPSConnection(api_host)
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': api_host
    }
    conn.request("GET", f"/api/esport/tournament/{tournament_id}/season/{season_id}/matches/last/{page}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Esport Team API Test")
    print("=" * 60 + "\n")
    result = get_esport_team('363904', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Tournament API Test")
    print("=" * 60 + "\n")
    result = get_esport_tournament('16026', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Season Info API Test")
    print("=" * 60 + "\n")
    result = get_esport_season_info('16026', '47832', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Season Last Matches API Test")
    print("=" * 60 + "\n")
    result = get_esport_season_last_matches('16026', '47832', '1', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
