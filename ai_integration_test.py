#!/usr/bin/env python3
"""
Azure AI Inference SDK Integration Test
This script tests the Azure AI Inference SDK integration with GitHub Models.
"""
import os
import http.client
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# RapidAPI Configuration
RAPIDAPI_KEY = "4ac75ea836mshd4804cec3a1eea0p1085f3jsn0b73793a40af"
RAPIDAPI_HOST = "esportapi1.p.rapidapi.com"

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
    print("Esport Season Info API Test")
    print("=" * 60 + "\n")
    result = get_esport_season_info('16026', '47832', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Season Last Matches API Test")
    print("=" * 60 + "\n")
    result = get_esport_season_last_matches('16026', '47832', '1', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
