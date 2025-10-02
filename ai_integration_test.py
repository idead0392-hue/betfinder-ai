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

def get_esport_game_bans(game_id, api_key, api_host):
    """
    Get esport game bans using RapidAPI.
    
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
    conn.request("GET", f"/api/esport/game/{game_id}/bans", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def get_esport_game_rounds(game_id, api_key, api_host):
    """
    Get esport game rounds using RapidAPI.
    
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
    conn.request("GET", f"/api/esport/game/{game_id}/rounds", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def get_esport_player(player_id, api_key, api_host):
    """
    Get esport player information using RapidAPI.
    
    Args:
        player_id: The esport player ID
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
    conn.request("GET", f"/api/esport/player/{player_id}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def get_esport_player_image(player_id, api_key, api_host):
    """
    Get esport player image using RapidAPI.
    
    Args:
        player_id: The esport player ID
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
    conn.request("GET", f"/api/esport/player/{player_id}/image", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def test_ai_inference():
    """
    Test Azure AI Inference SDK with GitHub Models.
    Uses GITHUB_TOKEN environment variable for authentication.
    """
    # Get GitHub token from environment variable
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable not set.")
        print("Please set it with: export GITHUB_TOKEN='your_token_here'")
        return
    
    # Initialize the client
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "gpt-4o-mini"  # or your preferred model
    
    try:
        print("Initializing Azure AI Inference client...")
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(github_token)
        )
        
        print(f"Testing model: {model_name}")
        print("Sending test prompt...\n")
        
        # Create test messages
        messages = [
            SystemMessage(content="You are a helpful AI assistant for sports betting analysis."),
            UserMessage(content="Explain in 2 sentences what makes a good betting opportunity.")
        ]
        
        # Get completion
        response = client.complete(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_tokens=200
        )
        
        # Display results
        print("=" * 60)
        print("MODEL OUTPUT:")
        print("=" * 60)
        print(response.choices[0].message.content)
        print("\n" + "=" * 60)
        print("TOKEN USAGE:")
        print("=" * 60)
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your GITHUB_TOKEN is valid and has proper permissions.")

if __name__ == "__main__":
    print("Azure AI Inference SDK Integration Test")
    print("=" * 40)
    print()
    
    # Test AI inference
    test_ai_inference()
    
    print("\n" + "=" * 60)
    print("Esport Game Statistics API Test")
    print("=" * 60 + "\n")
    result = get_esport_game_statistics('359554', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Game Bans API Test")
    print("=" * 60 + "\n")
    result = get_esport_game_bans('359554', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Game Rounds API Test")
    print("=" * 60 + "\n")
    result = get_esport_game_rounds('359440', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Player API Test")
    print("=" * 60 + "\n")
    result = get_esport_player('1078255', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    print("\n" + "=" * 60)
    print("Esport Player Image API Test")
    print("=" * 60 + "\n")
    result = get_esport_player_image('1078255', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
