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
        print("\nTest completed successfully!")
        print(f"Model: {model_name}")
        print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"\nError during API call: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your GITHUB_TOKEN is valid")
        print("2. Check that you have access to GitHub Models")
        print("3. Ensure the model name is correct")
        return

def get_esport_streaks(event_id, api_key, api_host):
    """
    Get esport event streaks using RapidAPI.
    
    Args:
        event_id: The esport event ID
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
    
    conn.request("GET", f"/api/esport/event/{event_id}/streaks", headers=headers)
    
    res = conn.getresponse()
    data = res.read()
    
    return data.decode("utf-8")

def get_esport_matches(date_path, api_key, api_host):
    """
    Get esport matches for a specific date using RapidAPI.
    
    Args:
        date_path: Date path in format "DD/MM/YYYY"
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
    
    conn.request("GET", f"/api/esport/matches/{date_path}", headers=headers)
    
    res = conn.getresponse()
    data = res.read()
    
    return data.decode("utf-8")

def get_esport_event_highlights(event_id, api_key, api_host):
    """
    Get esport event highlights using RapidAPI.
    
    Args:
        event_id: The esport event ID
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
    
    conn.request("GET", f"/api/esport/event/{event_id}/highlights", headers=headers)
    
    res = conn.getresponse()
    data = res.read()
    
    return data.decode("utf-8")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Azure AI Inference SDK - Integration Test")
    print("BetFinder AI Project")
    print("=" * 60 + "\n")
    test_ai_inference()
    
    # Test esport streaks API
    print("\n" + "=" * 60)
    print("Esport Streaks API Test")
    print("=" * 60 + "\n")
    
    result = get_esport_streaks(
        event_id="10944886",
        api_key=RAPIDAPI_KEY,
        api_host=RAPIDAPI_HOST
    )
    print(result)
    
    # Test esport matches API
    print("\n" + "=" * 60)
    print("Esport Matches API Test")
    print("=" * 60 + "\n")
    
    result = get_esport_matches('18/12/2022', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
    
    # Test esport event highlights API
    print("\n" + "=" * 60)
    print("Esport Event Highlights API Test")
    print("=" * 60 + "\n")
    
    result = get_esport_event_highlights('10945370', RAPIDAPI_KEY, RAPIDAPI_HOST)
    print(result)
