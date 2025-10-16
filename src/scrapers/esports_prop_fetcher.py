# src/scrapers/esports_prop_fetcher.py
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# Base URL for the projections
BASE_URL = "https://theesportslab.com/projections/"

# Map our internal sport names to the URL paths on the website
SPORT_URL_MAP = {
    "cs2": "cs2",
    "league_of_legends": "",  # LoL is at the root /projections/
    "dota2": "dota2",
    "valorant": "valorant"
}

def fetch_esports_props(sport: str) -> List[Dict[str, Any]]:
    """
    Fetches player props for a given eSport by scraping The Esports Lab.
    """
    sport_path = SPORT_URL_MAP.get(sport.lower())
    if sport_path is None:
        print(f"Unsupported eSport: {sport}")
        return []

    url = f"{BASE_URL}{sport_path}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    
    props = []
    
    # Find the table containing the props - this may need adjustment if the site structure changes
    prop_table = soup.find('table')
    if not prop_table:
        print(f"Could not find prop table on {url}")
        return []

    # Find all rows in the table body
    rows = prop_table.find('tbody').find_all('tr') if prop_table.find('tbody') else []
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 4:  # Ensure there are enough columns to parse
            try:
                player_name = cols[0].text.strip()
                stat_type = cols[1].text.strip()
                prop_line = float(cols[2].text.strip())
                projection = float(cols[4].text.strip())
                
                props.append({
                    'sport': sport,
                    'player_name': player_name,
                    'stat_type': stat_type,
                    'prop_line': prop_line,
                    'ai_projection': projection, # Using their projection as our 'AI projection'
                    'sportsbook': 'The Esports Lab', # Source of the data
                    'matchup': 'N/A', # This data is not available on the page
                    'team': 'N/A' # This data is not available on the page
                })
            except (ValueError, IndexError):
                # Skip rows that don't parse correctly
                continue
                
    return props
