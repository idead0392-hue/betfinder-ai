# src/scrapers/prizepicks_scraper.py
import os
import time
from typing import List, Dict
import random

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ..core.cfb_to_nfl_mapping import get_nfl_info
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

API_URL = "https://api.prizepicks.com/projections"

DEMON_KEYWORDS = [
    "over only", "must pick over", "cannot play under", "first pitch",
    "first score method", "2h only", "over-only", "promo-only", "over only"
]

def is_demon_prop(label: str, desc: str = "") -> bool:
    text = f"{label or ''} {desc or ''}".lower()
    return any(k in text for k in DEMON_KEYWORDS)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def get_session_with_retries():
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_prizepicks_api(sport: str = None, league_id: str = None) -> List[Dict]:
    """Fetch projections from PrizePicks public API."""
    session = get_session_with_retries()
    user_agent = random.choice(USER_AGENTS)
    
    params = {
        "page": 1,
        "per_page": 250,
    }
    if league_id:
        params["league_id"] = league_id
    if sport:
        params["sport"] = sport
    
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": user_agent,
    }
    
    items: List[Dict] = []
    
    try:
        resp = session.get(API_URL, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return items
        
        data = resp.json()
        included = {
            obj["id"]: obj
            for obj in data.get("included", [])
            if isinstance(obj, dict) and obj.get("type")
        }
        
        for proj in data.get("data", []):
            try:
                attr = proj.get("attributes", {})
                projection_type = attr.get("stat_type") or ""
                line_val = attr.get("line_score") or 0
                
                player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
                player_id = player_rel.get("id") if isinstance(player_rel, dict) else None
                player = included.get(player_id) if player_id else None
                
                player_name = (player or {}).get("attributes", {}).get("name") or "Unknown"
                league = (player or {}).get("attributes", {}).get("league", "") if isinstance(player, dict) else ""
                team = (player or {}).get("attributes", {}).get("team", "") if isinstance(player, dict) else ""
                
                nfl_info = get_nfl_info(player_name)
                if nfl_info:
                    league, team = nfl_info
                
                # --- Start of Enhanced Normalization ---
                league_lc = (league or "").strip().lower()
                sport_norm = ''
                aliases = {
                    'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
                    'nfl': 'football', 'cfb': 'college_football',
                    'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer',
                    # Enhanced eSports Aliases
                    'league of legends': 'league_of_legends', 'lol': 'league_of_legends',
                    'valorant': 'valorant', 'valo': 'valorant',
                    'dota 2': 'dota2',
                    'csgo': 'cs2', 'cs:go': 'cs2', 'cs2': 'cs2', 'counter-strike': 'cs2'
                }
                sport_norm = aliases.get(league_lc, '')

                # Infer from prop type if league name is not explicit
                if not sport_norm:
                    proj_lc = str(projection_type or '').lower()
                    if any(k in proj_lc for k in ['kda','creep','vision score']):
                        sport_norm = 'league_of_legends'
                    elif any(k in proj_lc for k in ['gpm','xpm','last hits','roshan']):
                        sport_norm = 'dota2'
                    elif any(k in proj_lc for k in ['adr', 'headshot', 'map kills']):
                        sport_norm = 'cs2'
                    elif any(k in proj_lc for k in ['acs','first blood','spike']):
                        sport_norm = 'valorant'
                
                items.append(
                    {
                        "Name": player_name,
                        "Points": line_val,
                        "Prop": projection_type,
                        "League": league,
                        "Sport": sport_norm,  # Use the normalized sport name
                        "Team": team,
                    }
                )
                # --- End of Enhanced Normalization ---
            except Exception:
                continue
    except Exception:
        pass
    
    return items

def main():
    output_csv = os.environ.get("PRIZEPICKS_CSV", "data/prizepicks_props.csv")
    items = fetch_prizepicks_api()
    
    if items:
        df = pd.DataFrame(items)
        df.to_csv(output_csv, index=False)
        print(f"Props scraped and saved to: {output_csv}")
    else:
        print("Failed to scrape PrizePicks props.")

if __name__ == "__main__":
    main()
