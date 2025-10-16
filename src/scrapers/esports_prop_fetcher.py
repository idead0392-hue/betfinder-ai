# src/scrapers/esports_prop_fetcher.py
from playwright.sync_api import sync_playwright
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
    Fetches player props for a given eSport by scraping The Esports Lab using Playwright.
    """
    sport_path = SPORT_URL_MAP.get(sport.lower())
    if sport_path is None:
        print(f"Unsupported eSport: {sport}")
        return []

    url = f"{BASE_URL}{sport_path}"
    props = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Navigate to the page and wait for it to be fully loaded
            page.goto(url, wait_until='networkidle', timeout=30000)

            # Locate the main props table
            table = page.locator('table').first
            if not table.is_visible():
                print(f"Could not find a visible prop table on {url}")
                browser.close()
                return []

            # Get all rows from the table body
            rows = table.locator('tbody tr').all()

            for row in rows:
                cols = row.locator('td').all()
                if len(cols) > 4:
                    try:
                        player_name = cols[0].text_content().strip()
                        stat_type = cols[1].text_content().strip()
                        prop_line_text = cols[2].text_content().strip()
                        projection_text = cols[4].text_content().strip()

                        # Ensure the text can be converted to a float
                        if prop_line_text and projection_text:
                            prop_line = float(prop_line_text)
                            projection = float(projection_text)
                        else:
                            continue

                        props.append({
                            'sport': sport,
                            'player_name': player_name,
                            'stat_type': stat_type,
                            'prop_line': prop_line,
                            'ai_projection': projection,
                            'sportsbook': 'The Esports Lab',
                            'matchup': 'N/A',
                            'team': 'N/A'
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Skipping a row due to parsing error: {e}")
                        continue
        
        except Exception as e:
            print(f"An error occurred while scraping {url}: {e}")
        
        finally:
            browser.close()
            
    return props
