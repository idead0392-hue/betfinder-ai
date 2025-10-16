# src/scrapers/prizepicks_scraper.py
import time
import random
from typing import List, Dict
from playwright.sync_api import sync_playwright
import pandas as pd
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ..core.cfb_to_nfl_mapping import get_nfl_info

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def fetch_prizepicks_props() -> List[Dict]:
    """
    Fetches props from PrizePicks by intercepting API calls via a headless browser.
    """
    items: List[Dict] = []
    captured_json: List[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=random.choice(USER_AGENTS))
        page = context.new_page()

        # Define a handler to intercept network responses
        def handle_response(response):
            if "api.prizepicks.com/projections" in response.url and response.status == 200:
                try:
                    captured_json.append(response.json())
                except Exception:
                    pass

        page.on("response", handle_response)

        try:
            # Navigate to the site to trigger the API calls
            page.goto("https://app.prizepicks.com/", wait_until="networkidle", timeout=30000)
            time.sleep(5) # Wait for any dynamic content to load

            # Process the captured API data
            for data in captured_json:
                included = {obj["id"]: obj for obj in data.get("included", [])}
                for proj in data.get("data", []):
                    # (Add your existing JSON parsing logic here)
                    pass # Placeholder for parsing logic
        
        except Exception as e:
            print(f"An error occurred while scraping PrizePicks: {e}")

        finally:
            browser.close()

    return items # Return parsed items

def main():
    output_csv = os.environ.get("PRIZEPICKS_CSV", "data/prizepicks_props.csv")
    props = fetch_prizepicks_props()

    if props:
        df = pd.DataFrame(props)
        df.to_csv(output_csv, index=False)
        print(f"Successfully scraped {len(props)} props and saved to {output_csv}")
    else:
        print("Failed to scrape any props from PrizePicks.")

if __name__ == "__main__":
    main()
