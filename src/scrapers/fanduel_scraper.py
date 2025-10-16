# src/scrapers/fanduel_scraper.py
import asyncio
from playwright.async_api import async_playwright
from typing import List, Dict

async def scrape_fanduel_props() -> List[Dict]:
    """
    Scrapes NBA player props from FanDuel using Playwright with more specific selectors.
    """
    props = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = "https://sportsbook.fanduel.com/navigation/nba"
            await page.goto(url, wait_until='networkidle', timeout=45000)

            # Example: Click a "Player Props" tab if it exists
            # This selector is hypothetical and needs to be verified on the actual site
            player_props_tab = page.locator('a:has-text("Player Props")')
            if await player_props_tab.is_visible():
                await player_props_tab.click()
                await page.wait_for_timeout(3000) # Wait for content to load

            # More specific selectors for scraping prop data
            # These selectors are hypothetical and need to be verified
            for event in await page.locator('[data-test-id="event-card"]').all():
                player_name = await event.locator('[data-test-id="player-name"]').text_content()
                market = await event.locator('[data-test-id="market-name"]').text_content()
                line = await event.locator('[data-test-id="prop-line"]').text_content()
                over_odds = await event.locator('[data-test-id="over-odds"]').text_content()
                under_odds = await event.locator('[data-test-id="under-odds"]').text_content()

                props.append({
                    'player_name': player_name.strip(),
                    'market': market.strip(),
                    'line': float(line),
                    'over_odds': over_odds.strip(),
                    'under_odds': under_odds.strip()
                })

        except Exception as e:
            print(f"An error occurred while scraping FanDuel: {e}")
        finally:
            await browser.close()
    
    return props

if __name__ == "__main__":
    results = asyncio.run(scrape_fanduel_props())
    if results:
        for prop in results:
            print(prop)
    else:
        print("No props were scraped from FanDuel.")
