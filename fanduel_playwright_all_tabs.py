import asyncio
from playwright.async_api import async_playwright

FANDUEL_TABS = [
    # Add more tabs as needed
    {"name": "NBA", "url": "https://sportsbook.fanduel.com/navigation/nba"},
    {"name": "NFL", "url": "https://sportsbook.fanduel.com/navigation/nfl"},
    {"name": "MLB", "url": "https://sportsbook.fanduel.com/navigation/mlb"},
    {"name": "NHL", "url": "https://sportsbook.fanduel.com/navigation/nhl"},
    {"name": "Soccer", "url": "https://sportsbook.fanduel.com/navigation/soccer"},
]

async def scrape_tab_props(page, tab):
    await page.goto(tab["url"])
    await page.wait_for_load_state('networkidle')
    props = []
    # Placeholder selectors, update for each tab as needed
    for el in await page.query_selector_all('div[data-test="event-row"]'):
        name = await el.query_selector_eval('span[data-test="participant-name"]', 'el => el.textContent')
        odds = await el.query_selector_eval('span[data-test="outcome-price"]', 'el => el.textContent')
        props.append({
            'name': name.strip() if name else None,
            'odds': odds.strip() if odds else None
        })
    return props

async def scrape_all_tabs():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        all_results = {}
        for tab in FANDUEL_TABS:
            print(f"Scraping {tab['name']}...")
            props = await scrape_tab_props(page, tab)
            all_results[tab['name']] = props
        await browser.close()
        return all_results

if __name__ == "__main__":
    results = asyncio.run(scrape_all_tabs())
    for tab, props in results.items():
        print(f"\n{tab} props:")
        for prop in props:
            print(prop)
