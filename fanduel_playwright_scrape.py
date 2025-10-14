import asyncio
from playwright.async_api import async_playwright

async def scrape_fanduel_props():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        # Example FanDuel NBA props URL (update as needed)
        url = "https://sportsbook.fanduel.com/navigation/nba"
        await page.goto(url)
        await page.wait_for_load_state('networkidle')
        # Example: Scrape player prop names and odds (update selectors as needed)
        props = []
        # This selector is a placeholder. You must inspect FanDuel's DOM for actual selectors.
        for el in await page.query_selector_all('div[data-test="event-row"]'):
            name = await el.query_selector_eval('span[data-test="participant-name"]', 'el => el.textContent')
            odds = await el.query_selector_eval('span[data-test="outcome-price"]', 'el => el.textContent')
            props.append({
                'name': name.strip() if name else None,
                'odds': odds.strip() if odds else None
            })
        await browser.close()
        return props

if __name__ == "__main__":
    results = asyncio.run(scrape_fanduel_props())
    for prop in results:
        print(prop)
