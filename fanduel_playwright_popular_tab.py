import asyncio
from playwright.async_api import async_playwright

POPULAR_TAB_URL = "https://sportsbook.fanduel.com/navigation/popular"

async def scrape_popular_tab():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(POPULAR_TAB_URL)
        await page.wait_for_load_state('networkidle')
        props = []
        # Find all soccer match blocks (li > div)
        match_blocks = await page.query_selector_all('li > div')
        soccer_odds = []
        for match in match_blocks:
            # Extract team names (should be two spans with aria-label inside each match)
            team_spans = await match.query_selector_all('span[aria-label][role="text"]')
            if len(team_spans) >= 2:
                home_team = await team_spans[0].get_attribute('aria-label')
                away_team = await team_spans[1].get_attribute('aria-label')
            else:
                home_team = away_team = None
            # Odds are in sibling divs after the match block
            odds_buttons = await match.query_selector_all('div[aria-label][role="button"]')
            odds = {'home': None, 'tie': None, 'away': None}
            for btn in odds_buttons:
                label = await btn.get_attribute('aria-label')
                if label:
                    if home_team and f'{home_team} to win' in label:
                        odds['home'] = await btn.text_content()
                    elif away_team and f'{away_team} to win' in label:
                        odds['away'] = await btn.text_content()
                    elif 'tie' in label or 'draw' in label:
                        odds['tie'] = await btn.text_content()
            soccer_odds.append({
                'home_team': home_team,
                'away_team': away_team,
                'odds_home': odds['home'],
                'odds_tie': odds['tie'],
                'odds_away': odds['away']
            })
        return soccer_odds
        for block in match_blocks:
            # Find team names
            team_spans = await block.query_selector_all('span[aria-label][role="text"]')
            teams = [await t.get_attribute('aria-label') for t in team_spans]
            # Find all odds divs inside this block
            odds_divs = await block.query_selector_all('div[aria-label][role="button"]')
            for odds_div in odds_divs:
                aria_label = await odds_div.get_attribute('aria-label')
                spans = await odds_div.query_selector_all('span')
                odds_value = await spans[0].text_content() if len(spans) > 0 else None
                # Extract bet type from aria-label (e.g., 'USA to win', 'both to tie', 'Australia to win')
                if aria_label and teams:
                    props.append({
                        'teams': teams,
                        'aria_label': aria_label,
                        'odds': odds_value
                    })
        await browser.close()
        return props
        await browser.close()
        return props

if __name__ == "__main__":
    results = asyncio.run(scrape_popular_tab())
    print("Popular tab props:")
    for prop in results:
        print(prop)
