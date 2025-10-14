import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
import sys
import argparse


async def scrape_esportslab(url: str, timeout: int = 30000) -> list[dict]:
    """
    Scrapes projection data from a given TheEsportsLab URL using Playwright.

    Args:
        url: The URL of the projections page to scrape.
        timeout: Navigation timeout in ms.

    Returns:
        A list of dictionaries, where each dictionary represents a player's projection.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=("Mozilla/5.0 (X11; Linux x86_64) "
                                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                 "Chrome/120.0.0.0 Safari/537.36"))
        try:
            await page.goto(url, wait_until='networkidle', timeout=timeout)

            # The data is typically within a table, let's find it.
            tables = await page.query_selector_all('table')
            if not tables:
                print(f"No tables found on {url}")
                return []

            # Assuming the main projections table is the first one found
            html_content = await tables[0].inner_html()

            # Use pandas to parse the HTML table
            df_list = pd.read_html(f'<table>{html_content}</table>')

            if not df_list:
                print(f"Could not parse table from {url}")
                return []

            df = df_list[0]

            # Clean and structure the DataFrame
            # Column names are often nested, so we may need to flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(c).strip() for c in col if c and str(c).strip()]) for col in df.columns.values]

            # Rename columns to a standard format if possible
            column_mapping = {
                'Player': 'player_name',
                'player': 'player_name',
                'Team': 'team',
                'team': 'team',
                'Stat': 'stat_type',
                'Stat Type': 'stat_type',
                'Line': 'line',
                'Projection': 'projection'
                # Add other mappings as needed
            }
            # Lowercase df columns for resilient mapping
            df.columns = [str(c).strip() for c in df.columns]
            lower_map = {c.lower(): c for c in df.columns}

            rename_map = {}
            for k, v in column_mapping.items():
                lk = k.lower()
                if lk in lower_map:
                    rename_map[lower_map[lk]] = v
            if rename_map:
                df = df.rename(columns=rename_map)

            # Try to coerce numeric columns
            for col in ['line', 'projection']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9\.-]', '', regex=True), errors='coerce')
                    except Exception:
                        pass

            # Normalize player/team strings
            for col in ['player_name', 'team']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

            # Convert to list of dicts
            return df.to_dict('records')

        except PlaywrightTimeoutError:
            print(f"Timeout navigating to {url}")
            return []
        except Exception as e:
            print(f"An error occurred while scraping {url}: {e}")
            return []
        finally:
            try:
                await browser.close()
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape TheEsportsLab projections with Playwright')
    parser.add_argument('urls', nargs='*', help='URLs to scrape', default=[
        'https://theesportslab.com/projections/cs2',
        'https://theesportslab.com/projections/lol',
        'https://theesportslab.com/projections/dota2'
    ])
    parser.add_argument('--timeout', type=int, default=30000, help='Navigation timeout in ms')
    args = parser.parse_args()

    for test_url in args.urls:
        print(f"--- Scraping {test_url} ---")
        try:
            data = asyncio.run(scrape_esportslab(test_url, timeout=args.timeout))
            if data:
                print(f"Successfully scraped {len(data)} projections.")
                print("First 3 entries:")
                for entry in data[:3]:
                    print(entry)
            else:
                print("Scraping failed or returned no data.")
        except Exception as e:
            print(f"Failed to run scrape for {test_url}: {e}")
        print("-" * (len(test_url) + 14))
