import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_table_rows(html):
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    # Find the main projections table, update selector if page structure changes
    for tr in soup.select('table tbody tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
        if cells:
            rows.append(cells)
    return rows

def scrape_pages(base_url, page_numbers):
    all_rows = []
    for page in page_numbers:
        url = f'{base_url}?page={page}'
        print(f'Scraping: {url}')
        resp = requests.get(url)
        resp.raise_for_status()
        all_rows.extend(get_table_rows(resp.text))
    return all_rows

# Adjust BASE URL as needed, for 'All Projections' section:
base_url = 'https://www.pickfinder.app/projections'
page_numbers = range(1, 24)  # For all pages 1-23
rows = scrape_pages(base_url, page_numbers)

# Adjust these columns if the site's table changes!
columns = [
    'Player', 'Team/Opp', 'Prop', 'Line', 'Apps',
    'Odds', 'IP', 'Def', 'Avg_L10', 'Diff', 'L5', 'L10', 'L15', 'H2H', 'Strk', 'SZN'
]

df = pd.DataFrame(rows, columns=columns[:len(rows[0])])
df.to_csv('pickfinder_all_projections.csv', index=False)
print('CSV exported: pickfinder_all_projections.csv')
EOFcat > pickfinder_scrape.py << 'EOF'
import pandas as pd

def get_table_rows(html):
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    # Find the main projections table, update selector if page structure changes
    for tr in soup.select('table tbody tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
        if cells:
            rows.append(cells)
    return rows

def scrape_pages(base_url, page_numbers):
    all_rows = []
    for page in page_numbers:
        url = f'{base_url}?page={page}'
        print(f'Scraping: {url}')
        resp = requests.get(url)
        resp.raise_for_status()
        all_rows.extend(get_table_rows(resp.text))
    return all_rows

# Adjust BASE URL as needed, for 'All Projections' section:
base_url = 'https://www.pickfinder.app/projections'
page_numbers = range(1, 24)  # For all pages 1-23
rows = scrape_pages(base_url, page_numbers)

# Adjust these columns if the site's table changes!
columns = [
    'Player', 'Team/Opp', 'Prop', 'Line', 'Apps',
    'Odds', 'IP', 'Def', 'Avg_L10', 'Diff', 'L5', 'L10', 'L15', 'H2H', 'Strk', 'SZN'
]

df = pd.DataFrame(rows, columns=columns[:len(rows[0])])
df.to_csv('pickfinder_all_projections.csv', index=False)
print('CSV exported: pickfinder_all_projections.csv')
