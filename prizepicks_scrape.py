import os
import sys
import json
import time
import csv
from typing import List, Dict
import random

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cfb_to_nfl_mapping import get_nfl_info
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


API_URL = "https://api.prizepicks.com/projections"

# Keywords indicating over-only or forbidden-under props
DEMON_KEYWORDS = [
    "over only", "must pick over", "cannot play under", "first pitch",
    "first score method", "2h only", "over-only", "promo-only", "over only"
]

def is_demon_prop(label: str, desc: str = "") -> bool:
    text = f"{label or ''} {desc or ''}".lower()
    return any(k in text for k in DEMON_KEYWORDS)

# Multiple user agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
]

def get_session_with_retries():
    """Create a requests session with retry strategy and random delays"""
    session = requests.Session()
    
    # Retry strategy
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
    """Fetch projections from PrizePicks public API with enhanced bypass techniques."""

    session = get_session_with_retries()

    # Random user agent
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
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://app.prizepicks.com",
        "Referer": "https://app.prizepicks.com/board",
        "User-Agent": user_agent,
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    items: List[Dict] = []

    endpoints = [
        "https://api.prizepicks.com/projections",
        "https://partner-api.prizepicks.com/projections",
        "https://api.prizepicks.com/picks",
    ]

    for attempt in range(3):
        # Random delay between attempts
        if attempt > 0:
            time.sleep(random.uniform(2, 5))
        for endpoint in endpoints:
            try:
                resp = session.get(endpoint, params=params, headers=headers, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                included = {
                    obj["id"]: obj
                    for obj in data.get("included", [])
                    if isinstance(obj, dict) and obj.get("type")
                }

                for proj in data.get("data", []):
                    try:
                        attr = proj.get("attributes", {})
                        projection_type = attr.get("stat_type") or attr.get("type") or ""
                        line_val = attr.get("line_score") or attr.get("line") or None
                        player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
                        player_id = player_rel.get("id") if isinstance(player_rel, dict) else None
                        player = included.get(player_id) if player_id else None
                        player_name = (player or {}).get("attributes", {}).get("name") or "Unknown"

                        # Default league/team (may be missing from API); patch via mapping
                        league = (player or {}).get("attributes", {}).get("league", "") if isinstance(player, dict) else ""
                        team = (player or {}).get("attributes", {}).get("team", "") if isinstance(player, dict) else ""

                        nfl_info = get_nfl_info(player_name)
                        if nfl_info:
                            league, team = nfl_info  # Force to NFL/team

                        # Extract matchup information from game relationship
                        matchup = "TBD vs TBD"
                        home_team = ""
                        away_team = ""
                        game_date = ""
                        game_time = ""

                        try:
                            game_rel = proj.get("relationships", {}).get("game", {}).get("data", {})
                            if game_rel and game_rel.get("id"):
                                game_id = game_rel["id"]
                                game_obj = included.get(game_id)

                                if game_obj and game_obj.get("type") == "game":
                                    game_info = (
                                        game_obj.get("attributes", {})
                                        .get("metadata", {})
                                        .get("game_info", {})
                                    )
                                    teams = game_info.get("teams", {})
                                    # Attempt to parse a start time from attributes or metadata
                                    start_raw = (
                                        game_obj.get("attributes", {}).get("start_time")
                                        or game_obj.get("attributes", {}).get("start_at")
                                        or game_obj.get("attributes", {}).get("scheduled_for")
                                        or game_obj.get("attributes", {}).get("commence_time")
                                        or game_info.get("start_time")
                                        or game_info.get("game_time")
                                        or game_info.get("scheduled")
                                    )
                                    try:
                                        dt = None
                                        if isinstance(start_raw, (int, float)):
                                            dt = datetime.fromtimestamp(float(start_raw), tz=timezone.utc)
                                        elif isinstance(start_raw, str) and start_raw:
                                            s = start_raw.strip()
                                            if s.endswith('Z'):
                                                s = s.replace('Z', '+00:00')
                                            try:
                                                dt = datetime.fromisoformat(s)
                                                if dt.tzinfo is None:
                                                    dt = dt.replace(tzinfo=timezone.utc)
                                            except Exception:
                                                dt = None
                                        if dt is not None:
                                            et = dt.astimezone(ZoneInfo('America/New_York'))
                                            game_date = et.strftime('%Y-%m-%d')
                                            game_time = et.strftime('%-I:%M %p ET')
                                    except Exception:
                                        pass

                                    home_info = teams.get("home", {})
                                    away_info = teams.get("away", {})

                                    home_team = home_info.get("name") or home_info.get("abbreviation", "")
                                    away_team = away_info.get("name") or away_info.get("abbreviation", "")

                                    if home_team and away_team:
                                        matchup = f"{away_team} @ {home_team}"
                                    elif home_team or away_team:
                                        # For individual sports like tennis, use vs format
                                        player_team = team if team else (home_team or away_team)
                                        opponent = away_team if home_team == player_team else home_team
                                        if opponent and player_team != opponent:
                                            matchup = f"{player_team} vs {opponent}"
                                else:
                                    # Game object exists but no team info - fallback
                                    matchup = "TBD vs TBD"
                            else:
                                # No game relationship - use team info if available
                                if team:
                                    matchup = f"{team} vs TBD"
                        except Exception:
                            # Fallback: use team info if available
                            if team:
                                matchup = f"{team} vs TBD"
                            else:
                                matchup = "TBD vs TBD"

                        # Detect over-only/demon props from stat type and any description
                        desc = str(attr.get("description") or "")
                        allow_under = not is_demon_prop(projection_type, desc)

                        # Attempt to derive a normalized sport from league/labels for better downstream mapping
                        league_lc = (league or "").strip().lower()
                        sport_norm = ''
                        aliases = {
                            'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
                            'nfl': 'football', 'cfb': 'college_football', 'ncaa football': 'college_football',
                            'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer', 'soccer': 'soccer',
                            'league of legends': 'league_of_legends', 'lol': 'league_of_legends',
                            'valorant': 'valorant', 'valo': 'valorant',
                            'dota 2': 'dota2', 'dota2': 'dota2',
                            'overwatch': 'overwatch', 'overwatch 2': 'overwatch', 'ow': 'overwatch',
                            'rocket league': 'rocket_league', 'rocket_league': 'rocket_league', 'rl': 'rocket_league',
                            'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo',
                            'apex': 'apex', 'apex legends': 'apex', 'apexlegends': 'apex'
                        }
                        sport_norm = aliases.get(league_lc, '')
                        if not sport_norm:
                            # Try to infer from projection type text for esports
                            proj_lc = str(projection_type or '').lower()
                            if any(k in proj_lc for k in ['kda','creep','vision score','cs ']):
                                sport_norm = 'league_of_legends'
                            elif any(k in proj_lc for k in ['gpm','xpm','last hits','denies','roshan']):
                                sport_norm = 'dota2'
                            elif 'adr' in proj_lc or 'headshot' in proj_lc or ('map' in proj_lc and 'kills' in proj_lc):
                                sport_norm = 'csgo'
                            elif any(k in proj_lc for k in ['acs','first blood','spike','defuse']):
                                sport_norm = 'valorant'
                            elif any(k in proj_lc for k in ['knockdowns','revives','placement']):
                                sport_norm = 'apex'

                        items.append(
                            {
                                "Name": player_name,
                                "Points": line_val,
                                "Prop": projection_type,
                                "League": league,
                                "Sport": sport_norm,
                                "Team": team,
                                "Matchup": matchup,
                                "Home_Team": home_team,
                                "Away_Team": away_team,
                                "Game_Date": game_date,
                                "Game_Time": game_time,
                                "Last_Updated": datetime.utcnow()
                                .replace(tzinfo=timezone.utc)
                                .isoformat()
                                .replace('+00:00', 'Z'),
                                "Allow_Under": allow_under,
                            }
                        )
                    except Exception:
                        continue

                if items:
                    return items
            except Exception:
                continue

    return items


def fetch_prizepicks_playwright() -> List[Dict]:
    """Use Playwright with stealth techniques to bypass anti-bot measures."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return []

    items: List[Dict] = []

    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Launch browser with stealth options
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--user-agent=' + random.choice(USER_AGENTS),
                ],
            )

            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=random.choice(USER_AGENTS),
                locale='en-US',
                timezone_id='America/New_York',
            )

            # Add stealth scripts
            context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });

                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });

                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });

                window.chrome = {
                    runtime: {},
                };
                """
            )

            page = context.new_page()

            captured_json: List[dict] = []

            def handle_response(response):
                try:
                    url = response.url
                    if "api.prizepicks.com/projections" in url and response.status == 200:
                        content_type = response.headers.get("content-type", "")
                        if "application/json" in content_type:
                            data = response.json()
                            captured_json.append(data)
                except Exception:
                    pass

            page.on("response", handle_response)

            # Navigate with random delay
            time.sleep(random.uniform(1, 3))
            page.goto("https://app.prizepicks.com/", wait_until="networkidle", timeout=30000)

            # Wait and interact like a human
            time.sleep(random.uniform(2, 4))

            # Try to click on different sports to trigger API calls
            try:
                sport_buttons = page.locator('[data-testid*="sport"], .sport-tab, [class*="sport"]').all()
                for button in sport_buttons[:3]:  # Try first 3 sports
                    try:
                        button.click()
                        time.sleep(random.uniform(1, 2))
                    except Exception:
                        continue
            except Exception:
                pass

            # Additional wait for network requests
            time.sleep(3)

            # Process captured data
            for data in captured_json:
                included = {
                    obj["id"]: obj
                    for obj in data.get("included", [])
                    if isinstance(obj, dict) and obj.get("type")
                }
                for proj in data.get("data", []):
                    try:
                        attr = proj.get("attributes", {})
                        projection_type = attr.get("stat_type") or attr.get("type") or ""
                        line_val = attr.get("line_score") or attr.get("line") or None
                        player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
                        player_id = player_rel.get("id") if isinstance(player_rel, dict) else None
                        player = included.get(player_id) if player_id else None
                        player_name = (player or {}).get("attributes", {}).get("name") or "Unknown"

                        league = (player or {}).get("attributes", {}).get("league", "") if isinstance(player, dict) else ""
                        team = (player or {}).get("attributes", {}).get("team", "") if isinstance(player, dict) else ""

                        nfl_info = get_nfl_info(player_name)
                        if nfl_info:
                            league, team = nfl_info

                        # Extract matchup information from game relationship
                        matchup = "TBD vs TBD"
                        home_team = ""
                        away_team = ""
                        game_date = ""
                        game_time = ""

                        try:
                            game_rel = proj.get("relationships", {}).get("game", {}).get("data", {})
                            if game_rel and game_rel.get("id"):
                                game_id = game_rel["id"]
                                game_obj = included.get(game_id)

                                if game_obj and game_obj.get("type") == "game":
                                    game_info = (
                                        game_obj.get("attributes", {})
                                        .get("metadata", {})
                                        .get("game_info", {})
                                    )
                                    teams = game_info.get("teams", {})

                                    start_raw = (
                                        game_obj.get("attributes", {}).get("start_time")
                                        or game_obj.get("attributes", {}).get("start_at")
                                        or game_obj.get("attributes", {}).get("scheduled_for")
                                        or game_obj.get("attributes", {}).get("commence_time")
                                        or game_info.get("start_time")
                                        or game_info.get("game_time")
                                        or game_info.get("scheduled")
                                    )
                                    try:
                                        dt = None
                                        if isinstance(start_raw, (int, float)):
                                            dt = datetime.fromtimestamp(float(start_raw), tz=timezone.utc)
                                        elif isinstance(start_raw, str) and start_raw:
                                            s = start_raw.strip()
                                            if s.endswith('Z'):
                                                s = s.replace('Z', '+00:00')
                                            try:
                                                dt = datetime.fromisoformat(s)
                                                if dt.tzinfo is None:
                                                    dt = dt.replace(tzinfo=timezone.utc)
                                            except Exception:
                                                dt = None
                                        if dt is not None:
                                            et = dt.astimezone(ZoneInfo('America/New_York'))
                                            game_date = et.strftime('%Y-%m-%d')
                                            game_time = et.strftime('%-I:%M %p ET')
                                    except Exception:
                                        pass

                                    home_info = teams.get("home", {})
                                    away_info = teams.get("away", {})

                                    home_team = home_info.get("name") or home_info.get("abbreviation", "")
                                    away_team = away_info.get("name") or away_info.get("abbreviation", "")

                                    if home_team and away_team:
                                        matchup = f"{away_team} @ {home_team}"
                                    elif home_team or away_team:
                                        player_team = team if team else (home_team or away_team)
                                        opponent = away_team if home_team == player_team else home_team
                                        if opponent and player_team != opponent:
                                            matchup = f"{player_team} vs {opponent}"
                                else:
                                    matchup = "TBD vs TBD"
                            else:
                                if team:
                                    matchup = f"{team} vs TBD"
                        except Exception:
                            if team:
                                matchup = f"{team} vs TBD"
                            else:
                                matchup = "TBD vs TBD"

                        # Detect demon props for Playwright flow as well
                        desc = str(attr.get("description") or "")
                        allow_under = not is_demon_prop(projection_type, desc)

                        items.append(
                            {
                                "Name": player_name,
                                "Points": line_val,
                                "Prop": projection_type,
                                "League": league,
                                "Team": team,
                                "Matchup": matchup,
                                "Home_Team": home_team,
                                "Away_Team": away_team,
                                "Game_Date": game_date,
                                "Game_Time": game_time,
                                "Last_Updated": datetime.utcnow()
                                .replace(tzinfo=timezone.utc)
                                .isoformat()
                                .replace('+00:00', 'Z'),
                                "Allow_Under": allow_under,
                            }
                        )
                    except Exception:
                        continue

        except Exception:
            pass
        finally:
            try:
                if context:
                    context.close()
            except Exception:
                pass
            try:
                if browser:
                    browser.close()
            except Exception:
                pass

    return items

def maybe_scrape_with_selenium(output_csv: str) -> bool:
    """Enhanced Selenium scraper with stealth techniques."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
    except Exception:
        return False

    driver = None
    try:
        # Enhanced Chrome options for stealth
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument(f"--user-data-dir=/tmp/selenium-profile-{int(time.time())}")
        
        driver = webdriver.Chrome(options=options)
        
        # Execute stealth script
        driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            window.chrome = {
                runtime: {},
            };
        """)
        
        wait = WebDriverWait(driver, 20)
        
        # Random delay before navigation
        time.sleep(random.uniform(1, 3))
        driver.get("https://app.prizepicks.com/")
        
        # Wait for page load with random delay
        time.sleep(random.uniform(3, 6))

        # Close any popups
        try:
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "close")))
            buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Close') or contains(@class,'close')]")
            if buttons:
                buttons[0].click()
                time.sleep(random.uniform(1, 2))
        except Exception:
            pass

        # Try to find projections using multiple selectors
        projection_selectors = [
            ".projection",
            "[data-testid*='projection']",
            ".pick-card",
            ".prop-card",
            "[class*='pick']",
            "[class*='prop']"
        ]
        
        rows = []
        
        for selector in projection_selectors:
            try:
                projections = driver.find_elements(By.CSS_SELECTOR, selector)
                if projections:
                    for proj in projections[:20]:  # Limit to first 20
                        try:
                            # Try multiple ways to extract data
                            name_selectors = [".name", ".player-name", "[data-testid*='name']", ".player", "[class*='name']"]
                            pts_selectors = [".presale-score", ".line", ".points", "[data-testid*='line']", "[class*='line']"]
                            type_selectors = [".text", ".stat-type", ".prop-type", "[data-testid*='stat']", "[class*='stat']"]
                            
                            name = "Unknown"
                            pts = "0"
                            ptype = "Unknown"
                            
                            for name_sel in name_selectors:
                                try:
                                    name_elem = proj.find_element(By.CSS_SELECTOR, name_sel)
                                    name = name_elem.text or name_elem.get_attribute('innerHTML')
                                    if name:
                                        break
                                except Exception:
                                    continue
                            
                            for pts_sel in pts_selectors:
                                try:
                                    pts_elem = proj.find_element(By.CSS_SELECTOR, pts_sel)
                                    pts = pts_elem.text or pts_elem.get_attribute('innerHTML')
                                    if pts:
                                        break
                                except Exception:
                                    continue
                            
                            for type_sel in type_selectors:
                                try:
                                    type_elem = proj.find_element(By.CSS_SELECTOR, type_sel)
                                    ptype = type_elem.text or type_elem.get_attribute('innerHTML')
                                    if ptype:
                                        break
                                except Exception:
                                    continue
                            
                            if name != "Unknown" and pts != "0":
                                # Apply mapping for NFL rookies or miscategorized players
                                league = ""
                                team = ""
                                nfl_info = get_nfl_info(name)
                                if nfl_info:
                                    league, team = nfl_info
                                    rows.append({
                                        "Name": name, 
                                        "Points": pts, 
                                        "Prop": ptype, 
                                        "League": league, 
                                        "Team": team,
                                        "Matchup": "TBD vs TBD",  # Selenium can't easily extract game relationships
                                        "Home_Team": "",
                                        "Away_Team": ""
                                    })
                                
                        except Exception:
                            continue
                    
                    if rows:
                        break  # Found data, stop trying other selectors
                        
            except Exception:
                continue

        if rows:
            df = pd.DataFrame(rows)
            # Ensure consistent columns
            for col in ["Name", "Points", "Prop", "League", "Sport", "Team", "Matchup", "Home_Team", "Away_Team"]:
                if col not in df.columns:
                    df[col] = ""
            # Ensure new columns exist
            for col in ["Game_Date","Game_Time","Last_Updated"]:
                if col not in df.columns:
                    df[col] = ""
            df = df[["Name", "Points", "Prop", "League", "Sport", "Team", "Matchup", "Home_Team", "Away_Team", "Game_Date", "Game_Time", "Last_Updated"]]
            df.to_csv(output_csv, index=False)
            return True
        return False
        
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


def fallback_to_test_data(output_csv: str) -> bool:
    """Stub: Disabled. No sample/test data will be generated or written."""
    return False


def main():
    output_csv = os.environ.get("PRIZEPICKS_CSV", "prizepicks_props.csv")
    # Optional sport hint to scope the query tightly
    sport_hint = os.environ.get("PRIZEPICKS_SPORT_PARAM") or None
    
    print("🎯 Starting PrizePicks scraper with bypass techniques...")
    
    # Strategy 1: Enhanced API approach with multiple endpoints
    print("📡 Trying enhanced API approach...")
    items = fetch_prizepicks_api(sport=sport_hint)
    if items:
        print(f"✅ API success! Found {len(items)} props")
    else:
        print("❌ API blocked/failed")
    
    # Strategy 2: Stealth Playwright
    if not items:
        print("🎭 Trying stealth Playwright...")
        items = fetch_prizepicks_playwright()
        if items:
            print(f"✅ Playwright success! Found {len(items)} props")
        else:
            print("❌ Playwright blocked/failed")
    
    # Strategy 3: Enhanced Selenium
    if not items:
        print("🤖 Trying enhanced Selenium...")
        success = maybe_scrape_with_selenium(output_csv)
        if success:
            print("✅ Selenium success!")
            print(f"Props saved to: {output_csv}")
            return
        else:
            print("❌ Selenium blocked/failed")
    
    # Strategy 4: Fallback to realistic test data (gated)
    if not items:
        PRIZEPICKS_ONLY = os.environ.get("PRIZEPICKS_ONLY", "true").lower() in ("1", "true", "yes")
        if PRIZEPICKS_ONLY:
            # Strict mode: do NOT generate any test data. Leave CSV untouched.
            print("🚫 Strict mode (PRIZEPICKS_ONLY) enabled — skipping test-data fallback. No props written.")
        else:
            print("📋 Using realistic test data as fallback...")
            success = fallback_to_test_data(output_csv)
            if success:
                print("✅ Test data generated successfully!")
                print(f"Props saved to: {output_csv}")
                return

    # Process successful API/Playwright results
    if items:
        df = pd.DataFrame(items)
        # Normalize Points column to float where possible
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None
        df["Points"] = df["Points"].map(to_float)
        
        # Write atomically to avoid partial reads while the app is auto-refreshing
        tmp_path = f"{output_csv}.tmp"
        # Ensure new columns exist
        for col in ["Game_Date","Game_Time","Last_Updated","Allow_Under","Sport"]:
            if col not in df.columns:
                df[col] = ""
        df.to_csv(tmp_path, index=False)
        try:
            os.replace(tmp_path, output_csv)
        except Exception:
            # Fallback to non-atomic write if replace not available
            df.to_csv(output_csv, index=False)
        
        print(f"✅ Props scraped and saved: {output_csv}")
        print(f"📊 Found {len(df)} total props")
        print("\n🎯 Sample props:")
        print(df.head(10).to_string(index=False))
    else:
        PRIZEPICKS_ONLY = os.environ.get("PRIZEPICKS_ONLY", "true").lower() in ("1", "true", "yes")
        if PRIZEPICKS_ONLY:
            print("❌ All scraping methods failed and strict mode is enabled — no fallback data will be written.")
        else:
            print("❌ All scraping methods failed, using fallback data")
            fallback_to_test_data(output_csv)


if __name__ == "__main__":
    main()
