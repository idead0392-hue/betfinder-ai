import os
import sys
import json
import time
import csv
from typing import List, Dict

import pandas as pd
import requests


API_URL = "https://api.prizepicks.com/projections"


def fetch_prizepicks_api(sport: str = None, league_id: str = None) -> List[Dict]:
    """Fetch projections from PrizePicks public API. Optionally filter by league or sport.
    Returns a list of dicts with Name, Points, Prop (generic fields to align with UI expectation).
    """
    params = {
        "page": 1,
        "per_page": 250
    }
    if league_id:
        params["league_id"] = league_id
    if sport:
        params["sport"] = sport

    headers = {
        "Accept": "application/json",
        "Origin": "https://app.prizepicks.com",
        "Referer": "https://app.prizepicks.com/",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    }

    items: List[Dict] = []
    for page in range(1, 8):  # paginate defensively up to ~1750 entries
        params["page"] = page
        try:
            resp = requests.get(API_URL, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            # stop on first failure (network or block)
            break
        data = resp.json()
        included = {obj["id"]: obj for obj in data.get("included", []) if isinstance(obj, dict) and obj.get("type")}

        for proj in data.get("data", []):
            try:
                attr = proj.get("attributes", {})
                league_id_fk = attr.get("league_id")
                projection_type = attr.get("stat_type") or attr.get("type") or ""
                line_val = attr.get("line_score") or attr.get("line") or None
                player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
                player_id = player_rel.get("id") if isinstance(player_rel, dict) else None
                player = included.get(player_id) if player_id else None
                player_name = (player or {}).get("attributes", {}).get("name") or "Unknown"

                items.append({
                    "Name": player_name,
                    "Points": line_val,
                    "Prop": projection_type
                })
            except Exception:
                continue

        # stop if fewer than requested per_page returned
        if len(data.get("data", [])) < params["per_page"]:
            break

    return items


def fetch_prizepicks_playwright() -> List[Dict]:
    """Use Playwright headless Chromium to load the web app and capture projections from network responses."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return []
    items: List[Dict] = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            captured_json = []

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

            page.goto("https://app.prizepicks.com/", wait_until="networkidle")
            # Give a small buffer for late XHRs
            page.wait_for_timeout(2000)

            # Optionally click a common sport to trigger specific projections
            try:
                page.locator("text=NBA").first.click()
                page.wait_for_timeout(1500)
            except Exception:
                pass

            # Aggregate all captured pages of projections
            for data in captured_json:
                included = {obj["id"]: obj for obj in data.get("included", []) if isinstance(obj, dict) and obj.get("type")}
                for proj in data.get("data", []):
                    try:
                        attr = proj.get("attributes", {})
                        projection_type = attr.get("stat_type") or attr.get("type") or ""
                        line_val = attr.get("line_score") or attr.get("line") or None
                        player_rel = proj.get("relationships", {}).get("new_player", {}).get("data", {})
                        player_id = player_rel.get("id") if isinstance(player_rel, dict) else None
                        player = included.get(player_id) if player_id else None
                        player_name = (player or {}).get("attributes", {}).get("name") or "Unknown"

                        items.append({
                            "Name": player_name,
                            "Points": line_val,
                            "Prop": projection_type
                        })
                    except Exception:
                        continue

            context.close()
            browser.close()
        except Exception:
            try:
                browser.close()
            except Exception:
                pass
    return items

def maybe_scrape_with_selenium(output_csv: str) -> bool:
    """Optional Selenium fallback if API blocked. Returns True if CSV written, else False."""
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
        # Use headless Chrome with a unique user data dir to avoid conflicts
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-data-dir=/tmp/selenium-profile-{int(time.time())}")
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 15)
        driver.get("https://app.prizepicks.com/")

        # Close pop-up if present
        try:
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "close")))
            # conservative locator for close button if exists
            buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Close') or contains(@class,'close')]")
            if buttons:
                buttons[0].click()
                time.sleep(1)
        except Exception:
            pass

        # Gather stat categories if available
        try:
            wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "stat-container")))
            cats_el = driver.find_element(By.CSS_SELECTOR, ".stat-container")
            categories = [t.strip() for t in cats_el.text.split("\n") if t.strip()]
        except Exception:
            categories = []

        rows = []
        if not categories:
            # if categories not found, attempt to scrape projections blocks directly
            projections = driver.find_elements(By.CSS_SELECTOR, ".projection")
            for proj in projections:
                try:
                    name = proj.find_element(By.CLASS_NAME, "name").text
                    pts = proj.find_element(By.CLASS_NAME, "presale-score").get_attribute('innerHTML')
                    ptype = proj.find_element(By.CLASS_NAME, "text").get_attribute('innerHTML')
                    rows.append({"Name": name, "Points": pts, "Prop": ptype})
                except Exception:
                    continue
        else:
            for category in categories:
                try:
                    el = driver.find_element(By.XPATH, f"//div[text()='{category}']")
                    el.click()
                    projections = WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".projection"))
                    )
                    for proj in projections:
                        try:
                            name = proj.find_element(By.CLASS_NAME, "name").text
                            pts = proj.find_element(By.CLASS_NAME, "presale-score").get_attribute('innerHTML')
                            ptype = proj.find_element(By.CLASS_NAME, "text").get_attribute('innerHTML')
                            rows.append({"Name": name, "Points": pts, "Prop": ptype})
                        except Exception:
                            continue
                except Exception:
                    continue

        if rows:
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            return True
        return False
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


def main():
    output_csv = os.environ.get("PRIZEPICKS_CSV", "prizepicks_props.csv")
    items = fetch_prizepicks_api()
    if not items:
        # Try Playwright network capture
        items = fetch_prizepicks_playwright()
    if not items:
        # Try Selenium fallback
        ok = maybe_scrape_with_selenium(output_csv)
        if ok:
            print(f"Props scraped via Selenium and saved: {output_csv}")
            return
        print("Failed to fetch PrizePicks props via API, Playwright, and Selenium.")
        sys.exit(1)

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
    df.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, output_csv)
    except Exception:
        # Fallback to non-atomic write if replace not available
        df.to_csv(output_csv, index=False)
    print(f"Props scraped and saved: {output_csv}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
