"""
Unified Flashscore data utility for agents and ML.

Key functions:
- fetch_sport_data(sport, league, season, country, odds=True, batch_size=100)
- fetch_from_config(config_path, section)

Environment toggles:
- DISABLE_FLASHSCORE_SCRAPE=1 to bypass network calls (returns empty DataFrame)

Notes:
- Depends on `flashscore-scraper` (FlexibleScraper) and pandas.
- Wraps calls with safe fallbacks and minimal logging.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

DISABLE = os.getenv("DISABLE_FLASHSCORE_SCRAPE", "0") == "1"

try:
    from flashscore_scraper import FlexibleScraper
    _HAS_FS = True
except Exception as e:
    FlexibleScraper = None  # type: ignore
    _HAS_FS = False


def _safe_empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def fetch_sport_data(sport: str, league: str, season: str, country: str,
                     odds: bool = True, batch_size: int = 100) -> pd.DataFrame:
    """Fetch data from Flashscore as a pandas DataFrame.

    Mirrors the user-provided example while adding safety and toggles.
    """
    if DISABLE or not _HAS_FS:
        return _safe_empty_df()

    filters: Dict[str, Any] = {
        "sports": [sport],
        "leagues": [league],
        "seasons": [season],
        "countries": [country],
    }
    try:
        scraper = FlexibleScraper(filters=filters)
        df = scraper.scrape(headless=True, batch_size=batch_size, scrape_odds=odds)
        if not isinstance(df, pd.DataFrame):
            return _safe_empty_df()
        return df
    except Exception as e:
        print(f"[flashscore_data] warning: scrape failed: {e}")
        return _safe_empty_df()


def fetch_from_config(config_path: str, section: str = "default") -> pd.DataFrame:
    """Fetch using a YAML config entry.

    YAML structure example:
    default:
      sport: football
      league: Premier League
      season: "2023/2024"
      country: England
      odds: true
      batch_size: 100
    """
    try:
        import yaml
    except Exception:
        print("[flashscore_data] warning: PyYAML not installed; returning empty DataFrame")
        return _safe_empty_df()

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        entry = cfg.get(section) or {}
        return fetch_sport_data(
            sport=str(entry.get("sport", "")),
            league=str(entry.get("league", "")),
            season=str(entry.get("season", "")),
            country=str(entry.get("country", "")),
            odds=bool(entry.get("odds", True)),
            batch_size=int(entry.get("batch_size", 100)),
        )
    except Exception as e:
        print(f"[flashscore_data] warning: config fetch failed: {e}")
        return _safe_empty_df()
