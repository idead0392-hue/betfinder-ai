"""
Props Data Fetcher

This module handles fetching betting props from live data sources
with a focus on PrizePicks and future providers.
Normalizes data into a consistent format for the application.
"""

from __future__ import annotations
import os
import csv
from datetime import datetime
from typing import List, Dict, Optional, Any
import subprocess
import json

# External source for PrizePicks scraping/API
try:
    from prizepicks_scrape import fetch_prizepicks_api
except Exception:
    fetch_prizepicks_api = None  # Will fall back to CSV


def is_event_time_valid(event_time: str) -> bool:
    """Check if event time is current/future (not stale)"""
    try:
        if not event_time:
            return False
        
        # Parse event time (handle multiple formats)
        event_dt = None
        for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M']:
            try:
                event_dt = datetime.strptime(event_time[:19], fmt)
                break
            except:
                continue
        
        if not event_dt:
            return False
        
        # Event must be in the future or within last 6 hours (for live events)
        now = datetime.utcnow()
        time_diff = (event_dt - now).total_seconds()
        
        return time_diff > -21600  # -6 hours to 24+ hours in future
    except:
        return False


class PropsDataFetcher:
    """
    Live data fetcher for betting props from PrizePicks and other providers.
    Provides normalized prop data with current/future event times only.
    """

    def __init__(self):
        # List of sportsbook Playwright scripts
        self.sportsbook_scripts = [
            'scrape_sportsbook.js',  # Example script path
            # Add more scripts for other sportsbooks as needed
        ]
    
    def fetch_all_props(self, max_props: int = 250) -> List[Dict]:
        """Fetch props from all sportsbook Playwright scripts and return normalized list."""
        all_props: List[Dict] = []

        # Sportsbook Playwright scripts only
        for script_path in self.sportsbook_scripts:
            try:
                sb_props = self.fetch_sportsbook_props(script_path, max_items=max(0, max_props - len(all_props)))
                all_props.extend(sb_props)
            except Exception as e:
                print(f"⚠️ Error fetching props from {script_path}: {e}")

        # Filter by valid time
        all_props = [p for p in all_props if is_event_time_valid(p.get('start_time', '')) or p.get('start_time') == '']

        # Limit
        return all_props[:max_props]

    def fetch_prizepicks_props(self, max_items: int = 1000) -> List[Dict]:
        """Fetch PrizePicks props from CSV cache or API and normalize."""
        items: List[Dict] = []

        # Try CSV cache first (fast path)
        if os.path.exists(self.pp_csv_path):
            try:
                with open(self.pp_csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        items.append(row)
                # print(f"📄 Loaded {len(items)} PrizePicks rows from CSV")
            except Exception as e:
                print(f"⚠️ Failed to read CSV {self.pp_csv_path}: {e}")