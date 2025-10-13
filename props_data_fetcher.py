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
        # Optional path to cached CSV from previous scrape
        self.pp_csv_path = os.getenv('PRIZEPICKS_CSV_PATH', 'prizepicks_props.csv')
    
    def fetch_all_props(self, max_props: int = 250) -> List[Dict]:
        """Fetch props from supported sources and return normalized list."""
        all_props: List[Dict] = []

        # PrizePicks first
        try:
            pp = self.fetch_prizepicks_props(max_items=max_props)
            all_props.extend(pp)
        except Exception as e:
            print(f"⚠️ Error fetching PrizePicks: {e}")

        # Underdog (optional; currently returns empty list)
        try:
            ud = self.fetch_underdog_props(max_items=max(0, max_props - len(all_props)))
            all_props.extend(ud)
        except Exception as e:
            print(f"⚠️ Error fetching Underdog: {e}")

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

        # If CSV empty, try API
        if not items and fetch_prizepicks_api:
            try:
                items = fetch_prizepicks_api()
                # print(f"🌐 Loaded {len(items)} PrizePicks rows from API")
            except Exception as e:
                print(f"⚠️ PrizePicks API error: {e}")

        # Normalize
        props: List[Dict] = []
        for it in items[:max_items]:
            norm = self._normalize_prizepicks_item(it)
            if norm:
                props.append(norm)
        return props

    def fetch_underdog_props(self, max_items: int = 500) -> List[Dict]:
        """Placeholder for Underdog fetch. Currently returns empty list."""
        return []
    
    def _normalize_prizepicks_item(self, item: Dict[str, Any]) -> Optional[Dict]:
        """Normalize a PrizePicks row (CSV or API) into internal prop format."""
        try:
            # Keys may be different casing depending on source
            name = item.get('Name') or item.get('name') or item.get('player_name') or ''
            line = item.get('Points') or item.get('line') or item.get('Line') or ''
            prop = item.get('Prop') or item.get('prop') or ''
            league = item.get('League') or item.get('league') or ''
            sport = item.get('Sport') or item.get('sport') or ''
            team = item.get('Team') or item.get('team') or ''
            matchup = item.get('Matchup') or item.get('matchup') or ''
            game_date = item.get('Game_Date') or ''
            game_time = item.get('Game_Time') or ''
            last_updated = item.get('Last_Updated') or ''

            # Compose start_time (best-effort ISO)
            start_time = ''
            try:
                if game_date and game_time:
                    # We don't have timezone offset; keep as plain string
                    start_time = f"{game_date} {game_time}"
                elif last_updated:
                    start_time = str(last_updated)
            except Exception:
                start_time = str(last_updated or '')

            # Parse numeric line
            try:
                line_val = float(line) if str(line).strip() != '' else 0.0
            except Exception:
                line_val = 0.0

            # Normalize sport label (keep as-is; agents handle mapping)
            over_under = None  # Not derivable reliably from PP data

            return {
                'player_name': name,
                'team': team,
                'pick': prop,
                'stat_type': str(prop).lower(),
                'line': line_val,
                'odds': -110,
                'confidence': 70.0,
                'expected_value': 0.0,
                'avg_l10': 0.0,
                'start_time': start_time,
                'sport': sport,
                'league': league,
                'matchup': matchup,
                'over_under': over_under,
            }
        except Exception:
            return None
        
        return 0.0  # Default


# Global instance
props_fetcher = PropsDataFetcher()