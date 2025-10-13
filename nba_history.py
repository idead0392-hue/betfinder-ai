"""
NBA historical data utilities using nbastatpy.
Provides a light wrapper for agents to fetch player/team/season data
and enrich prop items with recent averages.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Any, Optional


@lru_cache(maxsize=256)
def get_player_history(player_query: str, season: str = "2023", playoffs: bool = False) -> Dict[str, Any]:
    """Return key historical frames for a player.

    player_query: string like "Giannis" or full name; resolved by nbastatpy
    season: season year string, e.g., "2023"
    playoffs: True for playoff stats/logs
    """
    import time
    from nbastatpy.player import Player
    MAX_RETRIES = 10
    RETRY_DELAY = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            p = Player(player_query, season_year=season, playoffs=playoffs)
            out = {
                "career_stats": p.get_season_career_totals(),
                "awards": p.get_awards(),
                "game_logs": p.get_games_boxscore(),
            }
            if all(out.values()):
                return out
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(RETRY_DELAY)
    raise RuntimeError("Cannot fetch NBA player history with real data - aborting.")


@lru_cache(maxsize=256)
def get_team_history(team_query: str, season: str = "2023") -> Dict[str, Any]:
    if os.getenv('DISABLE_NBA_HISTORY', '0') == '1':
        return {"team_stats": None, "roster": None}
    from nbastatpy.team import Team
    t = Team(team_query, season_year=season)
    out = {
        "team_stats": None,
        "roster": None,
    }
    import time
    MAX_RETRIES = 10
    RETRY_DELAY = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            t = Team(team_query, season_year=season)
            out = {
                "team_stats": t.get_player_stats(),
                "roster": t.get_roster(),
            }
            if all(out.values()):
                return out
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(RETRY_DELAY)
    raise RuntimeError("Cannot fetch NBA team history with real data - aborting.")


def get_season_league_stats(season: str = "2023"):
    # nbastatpy Season API may vary; keep best-effort import
    if os.getenv('DISABLE_NBA_HISTORY', '0') == '1':
        return None
    import time
    MAX_RETRIES = 10
    RETRY_DELAY = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            from nbastatpy.season import Season
            s = Season(season_year=season)
            stats = s.get_league_stats()
            if stats:
                return stats
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(RETRY_DELAY)
    raise RuntimeError("Cannot fetch NBA league stats with real data - aborting.")


def enrich_prop_data_with_history(prop: Dict[str, Any], season: Optional[str] = None, playoffs: bool = False) -> Dict[str, Any]:
    """Add recent averages for points/assists to a prop dict using nbastatpy game logs.

    Expected prop keys:
    - player_name or player
    - event season (optional), or pass explicit season param
    """
    if os.getenv('DISABLE_NBA_HISTORY', '0') == '1':
        return prop

    player_name = prop.get("player_name") or prop.get("player")
    if not player_name:
        return prop

    use_season = season or prop.get("event_season") or "2023"
    try:
        data = get_player_history(player_name, season=use_season, playoffs=playoffs)
        logs = data.get("game_logs")
        if logs is not None and len(logs) > 0:
            # DataFrame expected; compute basic means for common stats if present
            for col, out_key in (("PTS", "history_avg_pts"), ("AST", "history_avg_ast"), ("REB", "history_avg_reb")):
                if col in logs.columns:
                    prop[out_key] = float(logs[col].mean())
    except Exception:
        # Leave prop unchanged on failure
        return prop

    return prop
