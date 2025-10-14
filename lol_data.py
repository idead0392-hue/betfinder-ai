"""
League of Legends data wrapper built on top of leaguepedia_parser_thomasbarrepitous.

Goals:
- Provide stable, cached helper functions for agents (no crashes on failures)
- Make network usage optional via env: DISABLE_EXTERNAL_LOL_DATA=1 to force safe fallbacks
- Normalize outputs into simple python dicts/lists for ease of use in analysis

Exposed helpers:
- get_regions()
- get_tournaments(region: str | None = None, year: int | None = None)
- get_player_match_history(player: str, limit: int = 10)
- get_team_match_performance(team: str, tournament: str | None = None)
- get_tournament_standings(tournament: str)
- get_tournament_mvp_candidates(tournament: str, min_games: int = 10)
- get_role_performance_comparison(tournament: str, role: str)
- get_champions_by_attributes(attr: str)
- get_champion_performance_stats(champion: str, tournament: str | None = None)

All functions return empty lists/dicts on failure and log a concise message.
"""
from __future__ import annotations

import os
import functools
from typing import Any, Dict, List, Optional

DISABLE = os.getenv("DISABLE_EXTERNAL_LOL_DATA", "0") == "1"

try:
    import leaguepedia_parser_thomasbarrepitous as lp
    _PKG_AVAILABLE = True
except Exception:
    lp = None  # type: ignore
    _PKG_AVAILABLE = False


def _should_bypass() -> bool:
    return DISABLE or not _PKG_AVAILABLE


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        # Keep logs minimal to avoid noisy UI
        print(f"[lol_data] warning: {getattr(fn, '__name__', 'call')} failed: {e}")
        return None


def _cache(ttl_seconds: int = 900):
    """Simple cache decorator using functools.lru_cache with optional TTL.
    For now, we rely on lru_cache only; agents typically refresh per session.
    """
    def decorator(func):
        cached = functools.lru_cache(maxsize=64)(func)
        return cached
    return decorator


@_cache()
def get_regions() -> List[str]:
    if _should_bypass():
        return []
    res = _safe_call(lp.get_regions)
    return list(res or [])


@_cache()
def get_tournaments(region: Optional[str] = None, year: Optional[int] = None) -> List[str]:
    if _should_bypass():
        return []
    if region and year:
        res = _safe_call(lp.get_tournaments, region, year)
    elif region:
        res = _safe_call(lp.get_tournaments, region)
    else:
        res = _safe_call(lp.get_tournaments)
    return list(res or [])


def _normalize_match(m: Any) -> Dict[str, Any]:
    # Accept objects with attributes or dict-like structures
    d: Dict[str, Any] = {}
    for key in [
        "champion", "kda_ratio", "kills", "deaths", "assists",
        "performance_grade", "did_win", "gold_share", "kill_participation",
        "date", "opponent", "tournament",
    ]:
        val = getattr(m, key, None)
        if val is None and isinstance(m, dict):
            val = m.get(key)
        d[key] = val
    return d


@_cache()
def get_player_match_history(player: str, limit: int = 10) -> List[Dict[str, Any]]:
    if _should_bypass() or not player:
        return []
    res = _safe_call(lp.get_player_match_history, player, limit=limit)
    if not res:
        return []
    return [_normalize_match(m) for m in res][:limit]


@_cache()
def get_team_match_performance(team: str, tournament: Optional[str] = None) -> List[Dict[str, Any]]:
    if _should_bypass() or not team:
        return []
    res = _safe_call(lp.get_team_match_performance, team, tournament=tournament)
    if not res:
        return []
    return [_normalize_match(m) for m in res]


@_cache()
def get_tournament_standings(tournament: str) -> List[Dict[str, Any]]:
    if _should_bypass() or not tournament:
        return []
    res = _safe_call(lp.get_tournament_standings, tournament)
    return list(res or [])


@_cache()
def get_tournament_mvp_candidates(tournament: str, min_games: int = 10) -> List[Dict[str, Any]]:
    if _should_bypass() or not tournament:
        return []
    res = _safe_call(lp.get_tournament_mvp_candidates, tournament, min_games=min_games)
    return list(res or [])


@_cache()
def get_role_performance_comparison(tournament: str, role: str) -> List[Dict[str, Any]]:
    if _should_bypass() or not tournament or not role:
        return []
    res = _safe_call(lp.get_role_performance_comparison, tournament, role)
    return list(res or [])


@_cache()
def get_champions_by_attributes(attr: str) -> List[str]:
    if _should_bypass() or not attr:
        return []
    res = _safe_call(lp.get_champions_by_attributes, attr)
    return list(res or [])


@_cache()
def get_champion_performance_stats(champion: str, tournament: Optional[str] = None) -> List[Dict[str, Any]]:
    if _should_bypass() or not champion:
        return []
    res = _safe_call(lp.get_champion_performance_stats, champion, tournament=tournament)
    if not res:
        return []
    return [_normalize_match(m) for m in res]


def summarize_player_recent_form(player: str, limit: int = 10) -> Dict[str, Any]:
    """Quick summary for agent scoring: avg KDA, KP, gold share, top champions.
    Returns empty dict if data unavailable.
    """
    history = get_player_match_history(player, limit=limit)
    if not history:
        return {}
    def safe(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    avg_kda = safe([m.get("kda_ratio") for m in history])
    avg_kp = safe([m.get("kill_participation") for m in history])
    avg_gold_share = safe([m.get("gold_share") for m in history])
    top_champs: List[str] = []
    for m in history:
        c = m.get("champion")
        if c and c not in top_champs:
            top_champs.append(c)
        if len(top_champs) >= 3:
            break

    return {
        "avg_kda": avg_kda,
        "avg_kill_participation": avg_kp,
        "avg_gold_share": avg_gold_share,
        "top_champions": top_champs,
        "sample": len(history),
    }
