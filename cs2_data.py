"""
CS2 data wrapper using cs2api.
- Provides async helpers to fetch today's, live, and finished matches
- Fetch player and team stats
- Exposes sync entry points via asyncio.run for callers in sync contexts
- Safe fallbacks and simple in-memory caching
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

try:
    from cs2api import CS2
except Exception:  # cs2api not installed or import error
    CS2 = None  # type: ignore

# Lightweight TTL cache
_CACHE: Dict[str, Dict[str, Any]] = {}
_DEFAULT_TTL = 60  # seconds


def _cache_get(key: str) -> Optional[Any]:
    ent = _CACHE.get(key)
    if not ent:
        return None
    if time.time() - ent.get("ts", 0) > ent.get("ttl", _DEFAULT_TTL):
        return None
    return ent.get("val")


def _cache_set(key: str, val: Any, ttl: int = _DEFAULT_TTL) -> None:
    _CACHE[key] = {"val": val, "ts": time.time(), "ttl": ttl}


async def _aget_cs2_data() -> Dict[str, Any]:
    if CS2 is None:
        return {"today": [], "live": [], "recent": [], "player_stats": {}, "team_stats": {}}

    timeout = int(os.getenv("CS2API_TIMEOUT", "30"))

    async with CS2(timeout=timeout) as cs2:
        out: Dict[str, Any] = {}
        try:
            out["today"] = await cs2.get_todays_matches()
        except Exception:
            out["today"] = []
        try:
            out["live"] = await cs2.get_live_matches()
        except Exception:
            out["live"] = []
        try:
            out["recent"] = await cs2.finished()
        except Exception:
            out["recent"] = []
        return out


def get_cs2_matches(force_refresh: bool = False) -> Dict[str, Any]:
    """Sync helper to get cached CS2 matches summary."""
    key = "cs2_matches"
    if not force_refresh:
        cached = _cache_get(key)
        if cached is not None:
            return cached
    try:
        val = asyncio.run(_aget_cs2_data())
    except RuntimeError:
        # If already in an event loop (unlikely in our usage), create a new loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            val = loop.run_until_complete(_aget_cs2_data())
        finally:
            loop.close()
    _cache_set(key, val, ttl=_DEFAULT_TTL)
    return val


async def _aget_player_and_team(player: Optional[str] = None, team: Optional[str] = None) -> Dict[str, Any]:
    if CS2 is None:
        return {"player_stats": {}, "team_stats": {}}
    timeout = int(os.getenv("CS2API_TIMEOUT", "30"))
    async with CS2(timeout=timeout) as cs2:
        out: Dict[str, Any] = {}
        if player:
            try:
                out["player_stats"] = await cs2.get_player_stats(player)
            except Exception:
                out["player_stats"] = {}
        if team:
            try:
                out["team_stats"] = await cs2.get_team_stats(team)
            except Exception:
                out["team_stats"] = {}
        return out


def get_player_and_team(player: Optional[str] = None, team: Optional[str] = None) -> Dict[str, Any]:
    key = f"cs2_pt:{player}:{team}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        val = asyncio.run(_aget_player_and_team(player, team))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            val = loop.run_until_complete(_aget_player_and_team(player, team))
        finally:
            loop.close()
    _cache_set(key, val, ttl=120)
    return val


def enrich_csgo_prop_with_cs2(prop: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort enrichment of a csgo prop with cs2api context.
    - Adds fields: cs2_match_status, cs2_last_updated, cs2_recent_form (if resolvable)
    """
    try:
        matches = get_cs2_matches()
        prop_player = (prop.get("player_name") or prop.get("player") or "").strip()
        prop_matchup = (prop.get("matchup") or "").strip().lower()

        # live indicator
        live = matches.get("live") or []
        is_live = False
        for m in live:
            text = str(m).lower()
            if prop_player and prop_player.lower() in text:
                is_live = True
                break
            if prop_matchup and prop_matchup and prop_matchup in text:
                is_live = True
                break
        prop["cs2_match_status"] = "live" if is_live else "scheduled"
        prop["cs2_last_updated"] = int(time.time())

        # basic recent form proxy
        recent = matches.get("recent") or []
        prop["cs2_recent_form"] = len(recent)
    except Exception:
        pass
    return prop
