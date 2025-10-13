"""
PrizePicks provider adapter for BetFinder-AI.

Behavior:
- Delegates scraping to a local module named `prizepicks_scraper` if present.
- Normalizes result items into a canonical schema used by the app.
- Returns a helpful error response if no scraper is available.
"""

import logging
import time
from typing import Any, Dict, List, Iterable, Optional

# Try to import repository's APIResponse / BaseAPIProvider if available
try:
    from api_providers import BaseAPIProvider, APIResponse, SportType
except Exception:
    # Fallback lightweight shim
    class APIResponse:
        @staticmethod
        def success_response(data=None, response_time=None, status_code: int = 200):
            return type("R", (), {"success": True, "data": data, "response_time": response_time, "status_code": status_code})

        @staticmethod
        def error_response(error_message: str, response_time: float = None, status_code: int = 500):
            return type("R", (), {"success": False, "error_message": error_message, "response_time": response_time, "status_code": status_code})

    class BaseAPIProvider:
        def __init__(self, provider_name: str = "prizepicks", base_url: str = ""):
            self.provider_name = provider_name
            self.base_url = base_url

    SportType = Optional[str]

logger = logging.getLogger(__name__)


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw scraped item into the repo's expected props schema.
    """
    player = item.get("player") or item.get("name") or item.get("athlete")
    team = item.get("team") or item.get("team_abbr") or item.get("team_short")
    matchup = item.get("matchup") or item.get("game") or item.get("event")
    stat_type = item.get("stat_type") or item.get("market") or item.get("stat")
    market = item.get("market") or stat_type
    line = item.get("line")
    if line is None:
        for k in ("projection", "value", "number", "points"):
            if k in item:
                line = item.get(k)
                break
    try:
        line = float(line) if line is not None else None
    except Exception:
        line = None
    return {
        "player": player,
        "team": team,
        "matchup": matchup,
        "stat_type": stat_type,
        "market": market,
        "line": line,
        "book": item.get("book") or item.get("source") or "PrizePicks",
        "raw": item,
        "timestamp": item.get("timestamp")
    }


class PrizePicksProvider(BaseAPIProvider):
    """
    PrizePicksProvider: adapter that calls your local PrizePicks scraping function,
    then normalizes and returns the props.
    """
    def __init__(self, scraper_module: Optional[str] = "prizepicks_scraper"):
        super().__init__(provider_name="prizepicks", base_url="")
        self._scraper_module_name = scraper_module
        self._scraper = None
        self._scraper_func_name_candidates = (
            "get_prizepicks_props",
            "fetch_prizepicks_props",
            "scrape_prizepicks_props",
            "get_props",
            "fetch_props",
        )
        self._init_scraper()

    def _init_scraper(self):
        """
        Attempt to import the user's scraping module. If not present, leave self._scraper None.
        """
        try:
            if not self._scraper_module_name:
                return
            mod = __import__(self._scraper_module_name, fromlist=["*"])
            for fn in self._scraper_func_name_candidates:
                if hasattr(mod, fn) and callable(getattr(mod, fn)):
                    self._scraper = getattr(mod, fn)
                    logger.info("PrizePicksProvider: using scraper function %s.%s", self._scraper_module_name, fn)
                    return
            if hasattr(mod, "scrape") and callable(mod.scrape):
                self._scraper = mod.scrape
                logger.info("PrizePicksProvider: using scraper function %s.scrape", self._scraper_module_name)
                return
            logger.warning("PrizePicksProvider: module '%s' imported but no known scrape function found.", self._scraper_module_name)
        except ImportError:
            logger.info("PrizePicksProvider: no local scraper module '%s' found. Provider will return an instructive error.", self._scraper_module_name)
        except Exception:
            logger.exception("PrizePicksProvider: error importing scraper module '%s'", self._scraper_module_name)

    def get_props(self, sport: SportType = None, date: Optional[str] = None, player: Optional[str] = None, market: Optional[str] = None, **kwargs):
        """
        Primary method to get player props from PrizePicks.
        Delegates to the scraper function when available; otherwise returns an error response explaining how to plug in your scraper.
        """
        start = time.time()
        try:
            if not self._scraper:
                msg = (
                    "No PrizePicks scraper detected. Add a local module named 'prizepicks_scraper' "
                    "with a function `get_prizepicks_props(sport=None, date=None, player=None, market=None, **kwargs)` "
                    "that returns an iterable of prop dicts. Example item keys: player, team, matchup, market, line, timestamp."
                )
                return APIResponse.error_response(msg, response_time=time.time() - start, status_code=500)
            raw_items = self._scraper(sport=sport, date=date, player=player, market=market, **kwargs)
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            normalized = []
            for it in raw_items or []:
                try:
                    norm = _normalize_item(dict(it))
                    normalized.append(norm)
                except Exception:
                    logger.exception("PrizePicksProvider: failed to normalize item: %r", it)
            return APIResponse.success_response(data={"data": normalized}, response_time=time.time() - start, status_code=200)
        except Exception as e:
            logger.exception("PrizePicksProvider.get_props failed")
            return APIResponse.error_response(str(e), response_time=time.time() - start, status_code=500)

    def get_player_props(self, *args, **kwargs):
        return self.get_props(*args, **kwargs)

    def get_odds(self, sport: SportType = None, market: str = None, **kwargs):
        """
        PrizePicks is not a traditional sportsbook; odds may not be applicable.
        This method attempts to return the props with the same schema.
        """
        return self.get_props(sport=sport, market=market, **kwargs)

    def get_competitions(self, sport: SportType = None, **kwargs):
        """
        If your scraper can list active competitions (leagues/tournaments), return them via the scraper.
        Implement a function in prizepicks_scraper like `get_prizepicks_competitions(...)` if needed.
        """
        try:
            if not self._scraper:
                return APIResponse.error_response("No scraper module available", response_time=0, status_code=500)
            mod = __import__(self._scraper_module_name, fromlist=["*"])
            if hasattr(mod, "get_prizepicks_competitions") and callable(mod.get_prizepicks_competitions):
                start = time.time()
                comps = mod.get_prizepicks_competitions(sport=sport, **kwargs)
                return APIResponse.success_response(data={"data": list(comps)}, response_time=time.time() - start, status_code=200)
            return APIResponse.error_response("Competitions function not implemented in scraper module", response_time=0, status_code=404)
        except Exception as e:
            logger.exception("PrizePicksProvider.get_competitions failed")
            return APIResponse.error_response(str(e), response_time=0, status_code=500)
