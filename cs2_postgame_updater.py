"""
Post-game updater for CS:GO (CS2).
- Uses cs2api finished() results to reconcile pick outcomes in PicksLedger.
- Best-effort matching by player name and matchup text.
- Designed to be run periodically (cron) or on-demand.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List

from picks_ledger import picks_ledger

try:
    from cs2api import CS2
except Exception:  # If cs2api missing, this module can still import
    CS2 = None  # type: ignore


async def _fetch_finished() -> List[Dict[str, Any]]:
    if CS2 is None:
        return []
    timeout = int(os.getenv("CS2API_TIMEOUT", "30"))
    async with CS2(timeout=timeout) as cs2:
        try:
            return await cs2.finished()
        except Exception:
            return []


def _name_in_text(name: str, text: str) -> bool:
    n = re.sub(r"\s+", " ", name).strip().lower()
    t = re.sub(r"\s+", " ", text).strip().lower()
    return n in t


def _extract_result_for_pick(pick: Dict[str, Any], finished_matches: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    player = pick.get('player_name') or ''
    matchup = pick.get('matchup') or ''
    stat = (pick.get('stat_type') or '').lower()
    line = float(pick.get('line') or 0)

    for m in finished_matches:
        txt = str(m)
        if (player and _name_in_text(player, txt)) or (matchup and _name_in_text(matchup, txt)):
            # naive parse: look for numbers near keywords
            # this is a placeholder; exact fields depend on cs2api shape
            actual_val = None
            try:
                if 'kill' in stat:
                    # find pattern like 'kills: 23'
                    import re as _re
                    mm = _re.search(r"kills\D+(\d+)", txt, _re.IGNORECASE)
                    if mm:
                        actual_val = float(mm.group(1))
                elif 'headshot' in stat:
                    mm = _re.search(r"headshot\D+(\d+)", txt, _re.IGNORECASE)
                    if mm:
                        actual_val = float(mm.group(1))
            except Exception:
                pass
            if actual_val is None:
                continue
            over_under = pick.get('over_under', 'over')
            outcome = 'won' if ((over_under == 'over' and actual_val > line) or (over_under == 'under' and actual_val < line)) else 'lost'
            return {'actual': actual_val, 'outcome': outcome}
    return None


def update_csgo_picks_from_finished() -> int:
    """Reconcile CS:GO picks with finished match results. Returns count of updates."""
    try:
        finished = asyncio.run(_fetch_finished())
    except RuntimeError:
        import asyncio as _a
        loop = _a.new_event_loop()
        try:
            _a.set_event_loop(loop)
            finished = loop.run_until_complete(_fetch_finished())
        finally:
            loop.close()

    csgo_picks = [p for p in picks_ledger.picks if p.get('sport') == 'csgo' and p.get('outcome') == 'pending']
    updated = 0
    for p in csgo_picks:
        res = _extract_result_for_pick(p, finished)
        if not res:
            continue
        picks_ledger.update_pick_outcome(p['pick_id'], res['outcome'], actual_result=res['actual'], profit_loss=None)
        updated += 1
    return updated


if __name__ == '__main__':
    cnt = update_csgo_picks_from_finished()
    print(f"Updated {cnt} csgo picks")
