# BetFinder AI - Player Prop Predictor Prototype

This update introduces a first-pass Streamlit UI module for player prop predictions, modeled after PlayerProps.ai, and integrates it with the existing provider abstraction where possible.

## New Module: player_prop_predictor.py

Purpose: Provide an interactive UI to explore player prop markets with filters and a computed AI/stat projection column.

Key features:
- Filters: sport, date, sportsbook, stat type (points/rebounds/assists/threes/yards/goals), team/player search
- Table columns: matchup, player, market, prop line, sportsbook (book), over/under odds, implied probability, AI/stat projection (stub engine), edge (proj-line), recent trends
- Odds helpers: American odds to implied probability, simple edge heuristic
- Loading and empty states: spinner on fetch, friendly messages if no data
- Provider integration: attempts to use SportbexProvider if available, with graceful fallback to demo data
- Projection engine: deterministic stub, designed to be replaced by a real model later

How to run:
- Local: `streamlit run player_prop_predictor.py`
- Codespaces: forward Streamlit port and open external preview

## Provider Integration

- The UI attempts to instantiate `SportbexProvider()` and call one of the following methods if present:
  - `get_player_props(sport, date, sportsbook, market, query)`
  - `fetch_props(sport, date, sportsbook, market, query)`
- Returned items are normalized to a common schema for display (see example structure in the module docstring).
- If the provider is missing or fails, the UI shows a demo list filtered by the selected stat type to keep the UX functional.

NOTE: The current `sportbex_provider.py` imports `BaseAPIProvider, APIResponse, RequestConfig` from `api_providers.py`, but those symbols are not exported yet. The predictor UI does not require those interfaces directly and will function in demo mode until the provider import issue is resolved.

## Projection Engine (stub)

Function: `projection_engine(player, stat_type, sport, team=None) -> Optional[float]`
- Deterministic mock returning a reasonable value for different stat ranges
- Replace with an ML/statistical model when ready
- Edge calculation: `edge = (projection - line) * f(probability)` where probability uses implied odds if available

## UX/Behavior

- Sidebar filters with “Load Props” button to control fetch timing
- Spinner while fetching
- If no props: shows a warning and tips to adjust filters
- Sorting, ascending toggle, Top N limiter on the table
- Caption clarifies the AI projection is stubbed

## File Changes

- Added: `player_prop_predictor.py` (new Streamlit UI prototype)
- No changes required to existing provider files for this prototype to render demo data

## Next Steps / TODOs

1) Provider Abstraction Fix
- Export `BaseAPIProvider`, `APIResponse`, `RequestConfig` from `api_providers.py` or adjust `sportbex_provider.py` imports
- Provide a stable provider method for props: `get_player_props(...)` (preferred) and shape output to the documented schema

2) Projection Integration
- Replace stub with a real projection pipeline (stat models, features, and weights per sport/market)
- Add confidence intervals and reason codes

3) Table Enhancements
- Add color formatting for edges and probabilities
- Add over/under recommendation column based on projection and odds
- Add pagination/virtualization for large result sets

4) Caching and Performance
- Cache provider responses (TTL) and memoize projections per query window
- Add a refresh and last-updated indicator

5) Testing
- Unit tests for odds conversion, edge calculation, and normalization logic
- Integration test that mocks provider responses and validates table output

6) Documentation
- Update SPORTBEX_PROVIDER_GUIDE.md with the prop schema and example outputs
- Expand README with app usage instructions and screenshots

## Example Provider Item Schema

```
{
  "sport": "NBA",
  "date": "2025-10-08",
  "matchup": "LAL @ BOS",
  "player": "LeBron James",
  "team": "LAL",
  "opponent": "BOS",
  "market": "points",
  "line": 27.5,
  "sportsbook": "DK",
  "over_odds": -115,
  "under_odds": -105,
  "recent_trends": {"last5_avg": 28.4, "last10_avg": 27.1}
}
```

The UI normalizes provider responses to this structure when possible.
