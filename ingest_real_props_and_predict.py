"""
Ingest real props and stats, feed to agents, retrain ML, and print predictions.

Usage: python3 ingest_real_props_and_predict.py

This script:
- Loads real props from PropsDataFetcher (PrizePicks CSV data) and optional PrizePicks/Underdog fetchers if present (no SportBex)
- Groups props by sport
- Feeds them to all sport agents (NBA, NFL, MLB, NHL, Soccer, CFB, plus CS2 where applicable)
- Triggers learn_from_history and make_picks on each agent with the real data
- Prints comprehensive output:
  1) Counts loaded per source and total per sport
  2) Each agent's top predictions with confidence
  3) ML model predictions overview
  4) Learning/training status per agent
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime

import pandas as pd

# Local imports
from sport_agents import create_sport_agent, ml_model

# Best-effort imports for providers/fetchers
props_fetcher = None
pp_fetch = None
ud_fetch = None
csv_data = None

# Map friendly sport names to agent creation keys
SPORT_KEYS = {
    'basketball': 'basketball',  # NBA
    'football': 'football',      # NFL
    'baseball': 'baseball',      # MLB
    'hockey': 'hockey',          # NHL
    'soccer': 'soccer',
    'college_football': 'college_football',
    # Esports
    'csgo': 'csgo'
}

# Helper: safe import wrappers
try:
    from props_data_fetcher import PropsDataFetcher
    props_fetcher = PropsDataFetcher()
except Exception as e:
    print(f"⚠️ Could not initialize PropsDataFetcher: {e}")

try:
    from props_data_fetcher import fetch_prizepicks_props as pp_fetch
except Exception:
    pp_fetch = None

try:
    from props_data_fetcher import fetch_underdog_props as ud_fetch
except Exception:
    ud_fetch = None

# csv_data is already initialized as None above - no import needed


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def group_by_sport(props: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for p in props:
        sport = (p.get('sport') or '').lower().strip()
        # Try to infer sport from prop/stat type if missing
        if not sport:
            stat = str(p.get('stat_type', '')).lower()
            if 'point' in stat or 'rebound' in stat or 'assist' in stat or 'block' in stat or 'steal' in stat or 'three' in stat:
                sport = 'basketball'
            elif 'yard' in stat or 'touchdown' in stat or 'reception' in stat or 'passing' in stat or 'rushing' in stat:
                sport = 'football'
            elif 'hit' in stat or 'run' in stat or 'home_run' in stat or 'strikeout' in stat:
                sport = 'baseball'
            elif 'goal' in stat or 'shot' in stat or 'penalty' in stat:
                sport = 'hockey'
            elif 'card' in stat or 'goal' in stat or 'assist' in stat:
                sport = 'soccer'
            elif 'kill' in stat or 'headshot' in stat or 'fantasy' in stat:
                sport = 'csgo'
        if sport in SPORT_KEYS:
            grouped[sport].append(p)
    return grouped
def load_pickfinder_csv(csv_path: str) -> List[Dict]:
    """Load and normalize pickfinder_all_projections.csv to agent prop dicts."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Could not load {csv_path}: {e}")
        return []

    props = []
    for _, row in df.iterrows():
        # Normalize columns to agent prop format
        prop = {
            'player_name': row.get('Player', ''),
            'team': row.get('Team/Opp', ''),
            'pick': row.get('Prop', ''),
            'stat_type': row.get('Prop', '').lower(),
            'line': float(row.get('Line', 0)) if row.get('Line', '') else 0.0,
            'odds': int(row.get('Odds', -110)) if str(row.get('Odds', '')).replace('+','').replace('-','').isdigit() else -110,
            'confidence': float(row.get('IP', 50)) if row.get('IP', '') else 50.0,
            'expected_value': float(row.get('Diff', 0)) if row.get('Diff', '') else 0.0,
            'avg_l10': float(row.get('Avg_L10', 0)) if row.get('Avg_L10', '') else 0.0,
            'start_time': '',  # Not available
            'sport': '',  # Will be inferred
            'over_under': 'over' if 'over' in str(row.get('Prop', '')).lower() else 'under' if 'under' in str(row.get('Prop', '')).lower() else None
        }
        props.append(prop)
    print(f"Loaded {len(props)} props from {csv_path}")
    return props


def load_all_real_props(max_props: int = 250) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """Load props from available sources and return grouped by sport and counts per source."""
    all_props: List[Dict] = []
    counts: Dict[str, int] = {}

    # Source 1: PrizePicks CSV data PropsDataFetcher (normalized)
    if props_fetcher:
        try:
            props = props_fetcher.fetch_all_props(max_props=max_props)
            counts['csv_props'] = len(props)
            all_props.extend(props)
        except Exception as e:
            print(f"⚠️ Error fetching from PropsDataFetcher: {e}")
            counts['csv_props'] = 0
    else:
        counts['csv_props'] = 0

    # Source 2: PrizePicks (if available)
    if pp_fetch:
        try:
            # Loads real props from PrizePicks CSV data and optional provider fetchers if present
            sports_to_pull = ['basketball', 'football', 'baseball', 'hockey', 'soccer']
            pp_total = 0
            for s in sports_to_pull:
                try:
                    data = pp_fetch(s)
                    if isinstance(data, list):
                        all_props.extend(data)
                        pp_total += len(data)
                except Exception:
                    continue
            counts['prizepicks'] = pp_total
        except Exception as e:
            print(f"⚠️ Error fetching PrizePicks: {e}")
            counts['prizepicks'] = 0
    else:
        counts['prizepicks'] = 0

    # Source 3: Underdog (if available)
    if ud_fetch:
        try:
            sports_to_pull = ['basketball', 'football', 'baseball', 'hockey', 'soccer']
            ud_total = 0
            for s in sports_to_pull:
                try:
                    data = ud_fetch(s)
                    if isinstance(data, list):
                        all_props.extend(data)
                        ud_total += len(data)
                except Exception:
                    continue
            counts['underdog'] = ud_total
        except Exception as e:
            print(f"⚠️ Error fetching Underdog: {e}")
            counts['underdog'] = 0
    else:
        counts['underdog'] = 0

    grouped = group_by_sport(all_props)
    return grouped, counts


def run_agent_pipeline(sport_key: str, props: List[Dict]) -> Dict[str, Any]:
    """Create agent, retrain from history, run picks on provided props, return report."""
    report: Dict[str, Any] = {
        'sport': sport_key,
        'input_props': len(props),
        'agent': None,
        'training_status': None,
        'picks_count': 0,
        'top_picks': []
    }

    try:
        agent = create_sport_agent(SPORT_KEYS[sport_key])
        report['agent'] = agent.agent_type

        # Refresh learning and training before predictions
        agent.learn_from_history()
        agent.train_ml_model()
        report['training_status'] = f"trained with historical picks; model v{ml_model.model_version}"

        # If no real props for this sport, fall back to agent's own fetch (which may include mocks/providers)
        run_props = props  # Only use real PrizePicks props, no fallback/sample

        # Make picks using provided real props or fallback
        picks = agent.make_picks(props_data=run_props)
        report['picks_count'] = len(picks)
        # Summarize top 5 picks
        report['top_picks'] = [
            {
                'pick': p.get('pick'),
                'player': p.get('player_name'),
                'over_under': p.get('over_under'),
                'line': p.get('line'),
                'odds': p.get('odds'),
                'confidence': p.get('confidence'),
                'expected_value': p.get('expected_value'),
                'ml_edge': p.get('ml_edge'),
                'ml_expected_roi': p.get('ml_expected_roi'),
                'ml_confidence': p.get('ml_prediction', {}).get('confidence')
            }
            for p in (picks[:5] if picks else [])
        ]
        return report
    except Exception as e:
        report['training_status'] = f"error: {e}"
        return report


def main():
    print("=== Ingest Real Props & Predict ===")
    print(f"Run at: {datetime.now().isoformat()}")

    # Try to load pickfinder_all_projections.csv first
    csv_path = 'pickfinder_all_projections.csv'
    csv_props = load_pickfinder_csv(csv_path)
    if csv_props:
        grouped = group_by_sport(csv_props)
        counts = {'pickfinder_csv': len(csv_props)}
        print(f"Loaded {len(csv_props)} props from PickFinder CSV.")
    else:
        grouped, counts = load_all_real_props(max_props=300)

    # Print source counts
    print("\n-- Data source counts --")
    for k, v in counts.items():
        print(f"{k}: {v}")

    # Ensure we cover required sports even if empty
    for sport in ['basketball', 'football', 'baseball', 'hockey', 'soccer', 'college_football', 'csgo']:
        grouped.setdefault(sport, [])

    # Run pipelines per sport
    reports: Dict[str, Dict[str, Any]] = {}
    for sport, props in grouped.items():
        if sport not in SPORT_KEYS:
            continue
        print(f"\n== {sport.upper()} ==")
        print(f"Props available: {len(props)}")
        report = run_agent_pipeline(sport, props)
        reports[sport] = report

        # Pretty-print top picks
        print("Top predictions:")
        if report['top_picks']:
            for i, p in enumerate(report['top_picks'], 1):
                print(f"  {i}. {p['pick']} | line={p['line']} | odds={p['odds']} | conf={p['confidence']}% | ml_conf={p['ml_confidence']}% | edge={p['ml_edge']} | roi={p['ml_expected_roi']}")
        else:
            print("  (no picks)")

        # Disabled: no sample ML predictions

        print(f"Training/learning: {report['training_status']}")
        print(f"Total picks generated: {report['picks_count']}")

    # Summary for NBA/NFL/CS2 specifically
    print("\n=== Validation Summary (NBA/NFL/CS2) ===")
    for key, label in [('basketball', 'NBA'), ('football', 'NFL'), ('csgo', 'CS2')]:
        rep = reports.get(key, {})
        print(f"{label}: {rep.get('picks_count', 0)} picks, {len(grouped.get(key, []))} props ingested")
        if rep.get('top_picks'):
            best = rep['top_picks'][0]
            print(f"  Best: {best['pick']} (conf={best['confidence']}%, edge={best['ml_edge']}, roi={best['ml_expected_roi']})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
