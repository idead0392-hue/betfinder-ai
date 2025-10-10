"""
Ingest real props and stats, feed to agents, retrain ML, and print predictions.

Usage: python3 ingest_real_props_and_predict.py

This script:
- Loads real props from PropsDataFetcher (SportBex-backed) and optional PrizePicks/Underdog fetchers if present
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

import sys
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Local imports
from sport_agents import create_sport_agent, ml_model

# Best-effort imports for providers/fetchers
props_fetcher = None
pp_fetch = None
ud_fetch = None
sportbex_provider = None

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

try:
    import sportbex_provider as sportbex_provider  # noqa: F401
    sportbex_provider_available = True
except Exception:
    sportbex_provider_available = False


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def group_by_sport(props: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for p in props:
        sport = (p.get('sport') or '').lower().strip()
        if sport in SPORT_KEYS:
            grouped[sport].append(p)
    return grouped


def load_all_real_props(max_props: int = 250) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """Load props from available sources and return grouped by sport and counts per source."""
    all_props: List[Dict] = []
    counts: Dict[str, int] = {}

    # Source 1: SportBex-backed PropsDataFetcher (normalized)
    if props_fetcher:
        try:
            props = props_fetcher.fetch_all_props(max_props=max_props)
            counts['sportbex_props'] = len(props)
            all_props.extend(props)
        except Exception as e:
            print(f"⚠️ Error fetching from PropsDataFetcher: {e}")
            counts['sportbex_props'] = 0
    else:
        counts['sportbex_props'] = 0

    # Source 2: PrizePicks (if available)
    if pp_fetch:
        try:
            # Pull for primary sports only to avoid volume blow-up
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
        'top_picks': [],
        'ml_samples': []
    }

    try:
        agent = create_sport_agent(SPORT_KEYS[sport_key])
        report['agent'] = agent.agent_type

        # Refresh learning and training before predictions
        agent.learn_from_history()
        agent.train_ml_model()
        report['training_status'] = f"trained with historical picks; model v{ml_model.model_version}"

        # If no real props for this sport, fall back to agent's own fetch (which may include mocks/providers)
        used_fallback = False
        run_props = props
        if not run_props:
            try:
                run_props = agent.fetch_props(max_props=60)
                used_fallback = True
            except Exception:
                run_props = []

        # Make picks using provided real props or fallback
        picks = agent.make_picks(props_data=run_props)
        report['picks_count'] = len(picks)
        if used_fallback:
            report['note'] = 'used_fallback_props_due_to_empty_real_feed'
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

        # Create a few direct ML predictions from sample props
        for sample in run_props[:3]:
            ml_pred = agent.get_ml_prediction(sample)
            report['ml_samples'].append({
                'player': sample.get('player_name') or sample.get('pick'),
                'stat': sample.get('stat_type'),
                'line': sample.get('line'),
                'odds': sample.get('odds'),
                'ml': ml_pred
            })

        return report
    except Exception as e:
        report['training_status'] = f"error: {e}"
        return report


def main():
    print("=== Ingest Real Props & Predict ===")
    print(f"Run at: {datetime.now().isoformat()}")

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

        # Print ML sample predictions
        print("ML samples:")
        for s in report['ml_samples']:
            ml = s['ml']
            print(f"  {s['player']} [{s['stat']}] line={s['line']} odds={s['odds']} -> conf={ml.get('confidence')}% edge={ml.get('edge')} roi={ml.get('expected_roi')}")

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
