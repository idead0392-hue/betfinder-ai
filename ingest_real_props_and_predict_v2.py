#!/usr/bin/env python3
"""
ingest_real_props_and_predict_v2.py

Ingests real scraped player prop/stat data from props_data_fetcher,
passes it to sport agents (BasketballAgent, FootballAgent, EsportsAgent, etc.),
learns from history via learn_from_history(), and generates predictions via make_picks().

Usage:
    python ingest_real_props_and_predict_v2.py

Scheduling (via cron):
    # Run daily at 9 AM
    0 9 * * * cd /workspaces/betfinder-ai && /usr/bin/python3 ingest_real_props_and_predict_v2.py

API Integration:
    # Import and call from your Flask/FastAPI endpoint:
    # from ingest_real_props_and_predict_v2 import run_predictions
    # result = run_predictions()
"""

import json
import sys
from datetime import datetime

# Import sport agents
from sport_agents import (
    BasketballAgent,
    FootballAgent,
    EsportsAgent,
    BaseballAgent,
    HockeyAgent,
    SoccerAgent
)

# Import data fetchers
from props_data_fetcher import PropsDataFetcher


def fetch_real_data():
    """
    Fetch real scraped player prop/stat data from available providers.
    Returns a dictionary with sport categories and their props.
    """
    print("\n" + "="*60)
    print("FETCHING REAL PROPS DATA")
    print("="*60)
    
    all_props = {}
    
    # Fetch from PropsDataFetcher (PrizePicks + Underdog)
    try:
        fetcher = PropsDataFetcher()
        
        # Fetch PrizePicks props
        print("\n[1/2] Fetching PrizePicks props...")
        prizepicks_props = fetcher.fetch_prizepicks_props()
        print(f"   Retrieved {len(prizepicks_props)} PrizePicks props")
        
        # Fetch Underdog props
        print("\n[2/2] Fetching Underdog props...")
        underdog_props = fetcher.fetch_underdog_props()
        print(f"   Retrieved {len(underdog_props)} Underdog props")
        
        # Merge props
        all_props_list = prizepicks_props + underdog_props
        
        # Organize by sport
        for prop in all_props_list:
            sport = prop.get('sport', 'Unknown')
            if sport not in all_props:
                all_props[sport] = []
            all_props[sport].append(prop)
                
    except Exception as e:
        print(f"[ERROR] Failed to fetch from PropsDataFetcher: {e}")
    # SportBexProvider removed: no longer used
    
    # Print summary
    print("\n" + "-"*60)
    print("DATA FETCH SUMMARY")
    print("-"*60)
    for sport, props in all_props.items():
        print(f"  {sport}: {len(props)} props")
    print("-"*60)
    
    return all_props


def run_predictions():
    """
    Main function to ingest real data, learn from history, and generate predictions.
    Returns a dictionary of predictions organized by sport.
    """
    print("\n" + "#"*60)
    print("# BETFINDER AI - REAL DATA INGESTION & PREDICTION")
    print("#"*60)
    print(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60 + "\n")
    
    # Step 1: Fetch real scraped data
    all_props = fetch_real_data()
    
    if not all_props:
        print("\n[WARNING] No props data fetched. Exiting.")
        return {}
    
    # Step 2: Initialize sport agents
    print("\n" + "="*60)
    print("INITIALIZING SPORT AGENTS")
    print("="*60)
    
    agents = {
        'Basketball': BasketballAgent(),
        'Football': FootballAgent(),
        'Esports': EsportsAgent(),
        'Baseball': BaseballAgent(),
        'Hockey': HockeyAgent(),
        'Soccer': SoccerAgent()
    }
    
    print(f"Initialized {len(agents)} sport agents")
    for sport_name in agents.keys():
        print(f"  - {sport_name}Agent")
    
    # Step 3: Process each sport and generate predictions
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60 + "\n")
    
    all_predictions = {}
    
    for sport_name, agent in agents.items():
        print(f"\n{'='*60}")
        print(f"SPORT: {sport_name.upper()}")
        print(f"{'='*60}")
        
        # Get props for this sport (try exact match and common variations)
        sport_props = []
        for key in all_props.keys():
            if sport_name.lower() in key.lower() or key.lower() in sport_name.lower():
                sport_props.extend(all_props[key])
        
        if not sport_props:
            print(f"[INFO] No props found for {sport_name}. Skipping.")
            continue
        
        print(f"\n[1/3] Props available: {len(sport_props)}")
        
        try:
            # Step 3a: Learn from historical data
            print("[2/3] Learning from history...")
            agent.learn_from_history()
            print("      ✓ Learning complete")
            
            # Step 3b: Generate predictions
            print("[3/3] Generating predictions...")
            predictions = agent.make_picks(props_data=sport_props)
            print(f"      ✓ Generated {len(predictions)} predictions")
            
            # Store predictions
            all_predictions[sport_name] = predictions
            
            # Print top 5 predictions
            if predictions:
                print(f"\n      TOP PREDICTIONS for {sport_name}:")
                for i, pick in enumerate(predictions[:5], 1):
                    player = pick.get('player_name', 'Unknown')
                    prop_type = pick.get('prop_type', 'Unknown')
                    line = pick.get('line', 'N/A')
                    pick_side = pick.get('pick', 'N/A')
                    confidence = pick.get('confidence', 0)
                    print(f"        {i}. {player} - {prop_type} {pick_side} {line} (Confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {sport_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Save predictions to file
    print("\n" + "="*60)
    print("SAVING PREDICTIONS")
    print("="*60)
    
    output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(all_predictions, f, indent=2, default=str)
        print(f"✓ Predictions saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save predictions: {e}")
    
    # Step 5: Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    total_predictions = sum(len(preds) for preds in all_predictions.values())
    print(f"Total predictions: {total_predictions}")
    for sport, preds in all_predictions.items():
        print(f"  {sport}: {len(preds)} predictions")
    print("="*60 + "\n")
    
    return all_predictions


if __name__ == "__main__":
    """
    Main entry point for standalone execution.
    """
    try:
        predictions = run_predictions()
        
        if predictions:
            print("\n✓ SUCCESS: Predictions generated and saved.")
            sys.exit(0)
        else:
            print("\n✗ WARNING: No predictions generated.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Process interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)