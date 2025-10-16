

# src/core/picks_ledger.py
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class PicksLedger:
    def __init__(self, file_path: str = "data/picks_ledger.json"):
        self.file_path = file_path
        self.picks = self.load_picks()

    def load_picks(self) -> List[Dict]:
        """Load picks from the ledger file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return []

    def save_picks(self):
        """Save picks to the ledger file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.picks, f, indent=2)

    def add_pick(self, pick_data: Dict):
        """Add a new pick to the ledger."""
        pick_data['timestamp'] = datetime.now().isoformat()
        self.picks.append(pick_data)
        self.save_picks()

    def get_all_picks(self) -> List[Dict]:
        """Return all picks from the ledger."""
        return self.picks

    def get_summary(self) -> Dict:
        """Generate a summary of all picks."""
        total_picks = len(self.picks)
        if total_picks == 0:
            return {
                'total_picks': 0,
                'win_rate': 0,
                'avg_confidence': 0
            }
        
        wins = sum(1 for p in self.picks if p.get('outcome') == 'won')
        losses = sum(1 for p in self.picks if p.get('outcome') == 'lost')
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        avg_confidence = statistics.mean(p.get('confidence', 50) for p in self.picks)
        
        return {
            'total_picks': total_picks,
            'win_rate': round(win_rate * 100, 2),
            'avg_confidence': round(avg_confidence, 2)
        }

# Singleton instance of the PicksLedger
picks_ledger = PicksLedger()
