# src/core/bankroll_manager.py
import json
import os
from typing import Dict

class BankrollManager:
    def __init__(self, file_path: str = "data/bankroll_data.json"):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self) -> Dict:
        """Load bankroll data from a file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {'total_bankroll': 1000, 'unit_size': 10}

    def save_data(self):
        """Save bankroll data to a file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_bankroll(self) -> float:
        """Get the current total bankroll."""
        return self.data.get('total_bankroll', 0)

    def get_unit_size(self) -> float:
        """Get the current unit size."""
        return self.data.get('unit_size', 0)

    def calculate_bet_size(self, odds: int, probability: float) -> float:
        """Calculate the optimal bet size using the Kelly Criterion."""
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        edge = (probability * decimal_odds) - 1
        if edge <= 0:
            return 0
        
        fraction = edge / (decimal_odds - 1)
        return self.get_bankroll() * fraction
