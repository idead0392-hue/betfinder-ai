# src/api/providers.py
from typing import NamedTuple, List, Dict, Optional
from enum import Enum
import random

class SportType(Enum):
    BASKETBALL = "basketball"
    FOOTBALL = "football"
    BASEBALL = "baseball"
    HOCKEY = "hockey"
    SOCCER = "soccer"
    TENNIS = "tennis"
    # Added eSports
    CS2 = "cs2"
    LEAGUE_OF_LEGENDS = "league_of_legends"
    DOTA2 = "dota2"
    VALORANT = "valorant"

class HealthCheckResponse(NamedTuple):
    success: bool
    response_time_ms: float
    error_message: Optional[str] = None

class PropsResponse(NamedTuple):
    success: bool
    data: List[Dict]
    error_message: Optional[str] = None

class MockSportsDataProvider:
    """A mock provider for sports data."""

    def health_check(self) -> HealthCheckResponse:
        """Simulates a health check to a generic provider API."""
        return HealthCheckResponse(success=True, response_time_ms=random.uniform(50, 150))

    def get_props(self, sport: SportType) -> PropsResponse:
        """Simulates fetching player props for a given sport."""
        mock_data = self._generate_mock_props(sport.value)
        return PropsResponse(success=True, data=mock_data)

    def capabilities(self) -> Dict:
        """Returns the capabilities of the provider."""
        return {
            "sports": [s.value for s in SportType],
            "markets": ["player_props", "moneylines", "spreads"]
        }

    def _generate_mock_props(self, sport: str) -> List[Dict]:
        """Generates mock prop data for a given sport."""
        return [
            {
                "player_name": f"{sport.capitalize()} Player {i}",
                "stat_type": "Points",
                "line_score": random.uniform(10.5, 30.5)
            } for i in range(1, 11)
        ]

def create_data_provider() -> MockSportsDataProvider:
    """Factory function to create a MockSportsDataProvider instance."""
    return MockSportsDataProvider()
