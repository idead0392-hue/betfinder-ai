"""
Mapping of players miscategorized as College Football to NFL with teams.

Update this file as new draft classes arrive or when PrizePicks lags updates.
"""

from typing import Optional, Tuple


# Player name -> (League, Team)
CFB_TO_NFL: dict[str, Tuple[str, str]] = {
    "Drake Maye": ("NFL", "Carolina Panthers"),
    "Bo Nix": ("NFL", "Denver Broncos"),
    "Jayden Daniels": ("NFL", "Washington Commanders"),
    "Michael Penix Jr.": ("NFL", "Atlanta Falcons"),
    "J.J. McCarthy": ("NFL", "Minnesota Vikings"),
    "Caleb Williams": ("NFL", "Chicago Bears"),
    # Add more mappings as necessary
}


def get_nfl_info(player_name: str) -> Optional[Tuple[str, str]]:
    """Return (league, team) if the player has transitioned to NFL, else None."""
    return CFB_TO_NFL.get(player_name)


def is_nfl_player(player_name: str) -> bool:
    """Returns True if player is now in NFL."""
    return player_name in CFB_TO_NFL
