"""
DFS pipeline helpers:
- Try to use fantasyfootball for data/features if available.
- Fallback to nfl_data_py for recent-season aggregates.
- Write player pool CSV compatible with pydfs_lineup_optimizer_enhanced.
- Run optimizer and return lineups.
"""
from __future__ import annotations

import csv
from typing import List, Optional


def build_player_pool_csv(
    out_csv: str = "player_pool.csv",
    position: str = "RB",
    scoring_source: str = "draftkings",
    seasons: Optional[List[int]] = None,
) -> str:
    """Create optimizer-ready player pool CSV.

    Attempts to use fantasyfootball; if not available, uses nfl_data_py.
    Columns: Name, Position, Team, Opponent, Salary, Projection
    """
    seasons = seasons or [2024, 2025]
    rows: List[List] = []

    # Try fantasyfootball path
    try:
        from fantasyfootball.data import FantasyData
        from fantasyfootball.features import FantasyFeatures

        fantasy_data = FantasyData(season_year_start=min(seasons), season_year_end=max(seasons))
        fantasy_data.create_fantasy_points_column(scoring_source=scoring_source)
        df = fantasy_data.data

        features = FantasyFeatures(df, position=position, y=f"ff_pts_{scoring_source}")
        features.create_future_week()
        feat_df = features.create_ff_signature().get("feature_df")

        # Heuristic salary/projection columns depending on source
        salary_col = "draftkings_salary" if scoring_source == "draftkings" else (
            "fanduel_salary" if scoring_source == "fanduel" else "yahoo_salary"
        )
        proj_col = f"ff_pts_{scoring_source}_pred" if f"ff_pts_{scoring_source}_pred" in feat_df.columns else f"ff_pts_{scoring_source}"

        for _, r in feat_df.iterrows():
            name = r.get("name") or r.get("player") or ""
            team = r.get("team") or r.get("team_abbr") or ""
            opp = r.get("opp") or r.get("opponent") or ""
            salary = r.get(salary_col) or 0
            proj = r.get(proj_col) or 0.0
            rows.append([name, position, team, opp, int(salary) if salary else 0, float(proj)])
    except Exception:
        # Fallback: nfl_data_py basic projections proxy
        import pandas as pd
        import nfl_data_py as nfl

        # Pull schedule and basic seasonal data
        schedule = nfl.import_schedules([max(seasons)])
        try:
            recent = nfl.import_seasonal_data([max(seasons)], downcast=True)
        except TypeError:
            recent = nfl.import_seasonal_data([max(seasons)])

        # Basic filter by position; approximate projection as last season fantasy points per game
        if "fantasy_points_ppr" not in recent.columns:
            recent["fantasy_points_ppr"] = 0.0

        if "position" in recent.columns:
            recent = recent[recent["position"] == position]
        # Normalize games column presence
        if "games" not in recent.columns:
            recent["games"] = 1
        recent_gp = (
            recent.groupby([
                recent.get("player_display_name", pd.Series(recent.get("player_name", "")).fillna("")),
                recent.get("team", pd.Series("")).fillna("")
            ])
            .agg({
                "fantasy_points_ppr": "sum",
                "games": "sum",
            })
            .reset_index()
        )
        recent_gp["ppg"] = recent_gp["fantasy_points_ppr"] / recent_gp["games"].replace({0: 1})

        # Map opponent from schedule roughly by team
        opp_map = {}
        if not schedule.empty:
            for _, s in schedule.iterrows():
                home = s.get("home_team")
                away = s.get("away_team")
                if isinstance(home, str) and isinstance(away, str):
                    opp_map.setdefault(home, away)  # last one wins

        for _, r in recent_gp.sort_values("ppg", ascending=False).head(400).iterrows():
            name = r.get("player_display_name") or r.get("player_name") or ""
            team = r.get("team") or ""
            opp = opp_map.get(team, "")
            salary = 5500  # neutral default
            proj = float(r["ppg"])  # naive proxy
            rows.append([name, position, team, opp, salary, proj])

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Position", "Team", "Opponent", "Salary", "Projection"])
        for row in rows:
            if row and row[0]:
                w.writerow(row)
    return out_csv


def optimize_lineups(
    player_pool_csv: str = "player_pool.csv",
    site: str = "DRAFTKINGS",
    sport: str = "FOOTBALL",
    lineups: int = 5,
):
    from pydfs_lineup_optimizer_enhanced import get_optimizer, Site, Sport

    site_enum = getattr(Site, site.upper())
    sport_enum = getattr(Sport, sport.upper())
    opt = get_optimizer(site_enum, sport_enum)
    opt.load_players_from_csv(player_pool_csv)
    return list(opt.optimize(lineups))


if __name__ == "__main__":
    csv_path = build_player_pool_csv(out_csv="player_pool.csv", position="RB", scoring_source="draftkings")
    lineups = optimize_lineups(csv_path, site="FANDUEL", sport="FOOTBALL", lineups=3)
    for l in lineups:
        print(l)
