"""NFL game-log loader via nflverse (nfl_data_py)."""

import pandas as pd


def load_nfl_qb_yards(seasons):
    """Return one row per QB per game: player, season, week, team, opponent, yards.

    seasons: e.g. [2024, 2025, 2026]
    """
    import nfl_data_py as nfl

    weekly = nfl.import_weekly_data(seasons)
    qb = weekly[weekly["position"] == "QB"].copy()
    qb = qb[qb["attempts"] > 0]  # drop gadget-play rows
    out = qb[["player_display_name", "season", "week", "recent_team",
              "opponent_team", "passing_yards"]].rename(columns={
        "player_display_name": "player",
        "recent_team": "team",
        "opponent_team": "opponent",
        "passing_yards": "yards",
    })
    return out.sort_values(["player", "season", "week"]).reset_index(drop=True)


def load_nfl_defense_rank(seasons):
    """Opponent pass yards allowed per game, per team-season. For the v1
    opponent-adjustment feature."""
    import nfl_data_py as nfl

    weekly = nfl.import_weekly_data(seasons)
    team_wk = (weekly[weekly["position"] == "QB"]
               .groupby(["season", "week", "opponent_team"])["passing_yards"]
               .sum().reset_index()
               .rename(columns={"opponent_team": "team", "passing_yards": "pass_yds_allowed"}))
    rank = (team_wk.groupby(["season", "team"])["pass_yds_allowed"]
            .mean().reset_index()
            .rename(columns={"pass_yds_allowed": "opp_pass_yds_allowed_per_game"}))
    return rank.sort_values(["season", "opp_pass_yds_allowed_per_game"]).reset_index(drop=True)
