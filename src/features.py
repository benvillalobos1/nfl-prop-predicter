"""Rolling-window feature engineering over per-game yardage."""

import pandas as pd


def add_rolling_features(df, windows=(3, 5), min_games=3):
    """Add rolling mean/std of `yards` per player. Rows with too little history are dropped.

    Expects columns: player, season, week, yards (+ anything else, preserved).
    """
    df = df.sort_values(["player", "season", "week"]).copy()
    for w in windows:
        grp = df.groupby("player")["yards"]
        df[f"roll_mean_{w}"] = grp.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=2).std(ddof=1)
        )
    df["games_seen"] = df.groupby("player").cumcount()
    return df[df["games_seen"] >= min_games].reset_index(drop=True)
