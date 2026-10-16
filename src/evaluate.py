"""Honest evaluation: walk-forward backtest + calibration."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .baseline import NormalCDFBaseline


def walk_forward(df, line_col="line", window=5, min_history=3):
    """Time-ordered backtest. For each player-game with enough history, predict
    P(yards > line) using only games before it. Returns DataFrame with p_hit, hit."""
    model = NormalCDFBaseline(window=window)
    rows = []
    for player, g in df.sort_values(["season", "week"]).groupby("player"):
        yards_hist = []
        for _, r in g.iterrows():
            if len(yards_hist) >= min_history:
                p = model.predict(np.array(yards_hist), r[line_col])["p_hit"]
                rows.append({"player": player, "season": r["season"], "week": r["week"],
                             "line": r[line_col], "yards": r["yards"],
                             "p_hit": p, "hit": int(r["yards"] > r[line_col])})
            yards_hist.append(r["yards"])
    return pd.DataFrame(rows)


def calibration_table(results, n_bins=5):
    """Are 70% predictions right 70% of the time? Returns (bin_center, frac_positive)."""
    prob_true, prob_pred = calibration_curve(results["hit"], results["p_hit"],
                                             n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted": prob_pred, "actual": prob_true})


def paper_roi(results, stake=10.0, decimal_payout=1.91, edge_threshold=0.0):
    """Hypothetical P&L flat-betting only picks where model edge > threshold."""
    model = NormalCDFBaseline()
    bets = results[results["p_hit"] - 1.0 / decimal_payout > edge_threshold]
    wins = bets["hit"].sum()
    profit = wins * stake * (decimal_payout - 1) - (len(bets) - wins) * stake
    return {"n_bets": int(len(bets)), "wins": int(wins),
            "profit": float(profit), "roi": float(profit / (len(bets) * stake)) if len(bets) else 0.0}
