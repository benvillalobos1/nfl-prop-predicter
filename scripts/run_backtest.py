"""Walk-forward backtest of the NormalCDFBaseline on real nflverse QB data.

Since real sportsbook lines aren't available historically, we test the model's
distributional assumption directly: for each QB-game with >=5 prior games, take
mu/sigma from the trailing 5-game window and price shaded lines at
line = mu + k*sigma for k in {-1, -0.5, 0, 0.5, 1}. A well-calibrated normal
model should satisfy: mean predicted P(hit) ~= actual hit rate, in every bin.

Run: /home/hatch/.venvs/bt-venv/bin/python scripts/run_backtest.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline import NormalCDFBaseline  # noqa: E402
from data import load_nfl_qb_yards  # noqa: E402

SEASONS = [2022, 2023, 2024]  # 2025 not yet published by nflverse; re-add once available
WINDOW = 5
K_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]


def main():
    print("Loading nflverse QB game logs...", flush=True)
    df = load_nfl_qb_yards(SEASONS)
    print(f"  {len(df)} QB-games, {df['player'].nunique()} QBs, "
          f"{df['season'].min()}-{df['season'].max()}", flush=True)

    model = NormalCDFBaseline(window=WINDOW)
    rows = []
    for player, g in df.sort_values(["season", "week"]).groupby("player"):
        hist = []
        for _, r in g.iterrows():
            if len(hist) >= WINDOW:
                h = np.asarray(hist[-WINDOW:], dtype=float)
                mu, sigma = h.mean(), h.std(ddof=1)
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = 1e-6
                for k in K_VALUES:
                    line = mu + k * sigma
                    p = model.predict(h, line)["p_hit"]
                    rows.append({
                        "player": player, "season": int(r["season"]),
                        "week": int(r["week"]), "k": k, "line": line,
                        "mu": mu, "sigma": sigma, "yards": float(r["yards"]),
                        "p_hit": p, "hit": int(r["yards"] > line),
                    })
            hist.append(float(r["yards"]))

    res = pd.DataFrame(rows)
    print(f"  {len(res)} predictions", flush=True)

    out = {"n_predictions": int(len(res)),
           "seasons": SEASONS, "window": WINDOW,
           "n_qb_games": int(len(df)), "n_qbs": int(df["player"].nunique())}

    # Overall calibration
    out["mean_p_hit"] = float(res["p_hit"].mean())
    out["actual_hit_rate"] = float(res["hit"].mean())
    out["brier"] = float(((res["p_hit"] - res["hit"]) ** 2).mean())

    # Calibration by predicted-probability decile
    res["bin"] = pd.qcut(res["p_hit"], 10, labels=False, duplicates="drop")
    cal = (res.groupby("bin", observed=True)
              .agg(n=("hit", "size"),
                   mean_pred=("p_hit", "mean"),
                   actual=("hit", "mean"))
              .reset_index(drop=True))
    out["calibration_deciles"] = cal.round(4).to_dict(orient="records")

    # Per-shade calibration: does the normal model price shaded lines right?
    per_k = (res.groupby("k", observed=True)
                .agg(n=("hit", "size"),
                     mean_pred=("p_hit", "mean"),
                     actual=("hit", "mean"))
                .reset_index())
    out["per_shade"] = per_k.round(4).to_dict(orient="records")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "backtest_summary.json").write_text(json.dumps(out, indent=2))
    res.to_csv(results_dir / "backtest_predictions.csv", index=False)

    print("\n==== RESULTS ====")
    print(f"predictions: {out['n_predictions']}")
    print(f"mean predicted P(hit): {out['mean_p_hit']:.3f}")
    print(f"actual hit rate:       {out['actual_hit_rate']:.3f}")
    print(f"Brier score:           {out['brier']:.4f}")
    print("\nPer-shade calibration (line = mu + k*sigma):")
    print(per_k.round(4).to_string(index=False))
    print("\nSaved to results/backtest_summary.json + backtest_predictions.csv")


if __name__ == "__main__":
    main()
