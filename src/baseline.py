"""v0 baseline: rolling mean/std -> normal CDF -> P(yards > line).

The benchmark every fancier model has to beat.
"""

import numpy as np
from scipy.stats import norm


class NormalCDFBaseline:
    def __init__(self, window=5):
        self.window = window

    def predict(self, yards, line):
        """yards: array-like of past game yardages (most recent last).
        line: the prop line. Returns dict with mu, sigma, z, p_hit."""
        hist = np.asarray(yards, dtype=float)[-self.window:]
        mu = hist.mean()
        sigma = hist.std(ddof=1) if len(hist) > 1 else 0.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1e-6
        z = (line - mu) / sigma
        p_hit = float(norm.sf(z))  # survival function = P(X > line)
        return {"mu": float(mu), "sigma": float(sigma), "z": float(z), "p_hit": p_hit}

    def edge_vs_payout(self, yards, line, decimal_payout):
        """Compare model probability to the payout-implied break-even probability."""
        p_hit = self.predict(yards, line)["p_hit"]
        p_needed = 1.0 / decimal_payout
        return {"p_hit": p_hit, "p_needed": p_needed, "edge": p_hit - p_needed}
