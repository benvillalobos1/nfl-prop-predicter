"""Reproduce the Josh Hoover over 199.5 pass yards calculation (Oct 15, 2026).

Run: python examples/hoover_example.py
"""

import sys
sys.path.insert(0, "src")

from baseline import NormalCDFBaseline

# Hoover's 2026 game-by-game passing yards (Indiana): UNT, Howard, WKU, Northwestern, Rutgers, Nebraska
hoover_yards = [231, 145, 273, 201, 251, 263]
# Winston's full starts (Giants): Titans, Cardinals, Commanders
winston_yards = [118, 250, 326]

model = NormalCDFBaseline(window=5)

hoover = model.predict(hoover_yards, line=199.5)
print(f"Hoover over 199.5: mu={hoover['mu']:.1f} sigma={hoover['sigma']:.1f} "
      f"-> P(hit) = {hoover['p_hit']:.1%}")

winston = model.predict(winston_yards, line=0.5)
print(f"Winston over 0.5:  mu={winston['mu']:.1f} sigma={winston['sigma']:.1f} "
      f"-> P(hit) = {winston['p_hit']:.1%}")

p_parlay = hoover["p_hit"] * winston["p_hit"]
print(f"2-pick parlay P(both) = {p_parlay:.1%}  (fair value ~{1/p_parlay:.2f}x, ticket pays 2.60x)")
