"""Prop Edge Finder dashboard: pick a QB, set a line, see P(hit) and the edge."""

import sys
sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm

from baseline import NormalCDFBaseline

st.set_page_config(page_title="Prop Edge Finder", layout="centered")
st.title("Prop Edge Finder — v0 baseline")

st.markdown("Normal-CDF model: rolling mean/std of recent games → P(yards > line).")

with st.sidebar:
    st.header("Inputs")
    raw = st.text_area("Past game yards (comma-separated, oldest first)",
                       "231, 145, 273, 201, 251, 263")
    line = st.number_input("Prop line", value=199.5, step=0.5)
    window = st.slider("Rolling window (games)", 3, 10, 5)
    payout = st.number_input("Decimal payout (e.g. 1.91)", value=1.91, step=0.01)

try:
    yards = np.array([float(x) for x in raw.split(",") if x.strip()])
except ValueError:
    st.error("Couldn't parse the yardages — use comma-separated numbers.")
    st.stop()

model = NormalCDFBaseline(window=window)
res = model.predict(yards, line)
edge = model.edge_vs_payout(yards, line, payout)

c1, c2, c3 = st.columns(3)
c1.metric("P(hit)", f"{res['p_hit']:.1%}")
c2.metric("Break-even needed", f"{edge['p_needed']:.1%}")
c3.metric("Edge", f"{edge['edge']:+.1%}")

# distribution plot with the line marked
xs = np.linspace(res["mu"] - 4 * res["sigma"], res["mu"] + 4 * res["sigma"], 400)
ys = norm.pdf(xs, res["mu"], res["sigma"])
fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.fill_between(xs[xs > line], ys[xs > line], alpha=0.3, label="hit region")
ax.axvline(line, linestyle="--", label=f"line {line}")
ax.set_xlabel("Passing yards")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

if edge["edge"] > 0:
    st.success("Model says +EV at this payout.")
else:
    st.warning("Model says no edge at this payout.")
