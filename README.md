# NFL Prop Predictor

Predict whether an NFL player prop line will hit. The model estimates the true
probability from game-log data and compares it against the sportsbook's implied
probability to flag +EV picks.

Built as a resume project: data collection → modeling → honest evaluation →
shipped dashboard.

## How it works

**v0 baseline (working):** rolling mean + rolling std dev of a QB's recent games
→ normal CDF → P(yards > line). Simple, transparent, and the benchmark every
fancier model has to beat.

**Roadmap:**
1. `v0` — normal-CDF baseline (done)
2. Feature engineering: rolling windows, home/away splits, opponent defensive rank, rest days
3. `v1` — distributional regression (NGBoost / quantile GBM): predict the full yardage
   distribution so P(over) works for *any* line
4. Walk-forward backtest + calibration curve + paper-trading ROI
5. Streamlit dashboard deployed publicly

## Project layout

```
nfl-prop-predictor/
├── src/
│   ├── data.py       # load NFL game logs via nflverse (nfl_data_py)
│   ├── features.py   # rolling-window feature engineering
│   ├── baseline.py   # v0 normal-CDF model
│   └── evaluate.py   # walk-forward backtest, calibration
├── app/
│   └── streamlit_app.py  # dashboard: pick a QB, set a line, see P(hit) + edge
├── examples/
│   └── hoover_example.py # the original calculation this project grew out of (CFB)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python examples/hoover_example.py   # reproduce the original v0 math
streamlit run app/streamlit_app.py  # launch the dashboard
```

## Honest limitations

- Small samples per player make std-dev estimates noisy.
- Players get hurt, schemes change, weather happens — the model doesn't know that (non-stationarity).
- Averages don't know opponent strength; a line vs. a top-5 defense isn't the same as vs. a bottom-5 one.
- Paper trading only. This is a modeling project, not betting advice.
