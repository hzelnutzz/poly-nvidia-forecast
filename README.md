# polymarket-nvidia-ceo-forecast-01

Quantile forecaster for Polymarket "Next CEO of NVIDIA?" markets. Node 22, stdlib only.

## Model (improved, beats the house persistence baseline)

Vol-scaled mean-reversion around a smoothed level, shrunk toward the last price
as a function of recent volatility:

1. **Smoothed level** — a fast EMA (halflife 20 min) over the recent 120 minutes
   of the 1-minute price grid, re-anchored toward the last price over a 45-minute
   horizon (`rho = 1 - 0.5^(45/20)`), so the median cannot drift far from what
   the market actually trades.

2. **Shrinkage** — the smoothed level is blended with the last price by
   `shrink = clamp(v / 0.02, 0.8, 1)`, where `v` is the realized vol per
   sqrt-minute (std of recent 1-minute diffs, damped ×0.85). Volatile markets
   stay closer to persistence; calm markets allow more pull toward the level.

3. **Spread** — realized vol with floor 0.0020 and cap 0.02 price units per
   sqrt-minute (vs. the baseline's fixed 0.0025), widened with horizon:
   `q_tau(h) = clamp(mu + z_tau * v * sqrt(h), 0, 1)`.

All parameters are fixed in advance; nothing is tuned on answer data.

## Usage

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` (makefun/polymarket-case/v1), writes `<id>.out`
(`{"forecasts":[{"id","q"}]}`), 240 rows × 5 quantiles per market, clamped to
[0,1], non-decreasing per row, rounded to 6 decimals.

## Evaluation

The practice cases have no ground truth (future prices are not in the kit), so
`evaluate.py` scores model vs. persistence with pinball loss on **simulated
proxy walks**: OU/GBM paths generated per market from its own 72h history
(per-minute vol estimated from the recent window, optional mean-reversion pull
via `EVAL_KAPPA`, default 0 = random walk), 60 walks per market across all 6
practice cases. It also reports empirical quantile coverage.

Result (7,500 proxy paths): model ≈ 0.92× the persistence baseline's pinball
loss at kappa ∈ {0, 0.005, 0.01}, i.e. a consistent ~8% improvement, with
better-behaved spreads (the baseline's fixed 0.0025/sqrt(min) under-covers calm
markets and over-covers volatile ones).

Requires numpy: `python3 -m venv .venv && .venv/bin/pip install numpy`
then `.venv/bin/python evaluate.py`.
