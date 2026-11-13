#!/usr/bin/env python3
"""Real backtest for the polymarket forecaster.

Unlike evaluate.py (simulated walks), this uses the case's own 1-minute price
grid: put a checkpoint at position p (with >=240 minutes after it), forecast
the next 240 minutes from the data up to p, and score against the ACTUAL
grid prices. That is true out-of-sample loss on the same data-generating
process as the final.
"""
import json
import math
import os
import sys
import glob

KIT = "/tmp/poly-kit/polymarket-nvidia-ceo-forecast-01-kit/input"
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
H = 240
RECENT = 120


def load_cases():
    out = []
    for f in sorted(glob.glob(os.path.join(KIT, "*.in"))):
        d = json.load(open(f))
        out.append((os.path.basename(f), d))
    return out


def pinball(pred, actual, qs=QS):
    """Mean pinball loss over the 5 quantiles."""
    tot = 0.0
    n = 0
    for row, a in zip(pred, actual):
        for q, p in zip(qs, row):
            d = a - p
            tot += (q - 1) * d if d < 0 else q * d
            n += 1
    return tot / n


def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


# ---------------- baselines / models ----------------

def persistence(prices, ctx):
    x = prices[-1]
    rows = []
    for h in range(1, H + 1):
        s = 0.0025 * math.sqrt(h)
        rows.append([clamp01(x + z * s) for z in Z])
    return rows


def realized_vol(prices, n=RECENT, scale=0.85, mode="var"):
    m = min(n, len(prices) - 1)
    if m < 3:
        return 0.002
    diffs = [prices[i] - prices[i - 1] for i in range(len(prices) - m, len(prices))]
    if mode == "mad":
        mad = sum(abs(d) for d in diffs) / len(diffs)
        return mad * math.sqrt(math.pi / 2) * scale
    mu = sum(diffs) / len(diffs)
    var = sum((d - mu) ** 2 for d in diffs) / (len(diffs) - 1)
    return math.sqrt(max(0.0, var)) * scale


def trend(prices, n=60):
    """Net move over the last n minutes per minute."""
    n = min(n, len(prices) - 1)
    if n < 2:
        return 0.0
    return (prices[-1] - prices[-1 - n]) / n


def sum_dev(ctx):
    """Sum of last prices across markets minus 1 (markets are mutually exclusive)."""
    if not ctx:
        return 0.0
    return sum(ctx) - 1.0


def model_v3(prices, ctx=None, *, k_sum=0.35, drift_k=0.15, drift_n=60,
             vol_mode="mad", vol_scale=0.85, floor=0.0020, cap=0.02,
             spread_h=1.0, pin_shrink=True, pin_lo=0.06, pin_hi=0.94):
    """Cross-market sum reversion + damped drift + robust vol spread."""
    p = prices
    x = p[-1]
    n_mkt = max(1, len(ctx)) if ctx else 1
    dev = sum_dev(ctx)
    # per-market correction: pull toward the event-sum = 1
    adj = k_sum * dev / n_mkt
    mu0 = clamp01(x - adj)
    # drift: short-term momentum, damped (prediction markets mean-revert)
    v = realized_vol(p, scale=vol_scale, mode=vol_mode)
    dr = trend(p, drift_n) * drift_k
    mu = clamp01(mu0 + dr)
    sig = min(cap, max(floor, v)) * spread_h
    rows = []
    for h in range(1, H + 1):
        s = sig * math.sqrt(h)
        # pinned market: shrink the side that would leave [0,1]
        row = []
        for z in Z:
            lo_s = s * pin_shrink if (pin_shrink and x < pin_lo) else s
            hi_s = s * pin_shrink if (pin_shrink and x > pin_hi) else s
            ss = lo_s if z < 0 else hi_s
            row.append(clamp01(mu + z * ss))
        # enforce monotone
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


MODELS = {
    "persistence": lambda p, c: persistence(p, c),
    "v2_current": lambda p, c: model_v3(p, c, k_sum=0, drift_k=0, vol_mode="var", vol_scale=0.85, pin_shrink=False),
    "v3_sum": lambda p, c: model_v3(p, c),
    "v3_sum_drift": lambda p, c: model_v3(p, c, drift_k=0.15),
    "v3_mad_nopin": lambda p, c: model_v3(p, c, pin_shrink=False),
}


def run(step=240, max_cp=6):
    cases = load_cases()
    scores = {k: [] for k in MODELS}
    per_case = {k: {} for k in MODELS}
    n_paths = 0
    for name, d in cases:
        markets = d["markets"]
        # checkpoints: every `step` minutes, walking back from the end
        n = len(markets[0]["prices"]) if markets else 0
        cps = []
        p = n - H
        while p >= RECENT + 30 and len(cps) < max_cp:
            cps.append(p)
            p -= step
        for cp in cps:
            # context: last prices of all markets at this checkpoint
            ctx = []
            live = []
            for m in markets:
                pr = m["prices"]
                if len(pr) > cp:
                    ctx.append(pr[cp - 1])
                    live.append((m, pr))
            for m, pr in live:
                hist = pr[:cp]
                actual = pr[cp:cp + H]
                if len(actual) < H:
                    continue
                n_paths += 1
                for k, fn in MODELS.items():
                    pred = fn(hist, ctx)
                    l = pinball(pred, actual)
                    scores[k].append(l)
                    per_case[k].setdefault(name, []).append(l)
    print(f"paths scored: {n_paths}")
    base = sum(scores["persistence"]) / max(1, len(scores["persistence"]))
    print(f"{'model':16s} {'pinball':>10s} {'ratio':>8s}")
    for k in MODELS:
        v = scores[k]
        if not v:
            continue
        mean = sum(v) / len(v)
        print(f"{k:16s} {mean:10.6f} {mean/base:8.4f}")
    return scores, per_case


if __name__ == "__main__":
    run()
