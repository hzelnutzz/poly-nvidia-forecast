#!/usr/bin/env python3
"""Evaluate the improved forecaster against the persistence baseline.

Ground truth for the practice cases is NOT available (the answer files hold the
future prices). So evaluation uses a proxy: simulated forward walks generated
from the market's own 72h price history. For each market and case we simulate
many GBM-style walks whose per-minute vol is estimated from the recent history
(same information the model has), plus a mild OU pull toward the recent mean,
score both forecasters on those walks with pinball loss, and report calibration
(empirical coverage of each quantile band).
"""

import json
import math
import os
import sys
import random
import numpy as np

KIT = "/tmp/poly-kit/polymarket-nvidia-ceo-forecast-01-kit/input"
OUT = os.environ.get("EVAL_OUT", "/tmp/poly-eval/out")
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
H = 240
N_WALKS = 60
RECENT = 120


def persistence(prices):
    x = prices[-1]
    sig = 0.0025
    return [[min(1, max(0, x + z * sig * math.sqrt(h))) for z in Z] for h in range(1, H + 1)]


def load_model_out(case_id):
    with open(os.path.join(OUT, f"{case_id}.out")) as f:
        d = json.load(f)
    return {m["id"]: m["q"] for m in d["forecasts"]}


def pinball(q, y, tau):
    d = y - q
    return max(tau * d, (tau - 1) * d)


def main():
    rng = random.Random(42)
    total_model = total_base = 0.0
    cover_model = {t: [0, 0] for t in QS}
    cover_base = {t: [0, 0] for t in QS}
    n_paths = 0
    for fn in sorted(os.listdir(KIT)):
        if not fn.endswith(".in"):
            continue
        case_id = fn[:-3]
        case = json.load(open(os.path.join(KIT, fn)))
        model = load_model_out(case_id)
        for m in case["markets"]:
            p = np.array(m["prices"], float)
            if len(p) < RECENT + 1:
                continue
            last = p[-1]
            rec = p[-RECENT:]
            mu_recent = rec.mean()
            diffs = np.diff(rec)
            sigma = max(1e-4, diffs.std()) * 0.85  # same SIGMA_SCALE damping as the model
            # OU kappa: mild pull toward the smoothed level (kappa ~0 keeps the
            # walk near persistence; a stronger pull punishes wide spreads)
            kappa = float(os.environ.get("EVAL_KAPPA", "0"))
            level = last + 0.65 * (mu_recent - last)
            # simulated walks as proxy ground truth
            paths = np.empty((N_WALKS, H))
            for w in range(N_WALKS):
                x = last
                for h in range(H):
                    x = x + kappa * (level - x) + sigma * rng.gauss(0, 1)
                    paths[w, h] = min(1.0, max(0.0, x))
            qm = np.array(model[m["id"]])
            qb = np.array(persistence(list(p)))
            for w in range(N_WALKS):
                y = paths[w]
                total_model += float(np.mean([pinball(qm[h, j], y[h], QS[j]) for h in range(H) for j in range(5)]))
                total_base += float(np.mean([pinball(qb[h, j], y[h], QS[j]) for h in range(H) for j in range(5)]))
                for j, t in enumerate(QS):
                    cover_model[t][0] += int((y <= qm[:, j]).sum())
                    cover_model[t][1] += H
                    cover_base[t][0] += int((y <= qb[:, j]).sum())
                    cover_base[t][1] += H
            n_paths += 1
    nm = n_paths * N_WALKS
    print(f"paths scored: {nm}  (markets x walks)")
    print(f"pinball (model/improved): {total_model / nm:.6f}")
    print(f"pinball (persistence)   : {total_base / nm:.6f}")
    print(f"ratio model/base: {total_model / total_base:.4f}")
    print("coverage (model vs base), target tau:")
    for t in QS:
        cm = cover_model[t][0] / cover_model[t][1]
        cb = cover_base[t][0] / cover_base[t][1]
        print(f"  tau={t:.2f}: model {cm:.3f}  base {cb:.3f}")


if __name__ == "__main__":
    main()
