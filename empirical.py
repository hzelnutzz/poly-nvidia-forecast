#!/usr/bin/env python3
"""Empirical quantiles instead of the normal's, with leave-one-case-out CV.

The house baseline assumes Gaussian residuals. Fit the residual distribution
of (actual - last) / (v * sqrt(h)) on the training cases and use ITS quantiles
as the forecast fan. This is legit: the practice cases carry only price
history, no answer file, and CV keeps us honest about overfit.
"""
import math
import glob
import json
import os

KIT = "/tmp/poly-kit/polymarket-nvidia-ceo-forecast-01-kit/input"
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
H = 240
RECENT = 120


def load():
    out = []
    for f in sorted(glob.glob(os.path.join(KIT, "*.in"))):
        out.append((os.path.basename(f), json.load(open(f))))
    return out


def pinball(pred, act, qs=QS):
    t = 0.0
    for row, a in zip(pred, act):
        for q, p in zip(qs, row):
            d = a - p
            t += (q - 1) * d if d < 0 else q * d
    return t / (len(pred) * len(qs))


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def vol(prices, n=RECENT, mode="mad", scale=1.0):
    m = min(n, len(prices) - 1)
    if m < 3:
        return 0.002
    d = [prices[i] - prices[i - 1] for i in range(len(prices) - m, len(prices))]
    if mode == "mad":
        return (sum(abs(x) for x in d) / len(d)) * math.sqrt(math.pi / 2) * scale
    mu = sum(d) / len(d)
    var = sum((x - mu) ** 2 for x in d) / (len(d) - 1)
    return math.sqrt(max(0, var)) * scale


def paths_of(d, step=120, max_cp=8):
    mk = d["markets"]
    n = len(mk[0]["prices"])
    cps, pos = [], n - H
    while pos >= RECENT + 30 and len(cps) < max_cp:
        cps.append(pos)
        pos -= step
    out = []
    for cp in cps:
        for m in mk:
            pr = m["prices"]
            if len(pr) > cp:
                hist, act = pr[:cp], pr[cp:cp + H]
                if len(act) == H:
                    out.append((hist, act))
    return out


def residuals(paths, vol_mode="mad", vol_scale=1.0, floor=0.0):
    """(z_residual, h, horizon_weight) for training."""
    rs = []
    for hist, act in paths:
        x = hist[-1]
        v = min(0.02, max(floor, vol(hist, mode=vol_mode, scale=vol_scale)))
        for h, a in enumerate(act, 1):
            s = v * math.sqrt(h)
            if s <= 0:
                continue
            rs.append((a - x) / s)
    return rs


def empirical_z(rs, qs=QS):
    rs = sorted(rs)
    n = len(rs)
    return [rs[min(n - 1, max(0, int(q * (n - 1))))] for q in qs]


def forecast(hist, zs, vol_mode="mad", vol_scale=1.0, floor=0.0):
    x = hist[-1]
    v = min(0.02, max(floor, vol(hist, mode=vol_mode, scale=vol_scale)))
    rows = []
    for h in range(1, H + 1):
        s = v * math.sqrt(h)
        row = [clamp(x + z * s) for z in zs]
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def baseline(hist):
    x = hist[-1]
    return [[clamp(x + z * 0.0025 * math.sqrt(h)) for z in Z] for h in range(1, H + 1)]


def cv():
    cases = load()
    per = {name: paths_of(d) for name, d in cases}
    configs = [
        ("mad_1.0", dict(vol_mode="mad", vol_scale=1.0, floor=0.001)),
        ("mad_0.85", dict(vol_mode="mad", vol_scale=0.85, floor=0.001)),
        ("var_1.0", dict(vol_mode="var", vol_scale=1.0, floor=0.001)),
        ("mad_1.15", dict(vol_mode="mad", vol_scale=1.15, floor=0.002)),
    ]
    for cname, cfg in configs:
        print(f"\n=== config {cname}")
        tot_base = tot_model = 0.0
        n = 0
        zrows = []
        for held, _ in cases:
            train = [p for nm, ps in per.items() if nm != held for p in ps]
            rs = residuals(train, **cfg)
            zs = empirical_z(rs)
            zrows.append((held, [round(z, 4) for z in zs]))
            for hist, act in per[held]:
                tot_model += pinball(forecast(hist, zs, **cfg), act)
                tot_base += pinball(baseline(hist), act)
                n += 1
        print(f"paths {n}  base {tot_base/n:.6f}  model {tot_model/n:.6f}  ratio {tot_model/tot_base:.4f}")
        for held, zs in zrows:
            print(f"   {held}: {zs}")
    # full-data z for the final fit
    allpaths = [p for ps in per.values() for p in ps]
    rs = residuals(allpaths)
    print("\nfull-sample empirical z:", [round(z, 4) for z in empirical_z(rs)])
    print("gaussian z           :", Z)


if __name__ == "__main__":
    cv()
