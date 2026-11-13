#!/usr/bin/env python3
"""Parameter sweep on the REAL backtest (actual future grid prices)."""
import math
import backtest as B

Z = B.Z
H = B.H


def make(center="persist", spread="const", sp=0.0025, vol_scale=0.85,
         vol_mode="mad", vol_n=120, floor=0.0, cap=9.0, hpow=0.5,
         drift_k=0.0, drift_n=60, k_sum=0.0, blend=0.0):
    def fn(prices, ctx):
        p = prices
        x = p[-1]
        dev = (sum(ctx) - 1.0) if ctx else 0.0
        n_mkt = len(ctx) if ctx else 1
        mu = x - k_sum * dev / n_mkt
        if drift_k:
            n = min(drift_n, len(p) - 1)
            if n > 1:
                mu += drift_k * (p[-1] - p[-1 - n]) / n
        mu = min(1.0, max(0.0, mu))
        if spread == "const":
            sig = sp
        else:
            v = B.realized_vol(p, n=vol_n, scale=vol_scale, mode=vol_mode)
            sig = min(cap, max(floor, v))
            if blend:
                sig = blend * sig + (1 - blend) * sp
        rows = []
        for h in range(1, H + 1):
            s = sig * (h ** hpow)
            row = [min(1.0, max(0.0, mu + z * s)) for z in Z]
            for i in range(1, 5):
                if row[i] < row[i - 1]:
                    row[i] = row[i - 1]
            rows.append(row)
        return rows
    return fn


def sweep():
    cases = B.load_cases()
    # pre-extract paths once
    paths = []
    for name, d in cases:
        markets = d["markets"]
        n = len(markets[0]["prices"]) if markets else 0
        cps = []
        pos = n - H
        while pos >= B.RECENT + 30 and len(cps) < 8:
            cps.append(pos)
            pos -= 120
        for cp in cps:
            ctx, live = [], []
            for m in markets:
                pr = m["prices"]
                if len(pr) > cp:
                    ctx.append(pr[cp - 1])
                    live.append((pr, pr[cp:cp + H]))
            for pr, act in live:
                if len(act) == H:
                    paths.append((pr[:cp], ctx, act))
    print("paths:", len(paths))

    base = sum(B.pinball(B.persistence(p, c), a) for p, c, a in paths) / len(paths)
    print(f"persistence baseline: {base:.6f}")

    cands = {}
    for sp in [0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005, 0.006]:
        cands[f"const_{sp}"] = make(spread="const", sp=sp)
    for vm in ["mad", "var"]:
        for vs in [0.6, 0.75, 0.85, 1.0]:
            for fl in [0.001, 0.002, 0.003]:
                cands[f"vol_{vm}_{vs}_f{fl}"] = make(spread="vol", vol_mode=vm, vol_scale=vs, floor=fl, cap=0.02)
    for bl in [0.25, 0.5, 0.75]:
        cands[f"blend_mad_{bl}"] = make(spread="vol", vol_mode="mad", vol_scale=0.85, floor=0.002, blend=bl)
    for hp in [0.4, 0.45, 0.5, 0.55]:
        cands[f"hpow_{hp}"] = make(spread="const", sp=0.0025, hpow=hp)
    for dk in [-0.3, -0.15, 0.05, 0.1, 0.2]:
        cands[f"drift_{dk}"] = make(spread="const", sp=0.0025, drift_k=dk)
    for ks in [0.05, 0.1, 0.2]:
        cands[f"sum_{ks}"] = make(spread="const", sp=0.0025, k_sum=ks)

    res = []
    for name, fn in cands.items():
        tot = 0.0
        for p, c, a in paths:
            tot += B.pinball(fn(p, c), a)
        mean = tot / len(paths)
        res.append((mean / base, name, mean))
    res.sort()
    print(f"{'ratio':>7s}  {'pinball':>10s}  model")
    for r, name, mean in res[:25]:
        print(f"{r:7.4f}  {mean:10.6f}  {name}")
    return res


if __name__ == "__main__":
    sweep()
