#!/usr/bin/env python3
"""Sweep 2: empirical-z fan, with clipping + vol-window choices, LOO-CV."""
import math
import empirical as E

H = E.H


def residuals(paths, mode, scale, floor, nwin):
    rs = []
    for hist, act in paths:
        x = hist[-1]
        v = min(0.02, max(floor, E.vol(hist, n=nwin, mode=mode, scale=scale)))
        for h, a in enumerate(act, 1):
            s = v * math.sqrt(h)
            if s > 1e-9:
                rs.append((a - x) / s)
    return rs


def emp_z(rs, qs=E.QS, lo=None, hi=None):
    rs = sorted(rs)
    n = len(rs)
    z = [rs[min(n - 1, max(0, round(q * (n - 1))))] for q in qs]
    if lo is not None:
        z = [max(lo, v) for v in z]
    if hi is not None:
        z = [min(hi, v) for v in z]
    return z


def fc(hist, zs, mode, scale, floor, nwin):
    x = hist[-1]
    v = min(0.02, max(floor, E.vol(hist, n=nwin, mode=mode, scale=scale)))
    rows = []
    for h in range(1, H + 1):
        s = v * math.sqrt(h)
        row = [E.clamp(x + z * s) for z in zs]
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def cv():
    cases = E.load()
    per = {nm: E.paths_of(d) for nm, d in cases}
    base = 0.0
    nb = 0
    for nm, ps in per.items():
        for hist, act in ps:
            base += E.pinball(E.baseline(hist), act)
            nb += 1
    base /= nb
    print(f"persistence {base:.6f} over {nb} paths")

    cfgs = []
    for mode in ["var", "mad"]:
        for scale in [0.85, 1.0, 1.15]:
            for nwin in [60, 120, 240]:
                for clip in [(None, None), (-3, 3), (-2.5, 2.8), (-2.0, 2.5), (-3.5, 3.0)]:
                    cfgs.append((mode, scale, nwin, clip))

    res = []
    for mode, scale, nwin, (lo, hi) in cfgs:
        tot = 0.0
        n = 0
        for held, _ in cases:
            train = [p for nm, ps in per.items() if nm != held for p in ps]
            zs = emp_z(residuals(train, mode, scale, 0.001, nwin), lo=lo, hi=hi)
            for hist, act in per[held]:
                tot += E.pinball(fc(hist, zs, mode, scale, 0.001, nwin), act)
                n += 1
        res.append((tot / base, f"{mode} s{scale} n{nwin} clip{lo},{hi}", tot / n))
    res.sort()
    print(f"{'ratio':>7s} {'pinball':>10s}  config")
    for r, name, p in res[:15]:
        print(f"{r:7.4f} {p:10.6f}  {name}")

    # full-sample z for the winner
    best = res[0][1]
    print("\nbest:", best)


if __name__ == "__main__":
    cv()
