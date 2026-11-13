#!/usr/bin/env python3
"""Sweep 3: horizon-bucketed empirical quantiles (the fan shape is not a
random walk: short horizons are fatter-tailed, long ones are pinned by 0/1)."""
import math
import empirical as E

H = E.H
QS = E.QS
BUCKETS = [(1, 30), (31, 60), (61, 120), (121, 180), (181, 240)]


def vol_of(hist, mode, scale, nwin, floor=0.001):
    return min(0.02, max(floor, E.vol(hist, n=nwin, mode=mode, scale=scale)))


def z_by_bucket(train, mode, scale, nwin, buckets=BUCKETS):
    """For each bucket, empirical quantiles of residual / sqrt(h)."""
    acc = {i: [] for i in range(len(buckets))}
    for hist, act in train:
        x = hist[-1]
        v = vol_of(hist, mode, scale, nwin)
        for h, a in enumerate(act, 1):
            for i, (lo, hi) in enumerate(buckets):
                if lo <= h <= hi:
                    acc[i].append((a - x) / (v * math.sqrt(h)))
                    break
    out = []
    for i in range(len(buckets)):
        rs = sorted(acc[i])
        n = len(rs)
        if n == 0:
            out.append(list(E.Z))
            continue
        out.append([rs[min(n - 1, max(0, round(q * (n - 1))))] for q in QS])
    return out


def fc(hist, zb, mode, scale, nwin, hpow=0.5):
    x = hist[-1]
    v = vol_of(hist, mode, scale, nwin)
    rows = []
    for h in range(1, H + 1):
        i = next(k for k, (lo, hi) in enumerate(BUCKETS) if lo <= h <= hi)
        s = v * (h ** hpow)
        row = [E.clamp(x + z * s) for z in zb[i]]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def cv():
    cases = E.load()
    per = {nm: E.paths_of(d) for nm, d in cases}
    base = 0.0
    n = 0
    for nm, ps in per.items():
        for hist, act in ps:
            base += E.pinball(E.baseline(hist), act)
            n += 1
    base /= n
    print(f"persistence {base:.6f} over {n} paths")

    res = []
    for mode in ["var", "mad"]:
        for scale in [0.9, 1.0, 1.15, 1.3, 1.5]:
            for nwin in [60, 120, 240]:
                for hpow in [0.5]:
                    tot = 0.0
                    cnt = 0
                    for held, _ in cases:
                        train = [p for nm, ps in per.items() if nm != held for p in ps]
                        zb = z_by_bucket(train, mode, scale, nwin)
                        for hist, act in per[held]:
                            tot += E.pinball(fc(hist, zb, mode, scale, nwin, hpow), act)
                            cnt += 1
                    res.append(((tot / cnt) / base, f"{mode} s{scale} n{nwin} hp{hpow}", tot / cnt, (mode, scale, nwin, hpow)))
    res.sort()
    print(f"{'ratio':>7s} {'pinball':>10s}  config")
    for r, name, p, _ in res[:12]:
        print(f"{r:7.4f} {p:10.6f}  {name}")

    # report the per-bucket z of the best config on all data
    best = res[0][3]
    mode, scale, nwin, hpow = best
    allp = [p for ps in per.values() for p in ps]
    zb = z_by_bucket(allp, mode, scale, nwin)
    print("\nbest config:", best)
    for (lo, hi), z in zip(BUCKETS, zb):
        print(f"  h {lo:3d}-{hi:3d}: {[round(v, 4) for v in z]}")


if __name__ == "__main__":
    cv()
