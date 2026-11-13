#!/usr/bin/env python3
"""Sweep 5: fit the fan shape directly per horizon bucket --
q = x + v * G_bucket(tau), no sqrt(h) assumption."""
import math
import empirical as E
import sweep3 as S3

H = E.H
QS = E.QS
B = S3.BUCKETS


def g_by_bucket(train, mode, scale, nwin, buckets=B):
    acc = {i: [] for i in range(len(buckets))}
    for hist, act in train:
        x = hist[-1]
        v = min(0.02, max(0.001, E.vol(hist, n=nwin, mode=mode, scale=scale)))
        for h, a in enumerate(act, 1):
            for i, (lo, hi) in enumerate(buckets):
                if lo <= h <= hi:
                    acc[i].append((a - x) / v)
                    break
    out = []
    for i in range(len(buckets)):
        rs = sorted(acc[i])
        n = len(rs)
        out.append([rs[min(n - 1, max(0, round(q * (n - 1))))] for q in QS] if n else [0, 0, 0, 0, 0])
    return out


def fc(hist, gb, mode, scale, nwin, buckets=B):
    x = hist[-1]
    v = min(0.02, max(0.001, E.vol(hist, n=nwin, mode=mode, scale=scale)))
    rows = []
    for h in range(1, H + 1):
        i = next(k for k, (lo, hi) in enumerate(buckets) if lo <= h <= hi)
        row = [E.clamp(x + g * v) for g in gb[i]]
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

    bucket_sets = {
        "b5": [(1, 30), (31, 60), (61, 120), (121, 180), (181, 240)],
        "b8": [(1, 15), (16, 30), (31, 60), (61, 90), (91, 120), (121, 160), (161, 200), (201, 240)],
        "b12": [(1 + 20 * i, 20 * (i + 1)) for i in range(12)],
    }
    res = []
    for bname, buckets in bucket_sets.items():
        for mode in ["var", "mad"]:
            for scale in [1.0, 1.15, 1.3, 1.5]:
                for nwin in [120, 240]:
                    tot, cnt = 0.0, 0
                    for held, _ in cases:
                        train = [p for nm, ps in per.items() if nm != held for p in ps]
                        gb = g_by_bucket(train, mode, scale, nwin, buckets)
                        for hist, act in per[held]:
                            tot += E.pinball(fc(hist, gb, mode, scale, nwin, buckets), act)
                            cnt += 1
                    res.append(((tot / cnt) / base, f"{bname} {mode} s{scale} n{nwin}", tot / cnt, (buckets, mode, scale, nwin)))
    res.sort()
    print(f"{'ratio':>7s} {'pinball':>10s}  config")
    for r, name, p, _ in res[:15]:
        print(f"{r:7.4f} {p:10.6f}  {name}")

    buckets, mode, scale, nwin = res[0][3]
    print("\nbest:", res[0][1])
    allp = [p for ps in per.values() for p in ps]
    gb = g_by_bucket(allp, mode, scale, nwin, buckets)
    for (lo, hi), g in zip(buckets, gb):
        print(f"  h {lo:3d}-{hi:3d}: {[round(v, 3) for v in g]}")


if __name__ == "__main__":
    cv()
