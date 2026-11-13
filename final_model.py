#!/usr/bin/env python3
"""Final candidate comparison: per-case detail + the model we will ship.

Ship candidate: per-horizon-bucket empirical quantiles of the standardized
residual, vol = blend of two realized-vol windows, floor 0.002, cap 0.02,
z clipped to [-3, 3] at short horizons (the extreme empirical values are
degenerate low-vol markets).
"""
import math
import empirical as E
import sweep3 as S3

H = E.H
QS = E.QS
B = S3.BUCKETS
ZCLIP = (-2.2, 2.6)


def vb(hist, scale=1.5, nwin=120, blend=0.5, floor=0.002, cap=0.02):
    v1 = E.vol(hist, n=nwin, mode="var", scale=scale)
    v2 = E.vol(hist, n=max(30, nwin // 2), mode="var", scale=scale)
    v = (1 - blend) * v1 + blend * v2
    return min(cap, max(floor, v))


def fit_z(paths, scale=1.5, nwin=120, blend=0.5):
    acc = {i: [] for i in range(len(B))}
    for hist, act in paths:
        x = hist[-1]
        v = vb(hist, scale, nwin, blend)
        for h, a in enumerate(act, 1):
            for i, (lo, hi) in enumerate(B):
                if lo <= h <= hi:
                    acc[i].append((a - x) / (v * math.sqrt(h)))
                    break
    zb = []
    for i in range(len(B)):
        rs = sorted(acc[i])
        n = len(rs)
        if not n:
            zb.append(list(E.Z))
            continue
        z = [rs[min(n - 1, max(0, round(q * (n - 1))))] for q in QS]
        zb.append([max(ZCLIP[0], min(ZCLIP[1], v)) for v in z])
    return zb


def forecast(hist, zb, scale=1.5, nwin=120, blend=0.5):
    x = hist[-1]
    v = vb(hist, scale, nwin, blend)
    rows = []
    for h in range(1, H + 1):
        i = next(k for k, (lo, hi) in enumerate(B) if lo <= h <= hi)
        s = v * math.sqrt(h)
        row = [E.clamp(x + z * s) for z in zb[i]]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def main():
    cases = E.load()
    per = {nm: E.paths_of(d) for nm, d in cases}
    print(f"{'case':10s} {'base':>9s} {'model':>9s} {'ratio':>7s}  paths")
    tb = tm = 0.0
    tcnt = 0
    for held, _ in cases:
        train = [p for nm, ps in per.items() if nm != held for p in ps]
        zb = fit_z(train)
        b = m = 0.0
        cnt = 0
        for hist, act in per[held]:
            b += E.pinball(E.baseline(hist), act)
            m += E.pinball(forecast(hist, zb), act)
            cnt += 1
        tb += b
        tm += m
        tcnt += cnt
        print(f"{held:10s} {b/cnt:9.6f} {m/cnt:9.6f} {m/b:7.4f}  {cnt}")
    print(f"{'TOTAL':10s} {tb/tcnt:9.6f} {tm/tcnt:9.6f} {tm/tb:7.4f}  {tcnt}")
    allp = [p for ps in per.values() for p in ps]
    zb = fit_z(allp)
    print("\nshipped z table (bucket: z at 5/25/50/75/95):")
    for (lo, hi), z in zip(B, zb):
        print(f"  {(lo,hi)}: {[round(v, 4) for v in z]}")


if __name__ == "__main__":
    main()
