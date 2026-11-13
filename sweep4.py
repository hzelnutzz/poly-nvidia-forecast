#!/usr/bin/env python3
"""Sweep 4: bucket z + boundary-aware spread (a price at 0.02 cannot fall
much), plus a vol blend of two windows."""
import math
import empirical as E
import sweep3 as S3

H = E.H
QS = E.QS
B = S3.BUCKETS


def vol_blend(hist, mode, scale, nwin, floor=0.001, blend=0.0):
    v1 = E.vol(hist, n=nwin, mode=mode, scale=scale)
    if not blend:
        v = v1
    else:
        v2 = E.vol(hist, n=max(30, nwin // 2), mode=mode, scale=scale)
        v = (1 - blend) * v1 + blend * v2
    return min(0.02, max(floor, v))


def fc(hist, zb, mode, scale, nwin, blend=0.0, bnd=0.0):
    x = hist[-1]
    v = vol_blend(hist, mode, scale, nwin, blend=blend)
    rows = []
    for h in range(1, H + 1):
        i = next(k for k, (lo, hi) in enumerate(B) if lo <= h <= hi)
        s = v * math.sqrt(h)
        row = []
        for z in zb[i]:
            ss = s
            if bnd:
                # a move of z*s cannot exceed the distance to the wall by more
                # than a factor: compress the tail that leaves [0,1]
                room = (1 - x) if z > 0 else x
                ss = min(s, bnd * max(room, 0.004))
            row.append(E.clamp(x + z * ss))
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
        for scale in [1.3, 1.5, 1.7, 2.0]:
            for nwin in [120]:
                for blend in [0.0, 0.3, 0.5]:
                    for bnd in [0.0, 0.25, 0.5, 1.0]:
                        tot, cnt = 0.0, 0
                        for held, _ in cases:
                            train = [p for nm, ps in per.items() if nm != held for p in ps]
                            zb = S3.z_by_bucket(train, mode, scale, nwin)
                            for hist, act in per[held]:
                                tot += E.pinball(fc(hist, zb, mode, scale, nwin, blend, bnd), act)
                                cnt += 1
                        res.append(((tot / cnt) / base, f"{mode} s{scale} n{nwin} blend{blend} bnd{bnd}", tot / cnt))
    res.sort()
    print(f"{'ratio':>7s} {'pinball':>10s}  config")
    for r, name, p in res[:15]:
        print(f"{r:7.4f} {p:10.6f}  {name}")
    return res


if __name__ == "__main__":
    cv()
