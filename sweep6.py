#!/usr/bin/env python3
"""Sweep 6: binary-option scaling. A Yes price is a probability: its
diffusion is vol * sqrt(p(1-p)) not vol * const. Also blend two vol windows
and bucket the z by horizon."""
import math
import empirical as E
import sweep3 as S3

H = E.H
QS = E.QS
B = S3.BUCKETS


def vbase(hist, mode, scale, nwin, blend):
    v1 = E.vol(hist, n=nwin, mode=mode, scale=scale)
    if blend:
        v2 = E.vol(hist, n=max(30, nwin // 2), mode=mode, scale=scale)
        v1 = (1 - blend) * v1 + blend * v2
    return v1


def vb(hist, mode, scale, nwin, blend=0.0, binary=0.0, floor=0.0035, cap=0.02):
    """vol per sqrt-minute; binary scales it by 2*sqrt(p(1-p))."""
    x = hist[-1]
    v = vbase(hist, mode, scale, nwin, blend)
    if binary:
        f = 2 * math.sqrt(max(1e-6, x * (1 - x)))
        v = v * ((1 - binary) + binary * f)
    return min(cap, max(floor, v))


def fc(hist, zb, mode, scale, nwin, blend=0.0, binary=0.0, floor=0.0035):
    x = hist[-1]
    v = vb(hist, mode, scale, nwin, blend, binary, floor)
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


def cv():
    cases = E.load()
    per = {nm: E.paths_of(d) for nm, d in cases}
    base, n = 0.0, 0
    for nm, ps in per.items():
        for hist, act in ps:
            base += E.pinball(E.baseline(hist), act)
            n += 1
    base /= n
    print(f"persistence {base:.6f} over {n} paths")

    res = []
    for binary in [0.0, 0.5, 1.0]:
        for scale in [1.0, 1.2, 1.5, 1.8]:
            for blend in [0.0, 0.5]:
                for floor in [0.002, 0.0035, 0.005]:
                    tot, cnt = 0.0, 0
                    for held, _ in cases:
                        train = [p for nm, ps in per.items() if nm != held for p in ps]
                        zb = S3.z_by_bucket(train, "var", scale, 120)
                        for hist, act in per[held]:
                            tot += E.pinball(fc(hist, zb, "var", scale, 120, blend, binary, floor), act)
                            cnt += 1
                    res.append(((tot / cnt) / base, f"bin{binary} s{scale} bl{blend} fl{floor}", tot / cnt))
    res.sort()
    print(f"{'ratio':>7s} {'pinball':>10s}  config")
    for r, name, p in res[:15]:
        print(f"{r:7.4f} {p:10.6f}  {name}")
    return res


if __name__ == "__main__":
    cv()
