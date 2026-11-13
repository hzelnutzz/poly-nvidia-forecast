#!/usr/bin/env python3
"""Sweep 7 (LOO-CV): does any center adjustment help the bucket-fan model?
Tests mean-reversion toward 0.5, toward the cross-market 1-sum level, and
short drift, all on top of the empirically fitted bucket fan."""
import math
import empirical as E
import final_model as FM

H = E.H
QS = E.QS
B = FM.B


def forecast(hist, zb, a_center=0.0, a_drift=0.0, drift_n=60, a_sum=0.0, ctx=None):
    x = hist[-1]
    v = FM.vb(hist)
    mu = x
    if a_center:
        mu += a_center * (0.5 - x)
    if a_drift:
        n = min(drift_n, len(hist) - 1)
        if n > 1:
            mu += a_drift * (hist[-1] - hist[-1 - n]) / n
    if a_sum and ctx:
        mu -= a_sum * (sum(ctx) - 1.0) / len(ctx)
    mu = E.clamp(mu)
    rows = []
    for h in range(1, H + 1):
        i = next(k for k, (lo, hi) in enumerate(B) if lo <= h <= hi)
        s = v * math.sqrt(h)
        row = [E.clamp(mu + z * s) for z in zb[i]]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def main():
    cases = E.load()
    # paths with context
    data = {}
    for nm, d in cases:
        mk = d["markets"]
        n = len(mk[0]["prices"])
        cps, pos = [], n - H
        while pos >= E.RECENT + 30 and len(cps) < 8:
            cps.append(pos)
            pos -= 120
        rows = []
        for cp in cps:
            ctx, live = [], []
            for m in mk:
                pr = m["prices"]
                if len(pr) > cp:
                    ctx.append(pr[cp - 1])
                    live.append(pr)
            for pr in live:
                act = pr[cp:cp + H]
                if len(act) == H:
                    rows.append((pr[:cp], act, list(ctx)))
        data[nm] = rows

    base = 0.0
    cnt = 0
    for nm, rows in data.items():
        for hist, act, ctx in rows:
            base += E.pinball(E.baseline(hist), act)
            cnt += 1
    base /= cnt
    print(f"persistence {base:.6f}")

    cfgs = [("none", dict())]
    for a in [-0.05, -0.02, 0.02, 0.05]:
        cfgs.append((f"center{a}", dict(a_center=a)))
    for a in [-0.3, -0.15, 0.15, 0.3]:
        cfgs.append((f"drift{a}", dict(a_drift=a)))
    for a in [0.05, 0.15, 0.3]:
        cfgs.append((f"sum{a}", dict(a_sum=a)))

    res = []
    for name, kw in cfgs:
        tot, n = 0.0, 0
        for held in data:
            train = [p[:2] for nm, rows in data.items() if nm != held for p in rows]
            zb = FM.fit_z(train)
            for hist, act, ctx in data[held]:
                tot += E.pinball(forecast(hist, zb, ctx=ctx, **kw), act)
                n += 1
        res.append(((tot / n) / base, name, tot / n))
    res.sort()
    print(f"{'ratio':>7s} {'pinball':>10s}  center adjustment")
    for r, name, p in res:
        print(f"{r:7.4f} {p:10.6f}  {name}")


if __name__ == "__main__":
    main()
