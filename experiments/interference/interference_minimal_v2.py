#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal interference simulation with overlapping covers to illustrate j-stability.
Writes results to --outdir (default: ./interference_out).
"""

import argparse
import numpy as np, pandas as pd
from pathlib import Path

def bump(angle_deg, center, width=40):
    d = np.abs((angle_deg - center + 180) % 360 - 180)   # circular distance
    x = np.clip(1 - d/width, 0, 1)
    return 0.5*(1 + np.cos(np.pi*(1 - x))) * (x>0)

def standardize(X):
    m = X.mean(0); s = X.std(0); s[s==0] = 1.0
    return (X - m)/s

def edge_presence(X, y, thresh=0.2):
    Xs = standardize(X); ys = (y - y.mean())/ (y.std() if y.std()!=0 else 1.0)
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    e1 = float(abs(beta[0]) >= thresh)   # E1 -> Y
    e2 = float(abs(beta[1]) >= thresh)   # E2 -> Y
    return e1, e2, beta

def frequency_on_cover(df, mask, K=10, seed=123):
    idx = np.where(mask.values)[0]
    if len(idx) < 200:   # too small to be meaningful
        return np.nan, np.nan, 0, []
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    shards = np.array_split(idx, K)
    pres = []
    for shard in shards:
        X = df.loc[shard, ["E1","E2"]].values
        y = df.loc[shard, "Y"].values
        e1, e2, beta = edge_presence(X, y)
        pres.append((e1,e2,beta))
    pres = np.array(pres, dtype=object)
    f1 = np.mean(pres[:,0].astype(float))
    f2 = np.mean(pres[:,1].astype(float))
    return f1, f2, len(idx), pres

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="interference_out", help="where to write outputs")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--charts", type=int, default=10, help="charts per cover")
    ap.add_argument("--pi", default="0.4,0.5,0.6,0.7")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    # ---------- simulate ----------
    T = 20000
    theta = rng.uniform(0, 360, size=T)            # degrees
    mix   = rng.normal(0, 1, size=T)               # mixing height proxy
    Z1    = rng.binomial(1, 0.5, size=T)           # plant 1 on/off
    Z2    = rng.binomial(1, 0.5, size=T)           # plant 2 on/off

    w1 = bump(theta, center=270, width=50)  # P1 influences when wind blows eastward
    w2 = bump(theta, center=90,  width=50)  # P2 influences when wind blows westward

    E1 = w1*Z1 + 0.1*rng.normal(size=T)
    E2 = w2*Z2 + 0.1*rng.normal(size=T)
    Y  = 1.0*E1 + 0.5*E2 + rng.normal(scale=0.5, size=T)

    df = pd.DataFrame({"theta":theta, "mix":mix, "Z1":Z1, "Z2":Z2, "E1":E1, "E2":E2, "Y":Y})

    # ---------- covers (overlapping) ----------
    covers = {
        "WS": ( (df["theta"]>=250) & (df["theta"]<=290) ),
        "WL": ( (df["theta"]>=230) & (df["theta"]<=310) ),
        "E" : ( (df["theta"]>=70)  & (df["theta"]<=110) ),
        "LM": ( df["mix"] <  -0.5 ),
    }
    covers["WL∩LM"] = covers["WL"] & covers["LM"]
    covers["E∩LM"]  = covers["E"]  & covers["LM"]

    # ---------- compute frequencies ----------
    pi_grid = [float(x) for x in args.pi.split(",") if x.strip()]
    rows, details = [], {}

    for name, m in covers.items():
        f1, f2, n, pres = frequency_on_cover(df, m, K=args.charts, seed=args.seed)
        details[name] = pres
        row = {"cover":name, "N":n, "freq_E1→Y":f1, "freq_E2→Y":f2}
        for pi in pi_grid:
            row[f"stable_E1@{pi}"] = int(f1 >= pi) if np.isfinite(f1) else np.nan
            row[f"stable_E2@{pi}"] = int(f2 >= pi) if np.isfinite(f2) else np.nan
        rows.append(row)

    tab = pd.DataFrame(rows).sort_values("cover")

    # ---------- write outputs ----------
    out = Path(args.outdir); out.mkdir(exist_ok=True, parents=True)
    tab.to_csv(out/"stability_by_cover.csv", index=False)

    ex = "WL∩LM" if "WL∩LM" in details else list(details.keys())[0]
    pres = details.get(ex, [])
    if len(pres) > 0:
        B = np.stack([np.array(b) for _,_,b in pres], axis=0).squeeze()
        pd.DataFrame(B, columns=["beta_E1","beta_E2"]).to_csv(out/f"betas_{ex}.csv", index=False)

    print("\n=== j-stable frequencies by cover ===")
    print(tab.to_string(index=False))
    print(f"\n[ok] wrote:\n  - {out/'stability_by_cover.csv'}\n  - {out/f'betas_{ex}.csv'} (if present)")

if __name__ == "__main__":
    main()
