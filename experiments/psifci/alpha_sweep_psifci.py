#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha sweep for PSI-FCI: pooled vs j-stable (intersection & k=E-1).

Generates:
  - <outdir>/alpha_sweep_psifci.csv   (metrics table)
  - <outdir>/alpha_sweep_f1.png       (F1 vs alpha)
  - <outdir>/alpha_sweep_shd.png      (SHD vs alpha)
  - Per-alpha artifacts under outdir/per_env_a<alpha> and outdir/pooled_a<alpha>

Assumptions:
  - run_psifci_jstable.py is in the same folder (or on PATH)
  - It writes per-env files named like fci_env*.csv  (fallbacks included)
  - CSVs are square adjacency matrices with headers (var names)

Usage example:
  python alpha_sweep_psifci.py \
    --data ./synth_jsheaf/synth_data.csv \
    --env-col env \
    --true ./synth_jsheaf/A_true_Jstable.csv \
    --alphas 0.005,0.01,0.02 \
    --outdir ./results_psifci_grid \
    --depth 0 --standardize --undirected
"""
import argparse, os, sys, json, glob, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def sh(cmd, cwd=None):
    print("[sh]", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return res

def natural_alphanum(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def find_env_csvs(d: Path):
    pats = ["fci_env*.csv", "fci_enve*.csv", "A_env*.csv", "A_env_*.csv"]
    hits = []
    for p in pats:
        hits.extend([str(x) for x in d.glob(p)])
    return sorted(set(hits), key=natural_alphanum)

def load_adj(path: str):
    df = pd.read_csv(path, index_col=0)
    # ensure square and same index/columns order
    assert set(df.index) == set(df.columns)
    df = df.reindex(index=sorted(df.index), columns=sorted(df.columns))
    return df

def align_adj(A: pd.DataFrame, names):
    # reindex to name order, fill NaNs with 0 (missing vars -> 0-rows/cols)
    return A.reindex(index=names, columns=names).fillna(0)

def make_undir_bool(A: np.ndarray):
    B = (A > 0)
    B = np.logical_or(B, B.T)
    np.fill_diagonal(B, False)
    return B

def bin_metrics(A_pred: np.ndarray, A_true: np.ndarray, undirected: bool):
    n = A_true.shape[0]
    if undirected:
        P = make_undir_bool(A_pred)
        T = make_undir_bool(A_true)
    else:
        P = (A_pred > 0)
        T = (A_true > 0)
        np.fill_diagonal(P, False)
        np.fill_diagonal(T, False)

    if undirected:
        tri = np.triu_indices(n, k=1)
        Pv = P[tri]; Tv = T[tri]
    else:
        mask = ~np.eye(n, dtype=bool)
        Pv = P[mask]; Tv = T[mask]

    TP = int(np.sum(np.logical_and(Pv, Tv)))
    FP = int(np.sum(np.logical_and(Pv, ~Tv)))
    FN = int(np.sum(np.logical_and(~Pv, Tv)))
    TN = int(np.sum(np.logical_and(~Pv, ~Tv)))

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # SHD (undirected): symmetric difference of edges
    if undirected:
        shd = int(np.sum(Pv != Tv))
    else:
        shd = int(np.sum(P != T))   # directed mismatch count over all entries

    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=round(prec,3),
                recall=round(rec,3), f1=round(f1,3), shd=shd)

# ----------------- main sweep -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--env-col", default=None)
    ap.add_argument("--true", required=True)
    ap.add_argument("--alphas", required=True, help="comma-separated, e.g. 0.005,0.01,0.02")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--undirected", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load ground truth & canonical variable order
    A_true_df = pd.read_csv(args.true, index_col=0)
    names_canon = list(A_true_df.index)
    A_true = A_true_df.values.astype(float)

    run_psifci = Path(__file__).with_name("run_psifci_jstable.py")
    if not run_psifci.exists():
        # allow PATH fallback
        run_psifci = Path("run_psifci_jstable.py")

    rows = []
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    for a in alphas:
        tag = f"a{int(a*1000):03d}"  # e.g., 0.005 -> a005
        per_env_dir = outdir / f"per_env_{tag}"
        pooled_dir  = outdir / f"pooled_{tag}"
        per_env_dir.mkdir(parents=True, exist_ok=True)
        pooled_dir.mkdir(parents=True, exist_ok=True)

        # -------- per-env PSI-FCI (writes fci_env*.csv) --------
        if args.env_col:
            cmd = [sys.executable, str(run_psifci),
                   "--data", args.data, "--env-col", args.env_col,
                   "--alpha", str(a), "--depth", str(args.depth),
                   "--outdir", str(per_env_dir)]
            if args.standardize: cmd.append("--standardize")
            sh(cmd)
            env_csvs = find_env_csvs(per_env_dir)
            if not env_csvs:
                raise RuntimeError(f"No per-env CSVs found in {per_env_dir}")

            # stack per-env adjacencies aligned to canonical var order
            A_stack = []
            for p in env_csvs:
                A_df = load_adj(p)
                A_df = align_adj(A_df, names_canon)
                A_stack.append(A_df.values.astype(float))
            A_stack = np.stack(A_stack, axis=0)  # (E,d,d)

            # j-stable intersection & k1 (support >= E-1)
            E = A_stack.shape[0]
            A_inter = (A_stack > 0).all(axis=0).astype(float)
            A_k1    = ((A_stack > 0).sum(axis=0) >= (E-1)).astype(float)

            # write aggregated adjacencies
            pd.DataFrame(A_inter, index=names_canon, columns=names_canon).to_csv(per_env_dir/"A_Jstable_intersection.csv")
            pd.DataFrame(A_k1,    index=names_canon, columns=names_canon).to_csv(per_env_dir/"A_Jstable_k1.csv")

            # score vs truth
            m_inter = bin_metrics(A_inter, A_true, undirected=args.undirected)
            m_k1    = bin_metrics(A_k1,    A_true, undirected=args.undirected)
            rows += [{"alpha": a, "method": "jstable_intersection", **m_inter}]
            rows += [{"alpha": a, "method": "jstable_k1",          **m_k1}]

        # -------- pooled PSI-FCI --------
        cmd = [sys.executable, str(run_psifci),
               "--data", args.data,
               "--alpha", str(a), "--depth", str(args.depth),
               "--outdir", str(pooled_dir)]
        if args.standardize: cmd.append("--standardize")
        sh(cmd)

        # find pooled CSV
        pooled_csv = pooled_dir / "fci_envpooled.csv"
        if not pooled_csv.exists():
            # fallback: any single CSV there
            candidates = list(pooled_dir.glob("*.csv"))
            if len(candidates) == 1:
                pooled_csv = candidates[0]
            else:
                raise FileNotFoundError(f"No pooled CSV at {pooled_csv} and ambiguous fallback: {candidates}")

        A_pooled_df = load_adj(str(pooled_csv))
        A_pooled_df = align_adj(A_pooled_df, names_canon)
        A_pooled = A_pooled_df.values.astype(float)

        m_pooled = bin_metrics(A_pooled, A_true, undirected=args.undirected)
        rows += [{"alpha": a, "method": "pooled", **m_pooled}]

    # -------- save CSV & plots --------
    df = pd.DataFrame(rows)
    df.sort_values(["alpha","method"], inplace=True)
    csv_out = outdir / "alpha_sweep_psifci.csv"
    df.to_csv(csv_out, index=False)
    print(f"[ok] wrote {csv_out}")

    # Plot F1 and SHD vs alpha
    methods = ["pooled", "jstable_intersection", "jstable_k1"]
    labels  = {"pooled":"PSI-FCI (pooled)",
               "jstable_intersection":"$j$-stable (intersection)",
               "jstable_k1":"$j$-stable ($k=E-1$)"}

    def lineplot(ykey, fname, ylabel):
        plt.figure(figsize=(5.0,3.2))
        for m in methods:
            d = df[df["method"]==m]
            if d.empty: continue
            plt.plot(d["alpha"], d[ykey], marker="o", label=labels[m])
        plt.xlabel(r"$\alpha$")
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        plt.tight_layout()
        out_png = outdir / fname
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[ok] wrote {out_png}")

    lineplot("f1",  "alpha_sweep_f1.png",  "F1")
    lineplot("shd", "alpha_sweep_shd.png", "SHD")

if __name__ == "__main__":
    main()
