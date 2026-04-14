#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCDI: tune sparsity / stability by held-out (validation) Gaussian log-likelihood,
then evaluate SHD / skeleton SHD on test.

Modes:
  - vanilla: one DCDI run on train; tune τ or top-k on val; evaluate on test
  - jstable: many runs on train; per-seed postproc; aggregate frequencies; tune π on val; evaluate on test

Requirements in PATH:
  - your DCDI runner (e.g., dcdi_run_patched_v15.py) that writes W.csv under --outdir
  - Python libs: numpy, pandas

Author: (you)
"""
import argparse, json, os, subprocess, sys, tempfile, shutil, math
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- utils ----------

def read_df(path):
    return pd.read_csv(path)

def split_train_val_test(df, seed=123, frac=(0.6,0.2,0.2)):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_tr = int(frac[0]*n)
    n_va = int(frac[1]*n)
    tr = idx[:n_tr]
    va = idx[n_tr:n_tr+n_va]
    te = idx[n_tr+n_va:]
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)

def standardize_train_apply(train_df, apply_df, cols):
    mu = train_df[cols].mean(0).values
    sd = train_df[cols].std(0).values
    sd[sd==0] = 1.0
    Xtr = (train_df[cols].values - mu)/sd
    Xap = (apply_df[cols].values - mu)/sd
    return Xtr, Xap, mu, sd

def gaussian_val_loglik(train_df, val_df, adj, names):
    """
    Surrogate Gaussian LL on val:
      for each node j, fit OLS y ~ X_Parents on train, compute NLL on val with variance from train residuals.
    """
    p = len(names)
    cols = names
    Xtr, Xva, mu, sd = standardize_train_apply(train_df, val_df, cols)
    ll = 0.0
    nva = Xva.shape[0]
    for j in range(p):
        parents = [i for i in range(p) if adj[i,j]==1 and i!=j]
        ytr = Xtr[:, j]
        yva = Xva[:, j]
        if len(parents)==0:
            # zero-mean Gaussian with var = var(y_tr)
            var = float(np.var(ytr))
            var = max(var, 1e-8)
            # sum log-likelihood across val
            ll += np.sum(-0.5*(np.log(2*np.pi*var) + (yva**2)/var))
        else:
            Xp_tr = Xtr[:, parents]
            # OLS fit on train
            beta, _, _, _ = np.linalg.lstsq(Xp_tr, ytr, rcond=None)
            resid_tr = ytr - Xp_tr.dot(beta)
            var = float(np.var(resid_tr))
            var = max(var, 1e-8)
            # LL on val
            Xp_va = Xva[:, parents]
            resid_va = yva - Xp_va.dot(beta)
            ll += np.sum(-0.5*(np.log(2*np.pi*var) + (resid_va**2)/var))
    return float(ll)

def shd_and_skel(G, H):
    """ Directed SHD = skeleton edits + orientation disagreement among common skeleton edges. """
    p = G.shape[0]
    SkG = ((G + G.T) > 0).astype(int); np.fill_diagonal(SkG, 0)
    SkH = ((H + H.T) > 0).astype(int); np.fill_diagonal(SkH, 0)
    skel_diff = int((SkG ^ SkH).sum()) // 2
    # orientation disagreements
    common = (SkG & SkH)
    orient = 0
    for i in range(p):
        for j in range(i+1,p):
            if common[i,j]:
                gij = G[i,j]-G[j,i]; hij = H[i,j]-H[j,i]
                if gij!=0 and hij!=0 and gij!=hij: orient += 1
    shd_dir = skel_diff + orient
    shd_skel = int((SkG ^ SkH).sum()) // 2
    return shd_dir, shd_skel

def load_W(outdir):
    W = pd.read_csv(Path(outdir)/"W.csv", index_col=0)
    names = list(W.columns)
    M = W.values.astype(float)
    return names, M

def adj_from_tau(W, tau):
    A = (np.abs(W) > float(tau)).astype(int)
    np.fill_diagonal(A, 0)
    return A

def adj_from_topk(W, K):
    p = W.shape[0]
    A = np.abs(W).copy()
    np.fill_diagonal(A, -np.inf)
    flat = A.ravel()
    idxs = np.argpartition(flat, -K)[-K:]
    G = np.zeros_like(A, dtype=int)
    for idx in idxs:
        i, j = divmod(int(idx), p)
        if i != j: G[i,j] = 1
    np.fill_diagonal(G, 0)
    return G

def write_adj(path, names, G):
    pd.DataFrame(G, index=names, columns=names).to_csv(path)

def run_runner(runner, data_csv, outdir, iters=20000, lr=1e-3, batch=128, seed=123):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, runner,
        "--data", str(data_csv),
        "--outdir", str(outdir),
        "--tau", "0.05", "--undirected",
        "--", "--num-train-iter", str(iters),
        "--train-batch-size", str(batch),
        "--eval-num-samples", "512",
        "--lr", str(lr), "--optimizer", "sgd", "--momentum", "0.9", "--device", "cpu",
        "--mu-init", "0.5", "--gamma-init", "1e-2", "--omega-mu", "1.0", "--omega-gamma", "1.0", "--h-threshold", "1e-4",
        "--seed", str(seed),
    ]
    print("[runner] "+" ".join(cmd))
    subprocess.run(cmd, check=True)

# ---------- vanilla pipeline ----------

def tune_eval_vanilla(args):
    data = read_df(args.data)
    # drop env/non-feature cols if present
    feat_cols = [c for c in data.columns if c not in (args.env_col,)]
    df = data[feat_cols].copy()
    tr, va, te = split_train_val_test(df, seed=args.split_seed)

    work = Path(args.outdir); work.mkdir(parents=True, exist_ok=True)
    (work/"splits").mkdir(exist_ok=True)
    tr.to_csv(work/"splits/train.csv", index=False)
    va.to_csv(work/"splits/val.csv", index=False)
    te.to_csv(work/"splits/test.csv", index=False)

    # train on train.csv
    train_out = work/"train_run"
    run_runner(args.runner, work/"splits/train.csv", train_out, iters=args.iters, lr=args.lr, batch=args.batch, seed=args.seed)

    names, W = load_W(train_out)
    # grids
    tau_grid = None
    topk_grid = None
    if args.tau_grid:
        # e.g. "0.005,0.01,0.02,0.03,0.05"
        tau_grid = [float(x) for x in args.tau_grid.split(",")]
    if args.topk_grid:
        # e.g. "10,15,17,20"
        topk_grid = [int(x) for x in args.topk_grid.split(",")]

    if (tau_grid is None) and (topk_grid is None):
        # default: tau quantiles
        off = np.abs(W)[~np.eye(W.shape[0], dtype=bool)]
        qs = np.linspace(0.80, 0.99, 10)
        tau_grid = np.unique(np.quantile(off, qs)).tolist()

    # tune by val LL (Gaussian surrogate)
    best = {"score": -1e99, "how": None, "param": None, "G": None}
    for tau in (tau_grid or []):
        G = adj_from_tau(W, tau)
        score = gaussian_val_loglik(tr, va, G, names)
        if score > best["score"]:
            best = {"score": score, "how": "tau", "param": tau, "G": G}
    for k in (topk_grid or []):
        G = adj_from_topk(W, k)
        score = gaussian_val_loglik(tr, va, G, names)
        if score > best["score"]:
            best = {"score": score, "how": "topk", "param": k, "G": G}

    # evaluate on test
    GT = pd.read_csv(args.gt, index_col=0).reindex(index=names, columns=names).fillna(0).values.astype(int)
    Gbest = best["G"]
    shd, shd_skel = shd_and_skel(Gbest, GT)
    # F1/PR directed
    TP = int((Gbest & GT).sum()); FP = int((Gbest & (1-GT)).sum()); FN = int(((1-Gbest) & GT).sum())
    prec = TP/(TP+FP) if TP+FP else 0.0; rec = TP/(TP+FN) if TP+FN else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0

    # write outputs
    write_adj(work/"G_chosen.csv", names, Gbest)
    curve = []
    if tau_grid:
        for tau in tau_grid:
            G = adj_from_tau(W, tau)
            score = gaussian_val_loglik(tr, va, G, names)
            curve.append({"how":"tau","param":float(tau),"val_ll":float(score),"edges":int(G.sum())})
    if topk_grid:
        for k in topk_grid:
            G = adj_from_topk(W, k)
            score = gaussian_val_loglik(tr, va, G, names)
            curve.append({"how":"topk","param":int(k),"val_ll":float(score),"edges":int(G.sum())})
    pd.DataFrame(curve).to_csv(work/"selection_curve.csv", index=False)

    summary = {
        "mode":"vanilla",
        "selection": {"how": best["how"], "param": best["param"], "val_ll": best["score"]},
        "directed": {"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"F1":f1,"SHD":shd, "edges": int(Gbest.sum())},
        "skeleton": {"SHD_skeleton": shd_skel}
    }
    (work/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[write] chosen adjacency -> {work/'G_chosen.csv'}; curves -> {work/'selection_curve.csv'}")

# ---------- j-stable pipeline ----------

def tune_eval_jstable(args):
    data = read_df(args.data)
    feat_cols = [c for c in data.columns if c not in (args.env_col,)]
    df = data[feat_cols].copy()
    tr, va, te = split_train_val_test(df, seed=args.split_seed)

    work = Path(args.outdir); work.mkdir(parents=True, exist_ok=True)
    spl = work/"splits"; spl.mkdir(exist_ok=True)
    tr.to_csv(spl/"train.csv", index=False)
    va.to_csv(spl/"val.csv", index=False)
    te.to_csv(spl/"test.csv", index=False)

    # multiple DCDI runs on train (or bootstrap subsets of train)
    runs_dir = work/"runs"; runs_dir.mkdir(exist_ok=True)
    names0 = None
    Ws = []
    for s in args.seeds:
        run_out = runs_dir/f"run_{s}"
        # optional bootstrap
        if args.bootstrap > 0:
            rng = np.random.RandomState(s)
            idx = rng.choice(len(tr), size=len(tr), replace=True)
            tr_boot = tr.iloc[idx].reset_index(drop=True)
            tmp_csv = run_out/"train_boot.csv"; run_out.mkdir(exist_ok=True)
            tr_boot.to_csv(tmp_csv, index=False)
            data_csv = tmp_csv
        else:
            data_csv = spl/"train.csv"
        run_runner(args.runner, data_csv, run_out, iters=args.iters, lr=args.lr, batch=args.batch, seed=s)
        names, W = load_W(run_out)
        if names0 is None: names0 = names
        elif names != names0: raise RuntimeError("Name mismatch across runs")
        Ws.append(W)

    p = len(names0)
    # per-seed postproc: choose top-k (= |E(GT)|) OR use a per-seed tau; here: topk=Kgrid (default: a single K=|E(GT)|)
    GT = pd.read_csv(args.gt, index_col=0).reindex(index=names0, columns=names0).fillna(0).values.astype(int)
    K_gt = int((GT>0).sum())
    Klist = [K_gt] if args.topk_grid is None else [int(x) for x in args.topk_grid.split(",")]

    # build frequencies for each (per-seed K) option
    records = []
    for K in Klist:
        A_seeds = []
        for W in Ws:
            A_seeds.append(adj_from_topk(W, K))
        A_seeds = np.array(A_seeds, dtype=int)  # [B, p, p]
        F = A_seeds.mean(axis=0)  # frequencies
        # scan pi
        pilist = [float(x) for x in args.pi_list.split(",")]
        for pi in pilist:
            Gpi = (F >= pi).astype(int); np.fill_diagonal(Gpi, 0)
            # val LL
            score = gaussian_val_loglik(tr, va, Gpi, names0)
            records.append({"K":K, "pi":pi, "val_ll":score, "edges": int(Gpi.sum())})

    dfrec = pd.DataFrame(records).sort_values(["val_ll"], ascending=False)
    dfrec.to_csv(work/"selection_curve_jstable.csv", index=False)
    best = dfrec.iloc[0].to_dict()
    Kbest, pibest = int(best["K"]), float(best["pi"])
    # rebuild chosen G
    A_seeds = [adj_from_topk(W, Kbest) for W in Ws]
    F = np.mean(A_seeds, axis=0)
    Gbest = (F >= pibest).astype(int); np.fill_diagonal(Gbest, 0)
    write_adj(work/"G_chosen.csv", names0, Gbest)
    pd.DataFrame(F, index=names0, columns=names0).to_csv(work/"F_freq.csv")

    # test metrics
    shd, shd_skel = shd_and_skel(Gbest, GT)
    TP = int((Gbest & GT).sum()); FP = int((Gbest & (1-GT)).sum()); FN = int(((1-Gbest) & GT).sum())
    prec = TP/(TP+FP) if TP+FP else 0.0; rec = TP/(TP+FN) if TP+FN else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0

    summary = {
        "mode":"jstable",
        "B": len(Ws),
        "selection": {"per_seed_topk": Kbest, "pi": pibest, "val_ll": float(best["val_ll"])},
        "directed": {"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"F1":f1,"SHD":shd, "edges": int(Gbest.sum())},
        "skeleton": {"SHD_skeleton": shd_skel}
    }
    (work/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[write] chosen adjacency -> {work/'G_chosen.csv'}; F -> {work/'F_freq.csv'}; curves -> {work/'selection_curve_jstable.csv'}")

# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser(description="Tune DCDI sparsity / j-stable π by held-out Gaussian LL and evaluate SHD")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # shared
    def add_shared(p):
        p.add_argument("--data", required=True, help="data.csv")
        p.add_argument("--gt", required=True, help="A_true.csv (square adjacency)")
        p.add_argument("--outdir", required=True)
        p.add_argument("--runner", required=True, help="Path to DCDI runner script")
        p.add_argument("--iters", type=int, default=20000)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--batch", type=int, default=128)
        p.add_argument("--seed", type=int, default=123)
        p.add_argument("--split-seed", type=int, default=42)
        p.add_argument("--env-col", default=None, help="Name of env column to ignore if present (drop from features)")

    pvan = sub.add_parser("vanilla", help="vanilla DCDI: tune τ/top-k by val LL")
    add_shared(pvan)
    pvan.add_argument("--tau-grid", default=None, help="comma list of tau values (e.g. 0.01,0.02,0.03)")
    pvan.add_argument("--topk-grid", default=None, help="comma list of top-k values")

    pjst = sub.add_parser("jstable", help="j-stable DCDI: B runs on train, tune π by val LL")
    add_shared(pjst)
    pjst.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    pjst.add_argument("--pi-list", default="0.30,0.35,0.40,0.45,0.50")
    pjst.add_argument("--topk-grid", default=None, help="per-seed top-k options (default: K=|E(GT)|)")
    pjst.add_argument("--bootstrap", type=int, default=0, help=">0 to bootstrap train per run (use seed for RNG)")

    args = ap.parse_args()
    if args.cmd == "vanilla":
        tune_eval_vanilla(args)
    else:
        args.seeds = [int(x) for x in args.seeds.split(",")]
        tune_eval_jstable(args)

if __name__ == "__main__":
    main()
