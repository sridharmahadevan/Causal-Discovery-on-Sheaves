#!/usr/bin/env python3
import argparse, json, glob, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

def load_W(path: Path) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    names = list(df.columns)
    M = df.values.astype(float)
    assert M.shape[0] == M.shape[1], f"{path} not square"
    return names, M

def load_gt(path: Path) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "src" in cols and "dst" in cols:
        src, dst = cols["src"], cols["dst"]
        names = sorted(set(df[src]).union(set(df[dst])))
        idx = {n:i for i,n in enumerate(names)}
        A = np.zeros((len(names), len(names)), dtype=int)
        for s,d in zip(df[src], df[dst]):
            A[idx[s], idx[d]] = 1
        return names, A
    else:
        df = pd.read_csv(path, index_col=0)
        names = list(df.columns)
        A = (df.values > 0).astype(int)
        assert A.shape[0] == A.shape[1], "GT adjacency must be square"
        return names, A

def align_names(names: List[str], target: List[str]) -> List[int]:
    m = {n:i for i,n in enumerate(target)}
    try:
        return [m.n for n in names]
    except Exception:
        raise SystemExit("Name mismatch between W.csv and GT; ensure you used the *aligned* GT or subset by names.")

def threshold_matrix(M: np.ndarray, mode: str, tau=None, auto=None, topk_edges=None, topk_per_node=None) -> np.ndarray:
    p = M.shape[0]
    A = np.abs(M).copy()
    np.fill_diagonal(A, -np.inf)
    if mode == "tau":
        G = (A > float(tau)).astype(int)
    elif mode == "auto":
        q = {"p90":0.9, "p95":0.95}[auto.lower()]
        thr = np.quantile(A[A>-np.inf], q)
        G = (A > thr).astype(int)
    elif mode == "topk_edges":
        K = int(topk_edges)
        flat = A.ravel()
        idxs = np.argpartition(flat, -K)[-K:]
        G = np.zeros_like(A, dtype=int)
        for idx in idxs:
            i, j = divmod(idx, p)
            if i != j: G[i,j] = 1
    elif mode == "topk_per_node":
        K = int(topk_per_node)
        G = np.zeros((p,p), dtype=int)
        for j in range(p):
            idxs = np.argpartition(A[:,j], -K)[-K:]
            for i in idxs:
                if i != j and math.isfinite(A[i,j]):
                    G[i, j] = 1
    else:
        raise ValueError("unknown threshold mode")
    np.fill_diagonal(G, 0)
    return G

def directed_metrics(G, H):
    TP = int((G & H).sum())
    FP = int((G & (1-H)).sum())
    FN = int(((1-G) & H).sum())
    prec = TP / (TP + FP) if (TP+FP)>0 else 0.0
    rec  = TP / (TP + FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    SkG = ((G + G.T) > 0).astype(int); np.fill_diagonal(SkG, 0)
    SkH = ((H + H.T) > 0).astype(int); np.fill_diagonal(SkH, 0)
    skel_diff = int((SkG ^ SkH).sum()) // 2
    common = (SkG & SkH)
    P = G.astype(int); Q = H.astype(int)
    orient_disagree = 0
    n = G.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if common[i,j]:
                gij = P[i,j] - P[j,i]; hij = Q[i,j] - Q[j,i]
                if gij != 0 and hij != 0 and gij != hij:
                    orient_disagree += 1
    shd = skel_diff + orient_disagree
    return {"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1,"SHD":shd,"edges":int(G.sum())}

def skeleton_metrics(G, H):
    SkG = ((G + G.T) > 0).astype(int); np.fill_diagonal(SkG, 0)
    SkH = ((H + H.T) > 0).astype(int); np.fill_diagonal(SkH, 0)
    TP = int((SkG & SkH).sum()) // 2
    FP = int((SkG & (1-SkH)).sum()) // 2
    FN = int(((1-SkG) & SkH).sum()) // 2
    prec = TP / (TP + FP) if (TP+FP)>0 else 0.0
    rec  = TP / (TP + FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    shd  = int((SkG ^ SkH).sum()) // 2
    return {"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1,"SHD_skeleton":shd,"edges":int(SkG.sum())//2}

def main():
    ap = argparse.ArgumentParser(description="J-stability evaluation for DCDI across multiple runs")
    ap.add_argument("--glob", required=True, help="Glob for W.csv paths (e.g. 'runs/run_*/W.csv')")
    ap.add_argument("--gt", required=True, help="Ground-truth CSV (adjacency with headers or edge list src,dst)")
    ap.add_argument("--mode", choices=["directed","skeleton"], default="directed")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--auto", choices=["p90","p95"], default=None)
    ap.add_argument("--topk-edges", type=int, default=None)
    ap.add_argument("--topk-per-node", type=int, default=None)
    ap.add_argument("--pi-list", default="0.5,0.7,0.9", help="comma-separated stability thresholds")
    ap.add_argument("--out", required=True, help="Output directory for frequency and summaries")
    args = ap.parse_args()

    paths = sorted([Path(p) for p in glob.glob(args.glob)])
    if not paths:
        raise SystemExit(f"No files match: {args.glob}")

    names0, M0 = load_W(paths[0])
    runs = []
    for p in paths:
        n, M = load_W(p)
        if n != names0:
            raise SystemExit(f"Name mismatch between {paths[0]} and {p}")
        runs.append(M)

    names_gt, Agt = load_gt(Path(args.gt))
    order = {n:i for i,n in enumerate(names_gt)}
    try:
        idx = [order[n] for n in names0]
    except KeyError:
        raise SystemExit("GT names must match W.csv columns; use your aligned GT or subset")
    Agt = Agt[np.ix_(idx, idx)]

    # select thresholding mode
    mode = None; kwargs = {}
    if args.tau is not None:
        mode = "tau"; kwargs["tau"] = args.tau
    elif args.auto is not None:
        mode = "auto"; kwargs["auto"] = args.auto
    elif args.topk_edges is not None:
        mode = "topk_edges"; kwargs["topk_edges"] = args.topk_edges
    elif args.topk_per_node is not None:
        mode = "topk_per_node"; kwargs["topk_per_node"] = args.topk_per_node
    else:
        mode = "auto"; kwargs["auto"] = "p95"

    # binarize each run
    Gs = []
    for M in runs:
        G = threshold_matrix(M, mode, **kwargs)
        Gs.append(G.astype(int))

    B = len(Gs); p = M0.shape[0]
    F = np.zeros((p,p), dtype=float)
    for G in Gs:
        F += G
    F /= float(B)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(F, index=names0, columns=names0).to_csv(outdir/"F_freq.csv")

    # evaluate across pi thresholds
    pi_vals = [float(x) for x in args.pi_list.split(",") if x.strip()]
    rows = []
    for pi in pi_vals:
        Gpi = (F >= pi).astype(int); np.fill_diagonal(Gpi, 0)
        if args.mode == "directed":
            m = directed_metrics(Gpi, Agt)
        else:
            m = skeleton_metrics(Gpi, Agt)
        m["pi"] = pi; rows.append(m)
        pd.DataFrame(Gpi, index=names0, columns=names0).to_csv(outdir/f"G_pi_{pi:.2f}.csv")
    df = pd.DataFrame(rows).sort_values("pi")
    df.to_csv(outdir/"jstable_summary.csv", index=False)
    print(df.sort_values("f1", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
