#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

# ---- optional deps; fail gracefully if missing ----
try:
    from causallearn.search.ConstraintBased.FCI import fci as cl_fci
    from causallearn.utils.cit import fisherz
except Exception as e:
    cl_fci = None
    fisherz = None

def _prep_for_scoring(A, A_true, undirected=False, keep_diag=False):
    import numpy as np
    A = np.asarray(A)
    A_true = np.asarray(A_true)
    if undirected:
        A = ((A > 0) | (A.T > 0)).astype(int)
        A_true = ((A_true > 0) | (A_true.T > 0)).astype(int)
    else:
        A = (A > 0).astype(int)
        A_true = (A_true > 0).astype(int)
    if not keep_diag:
        np.fill_diagonal(A, 0)
        np.fill_diagonal(A_true, 0)
    return A, A_true


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _adjacency_from_cl_graph(G, varnames: List[str]) -> np.ndarray:
    """
    Extract an undirected adjacency (skeleton) from a CausalLearn PAG/Graph-like object.
    Fall back to parsing .graph dict if needed.
    """
    d = len(varnames)
    A = np.zeros((d, d), dtype=float)

    if G is None:
        return A

    # Try API: is_adjacent_to with Node objects
    try:
        nodes = getattr(G, "nodes", None) or getattr(G, "get_nodes", lambda: None)()
        # If Node objects exist and lengths match, build a mapping name->index
        if nodes is not None and len(nodes) == d:
            # nodes may be Node objects with .get_name()
            def node_name(n):
                return getattr(n, "get_name", lambda: str(n))()

            name2idx = {name: i for i, name in enumerate(varnames)}
            idx_of = []
            ok = True
            for n in nodes:
                nm = node_name(n)
                if nm not in name2idx:
                    ok = False
                    break
                idx_of.append(name2idx[nm])

            if ok and hasattr(G, "is_adjacent_to"):
                for ii, ni in enumerate(nodes):
                    for jj, nj in enumerate(nodes):
                        if jj <= ii: 
                            continue
                        try:
                            if G.is_adjacent_to(ni, nj):
                                i = idx_of[ii]; j = idx_of[jj]
                                A[i, j] = A[j, i] = 1.0
                        except Exception:
                            pass
                return A
    except Exception:
        pass

    # Fallback: parse G.graph as nested dict(Node -> dict(Node -> Edge))
    try:
        graph_attr = getattr(G, "graph", None)
        if isinstance(graph_attr, dict):
            # Build name->index using keys' names when possible
            def node_name(n):
                return getattr(n, "get_name", lambda: str(n))()

            name2idx = {name: i for i, name in enumerate(varnames)}

            for ni, nbrs in graph_attr.items():
                try:
                    i = name2idx[node_name(ni)]
                except Exception:
                    continue
                if not isinstance(nbrs, dict):
                    continue
                for nj, edge in nbrs.items():
                    if edge is None:
                        continue
                    try:
                        j = name2idx[node_name(nj)]
                    except Exception:
                        continue
                    if i != j:
                        A[i, j] = 1.0
                        A[j, i] = 1.0
            return A
    except Exception:
        pass

    # Last resort: return zeros (caller can warn)
    return A

# --- robust extractor for causallearn FCI output -> (A, names) ---
def _extract_pag_adjacency_cl(fci_res, names_hint=None):
    """
    Accepts various causallearn FCI return shapes and builds an undirected
    skeleton adjacency A (d x d) with node names.

    fci_res can be:
      - an object with attribute .G (CausalGraph-like)
      - a dict with key 'G'
      - a tuple whose first element is a graph-like object
      - (rare) already a graph-like object with .get_nodes()
    """
    import numpy as np

    # 1) locate a graph-like 'G'
    G = None
    if hasattr(fci_res, "G"):
        G = fci_res.G
    elif isinstance(fci_res, dict) and "G" in fci_res:
        G = fci_res["G"]
    elif isinstance(fci_res, tuple) and len(fci_res) > 0:
        first = fci_res[0]
        if hasattr(first, "G"):
            G = first.G
        else:
            G = first
    else:
        G = fci_res  # hope it's graph-like

    # 2) get node names
    names = None
    try:
        nodes_obj = G.get_nodes()
        def _n2name(n):
            if hasattr(n, "get_name"):
                return str(n.get_name())
            return str(n)
        names = [_n2name(n) for n in nodes_obj]
    except Exception:
        # try networkx-like
        try:
            names = list(G.nodes())
        except Exception:
            # fall back to hint
            if names_hint is not None:
                names = list(names_hint)
            else:
                raise RuntimeError("Could not extract node names from FCI graph.")
    d = len(names)
    idx = {n:i for i,n in enumerate(names)}
    A = np.zeros((d, d), dtype=float)

    # 3) fill skeleton (adjacency, ignoring edge marks)
    filled = False
    try:
        nodes_obj = G.get_nodes()
        def _n2name(n):
            if hasattr(n, "get_name"):
                return str(n.get_name())
            return str(n)
        for ni in nodes_obj:
            i = idx[_n2name(ni)]
            try:
                adj = G.get_adjacent_nodes(ni)
            except Exception:
                adj = []
            for nj in adj:
                j = idx[_n2name(nj)]
                A[i, j] = 1.0
                A[j, i] = 1.0
        filled = True
    except Exception:
        pass

    if not filled:
        # last-ditch: try edges-like container
        try:
            edges = getattr(G, "edges", None)
            if edges is None and hasattr(G, "get_graph"):
                g2 = G.get_graph()
                edges = getattr(g2, "edges", None)
            if edges is None:
                raise RuntimeError("No edges iterable on graph.")
            for e in edges:
                # accommodate different edge object types
                u, v = None, None
                if hasattr(e, "get_node1") and hasattr(e, "get_node2"):
                    u = str(e.get_node1().get_name()) if hasattr(e.get_node1(), "get_name") else str(e.get_node1())
                    v = str(e.get_node2().get_name()) if hasattr(e.get_node2(), "get_name") else str(e.get_node2())
                elif isinstance(e, (tuple, list)) and len(e) >= 2:
                    u, v = str(e[0]), str(e[1])
                if u in idx and v in idx:
                    i, j = idx[u], idx[v]
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        except Exception as ex:
            raise RuntimeError(f"Could not extract adjacency from causallearn graph: {ex}")

    np.fill_diagonal(A, 0.0)
    return A, names


def fci_skeleton(df, alpha=0.01, depth=0, standardize=False):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    if isinstance(df, np.ndarray):
        X = df
        names = [f"X{i}" for i in range(X.shape[1])]
    else:
        names = list(df.columns)
        X = df.to_numpy(dtype=float)

    if standardize:
        X = StandardScaler().fit_transform(X)

    res = fci(X, fisherz, alpha, verbose=False, depth=depth)
    A, names = _extract_pag_adjacency_cl(res, names_hint=names)
    return A, names


def score(A_pred: np.ndarray, A_true: np.ndarray) -> dict:
    """
    Compare undirected skeletons with entries in {0,1}. Return basic metrics.
    """
    pred = (A_pred > 0).astype(int)
    true = (A_true > 0).astype(int)

    tp = int((pred & true).sum())
    fp = int((pred & (1 - true)).sum())
    fn = int(((1 - pred) & true).sum())
    tn = int(((1 - pred) & (1 - true)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    shd = fp + fn  # undirected structural hamming distance

    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=round(precision, 3),
                recall=round(recall, 3),
                f1=round(f1, 3),
                shd=int(shd))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV with columns [vars..., optional env]")
    p.add_argument("--env-col", default=None, help="Name of environment/regime column (per-env mode if set)")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--true", default=None, help="CSV adjacency for scoring (optional)")
    p.add_argument("--outdir", required=True, help="Where to write outputs")
    # add alongside your other parser.add_argument(...) calls
    p.add_argument(
        "--undirected",
        action="store_true",
    help="Treat predicted/true adjacencies as undirected when scoring."
    )
    
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"[load] n={len(df)} d={len(df.columns)} env_col={args.env_col!r}")

    # ----- PER-ENV MODE -----
    if args.env_col and args.env_col in df.columns:
        per_env_As = []
        env_ids = []
        names_ref: List[str] = []

        for e, sub in df.groupby(args.env_col):
            print(f"[FCI] env={e} n={len(sub)} alpha={args.alpha}")
            sub = sub.drop(columns=[args.env_col])
            varnames = list(sub.columns)

            A, names = fci_skeleton(
                sub[varnames],
                alpha=args.alpha,
                depth=args.depth,
                standardize=args.standardize,
            )
            if not names_ref:
                names_ref = list(names)
            elif list(names) != list(names_ref):
                raise ValueError(f"[error] variable name mismatch in env={e}: {names} != {names_ref}")

            # write per-env CSV
            out_csv = outdir / f"fci_env{e}.csv"
            pd.DataFrame(A, index=names, columns=names).to_csv(out_csv)
            print(f"[ok] wrote {out_csv}")

            per_env_As.append(A)
            env_ids.append(str(e))

        if len(per_env_As) == 0:
            print("[warn] no per-env results produced; skipping aggregation.")
            return

        # aggregate: intersection / union / support
        A_stack = np.stack(per_env_As, axis=0)  # (E, d, d)
        A_inter = (A_stack > 0).all(axis=0).astype(float)
        A_union = (A_stack > 0).any(axis=0).astype(float)
        support = (A_stack > 0).sum(axis=0)

        np.fill_diagonal(A_inter, 0.0)
        np.fill_diagonal(A_union, 0.0)
        np.fill_diagonal(support, 0)

        out_inter = outdir / "A_Jstable_fci.csv"
        out_union = outdir / "A_union.csv"
        out_support = outdir / "support_counts.csv"
        pd.DataFrame(A_inter, index=names_ref, columns=names_ref).to_csv(out_inter)
        pd.DataFrame(A_union, index=names_ref, columns=names_ref).to_csv(out_union)
        pd.DataFrame(support, index=names_ref, columns=names_ref).to_csv(out_support)
        print(f"[ok] wrote {out_inter}")
        print(f"[ok] wrote {out_union}")
        print(f"[ok] wrote {out_support}")

        # Optional scoring if ground truth provided
        if args.true:
            A_true = pd.read_csv(args.true, index_col=0).reindex(index=names_ref, columns=names_ref).fillna(0).to_numpy()
            rep = {"alpha": args.alpha, "env_values": env_ids,
                   "intersection": score(A_inter, A_true),
                   "union": score(A_union, A_true)}
            (outdir / "report.json").write_text(json.dumps(rep, indent=2))
            print(f"[ok] wrote {outdir / 'report.json'}")
        return

    # ----- POOLED MODE -----
    # --- pooled PSI-FCI (no env_col) ---
    print(f"[FCI] pooled n={len(df)} alpha={args.alpha}")

    # keep numeric columns only (drops 'env' or any other non-numeric metadata)
    df_pooled = df.select_dtypes(include=[np.number]).copy()
    dropped = [c for c in df.columns if c not in df_pooled.columns]
    if dropped:
        print(f"[pooled] dropping non-numeric columns: {dropped}")

    A, names = fci_skeleton(
        df_pooled,
        alpha=args.alpha,
        depth=args.depth,
        standardize=args.standardize,
    )

    pooled_csv = outdir / "fci_envpooled.csv"
    pd.DataFrame(A, index=names, columns=names).to_csv(pooled_csv)
    print(f"[ok] wrote {pooled_csv}")

    # optional scoring
    if args.true:
        A_true = pd.read_csv(args.true, index_col=0).reindex(index=names, columns=names).fillna(0).values.astype(int)
        rep = {"mode": "pooled", "alpha": args.alpha}
        # NEW
        A_eval, Atrue_eval = _prep_for_scoring(A, A_true, undirected=args.undirected)
        rep.update(score(A_eval, Atrue_eval))
#        rep.update(score(A, A_true, undirected=args.undirected))
        (outdir / "report.json").write_text(json.dumps(rep, indent=2))
        print(f"[ok] wrote {outdir/'report.json'}")

if __name__ == "__main__":
    main()
