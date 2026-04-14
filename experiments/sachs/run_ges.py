#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ges.py
-----------
Run GES on a CSV (pooled or per-regime) and save adjacency matrices.
- Tries CausalLearn first; falls back to pgmpy if available.
- Exports a square CSV with row/column names (variable order preserved).

Examples
--------
# 1) Vanilla pooled
python run_ges.py --data synth.csv --out A_pooled.csv

# 2) Per-regime (env column)
python run_ges.py --data synth.csv --env-col env --per-env --outdir ./per_env

# 3) Per-regime with standardization and BIC
python run_ges.py --data synth.csv --env-col env --per-env --standardize --score bic --outdir ./per_env
"""
import argparse, sys, json
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

import os, json, time
import joblib

from sklearn.preprocessing import StandardScaler

# --- lightweight local scorer (fallback if eval_adj isn't available) ---
def score_adj(A_pred, A_true, undirected=False):
    import numpy as _np

    P = _np.asarray(A_pred).astype(int)
    T = _np.asarray(A_true).astype(int)

    # binarize and (optionally) symmetrize
    P = (P > 0).astype(int)
    T = (T > 0).astype(int)
    if undirected:
        P = ((P + P.T) > 0).astype(int)
        T = ((T + T.T) > 0).astype(int)

    # ignore self-loops
    _np.fill_diagonal(P, 0)
    _np.fill_diagonal(T, 0)

    TP = int(_np.sum((P == 1) & (T == 1)))
    FP = int(_np.sum((P == 1) & (T == 0)))
    FN = int(_np.sum((P == 0) & (T == 1)))
    TN = int(_np.sum((P == 0) & (T == 0)))

    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Simple SHD proxy: additions + deletions (treat reversals as 1)
    shd = FP + FN

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": prec, "recall": rec, "f1": f1, "shd": shd
    }

def score_adj(A_pred, A_true, undirected=False):
    A_pred = (np.asarray(A_pred) > 0).astype(int)
    A_true = (np.asarray(A_true) > 0).astype(int)

    if undirected:
        A_pred = ((A_pred + A_pred.T) > 0).astype(int)
        A_true = ((A_true + A_true.T) > 0).astype(int)
        iu = np.triu_indices_from(A_true, k=1)
        y_true, y_pred = A_true[iu], A_pred[iu]
    else:
        y_true, y_pred = A_true.ravel(), A_pred.ravel()

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Simple SHD = FP + FN (matches our eval usage for adjacency-level scoring)
    shd = FP + FN

    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=precision, recall=recall, f1=f1, shd=shd)


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd


def _extract_adj_from_causallearn(g) -> np.ndarray:
    """Heuristic, version-robust extraction of adjacency from a CausalLearn GES graph."""
    import numpy as np
    # If a tuple: (graph, score)
    if isinstance(g, tuple):
        g = g[0]

    # 1) Common attribute: .graph (numpy array)
    mat = getattr(g, "graph", None)
    if isinstance(mat, np.ndarray):
        # Nonzero entries indicate a connection; orientation is encoded by asymmetry.
        return (mat != 0).astype(int)

    # 2) Nested .G.graph
    GG = getattr(g, "G", None)
    if GG is not None:
        mat2 = getattr(GG, "graph", None)
        if isinstance(mat2, np.ndarray):
            return (mat2 != 0).astype(int)
        # Try to iterate edges if available
        try:
            nodes = GG.get_nodes()
            n = len(nodes)
            amat = np.zeros((n, n), dtype=int)
            # get_graph_edges may exist; if not, this will raise
            edges = GG.get_graph_edges()
            for e in edges:
                u = getattr(e, "node1", None)
                v = getattr(e, "node2", None)
                ep1 = str(getattr(e, "endpoint1", ""))
                ep2 = str(getattr(e, "endpoint2", ""))
                i = nodes.index(u)
                j = nodes.index(v)
                # TAIL -> ARROW means u -> v (heuristic for DAGs)
                if ep1.endswith("TAIL") and ep2.endswith("ARROW"):
                    amat[i, j] = 1
                elif ep2.endswith("TAIL") and ep1.endswith("ARROW"):
                    amat[j, i] = 1
                else:
                    # undirected/circle marks; record both
                    amat[i, j] = 1
                    amat[j, i] = 1
            return amat
        except Exception:
            pass

    # 3) GraphUtils helper if present
    try:
        from causallearn.utils.GraphUtils import GraphUtils
        amat = GraphUtils.to_amat(g)
        if isinstance(amat, np.ndarray):
            return (amat != 0).astype(int)
    except Exception:
        pass

    raise RuntimeError("Could not extract adjacency from CausalLearn GES graph (version mismatch?)")


def _run_ges_causallearn(X, varnames, score='bic', standardize=True, **_):
    """
    Run GES via CausalLearn and return a (d x d) 0/1 adjacency (np.ndarray).
    Robust to CausalLearn versions that return:
      - an object with .G or .graph
      - a (G, score) tuple
      - a dict containing 'A'/'adjacency_matrix' or 'G'/'graph'/... etc.
    """
    import numpy as np
    import causallearn
    from causallearn.search.ScoreBased.GES import ges

    print(f"[ges|causallearn] version={getattr(causallearn, '__version__', '?')}", flush=True)

    # Optional z-score standardization
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Map our 'score' string to CausalLearn's score_func name
    score_func = 'local_score_BIC'
    if isinstance(score, str) and score.lower() in ('bdeu', 'bdeu_score', 'bdeu-score'):
        score_func = 'local_score_BDeu'

    # Call CausalLearn
    res = ges(X, score_func=score_func)

    # ---- Normalize to either adjacency A or graph G ----
    A = None
    G = None

    def _as_adjacency(Alike):
        Aarr = np.asarray(Alike)
        return (Aarr != 0).astype(int)

    # 1) Handle dict returns
    if isinstance(res, dict):
        print(f"[ges|causallearn] dict keys={list(res.keys())}", flush=True)

        # direct adjacency present?
        for k in ('adjacency_matrix', 'adjacency', 'adj_mat', 'amat', 'A', 'W'):
            if k in res and res[k] is not None:
                try:
                    return _as_adjacency(res[k])
                except Exception:
                    pass

        # graph-like entry present?
        for k in ('G', 'graph', 'dag', 'pdag', 'cpdag', 'cg', 'result', 'estimated_graph'):
            if k in res and res[k] is not None:
                G = res[k]
                break

        # last resort: if edges/nodes are provided explicitly
        if G is None and 'nodes' in res and 'edges' in res:
            names = list(res['nodes'])
            name_to_idx = {str(n): i for i, n in enumerate(names)}
            d = len(names)
            A = np.zeros((d, d), dtype=int)
            for e in res['edges']:
                # accept (u,v) or (u,v,orient)
                u, v = e[0], e[1]
                iu, iv = name_to_idx[str(u)], name_to_idx[str(v)]
                # orientation unknown -> conservative
                A[iu, iv] = 1
                A[iv, iu] = 1
            return A

    # 2) Handle tuple returns (G, score)
    if isinstance(res, tuple) and len(res) >= 1:
        G = res[0]

    # 3) Handle object returns
    if G is None:
        # sometimes res is already the graph
        G = getattr(res, 'G', None) or getattr(res, 'graph', None) or res

    # If we have adjacency already, return it
    if isinstance(G, np.ndarray):
        return _as_adjacency(G)

    # Try GraphUtils.to_amat
    if G is not None:
        try:
            from causallearn.utils.GraphUtils import GraphUtils
            A = GraphUtils.to_amat(G)
            return _as_adjacency(A)
        except Exception:
            pass

        # Try dense matrix on G.graph
        try:
            Ggraph = getattr(G, 'graph', None)
            if Ggraph is not None:
                if hasattr(Ggraph, 'toarray'):
                    A = Ggraph.toarray()
                else:
                    A = np.asarray(Ggraph)
                return _as_adjacency(A)
        except Exception:
            pass

        # Rebuild from node/edge API
        try:
            # collect node names (ordered)
            if hasattr(G, 'nodes'):
                nodes = list(G.nodes)
            elif hasattr(G, 'get_nodes'):
                nodes = list(G.get_nodes())
            else:
                nodes = []

            if not nodes:
                raise RuntimeError("Empty node set in G; cannot build adjacency.")

            def _nm(n):
                nm = getattr(n, 'name', None)
                if nm is None and hasattr(n, 'get_name'):
                    nm = n.get_name()
                return str(nm if nm is not None else n)

            names = [_nm(n) for n in nodes]
            idx = {name: i for i, name in enumerate(names)}
            d = len(names)
            A = np.zeros((d, d), dtype=int)

            # edges
            if hasattr(G, 'get_graph_edges'):
                edges = G.get_graph_edges()
            elif hasattr(G, 'edges'):
                edges = G.edges
            else:
                edges = []

            for e in edges:
                n1 = getattr(e, 'node1', None) or (hasattr(e, 'get_node1') and e.get_node1())
                n2 = getattr(e, 'node2', None) or (hasattr(e, 'get_node2') and e.get_node2())
                i, j = idx[_nm(n1)], idx[_nm(n2)]

                ep1 = getattr(e, 'endpoint1', None) or (hasattr(e, 'get_endpoint1') and e.get_endpoint1())
                ep2 = getattr(e, 'endpoint2', None) or (hasattr(e, 'get_endpoint2') and e.get_endpoint2())
                ep1n = (str(getattr(ep1, 'name', '')).upper() if ep1 is not None else '')
                ep2n = (str(getattr(ep2, 'name', '')).upper() if ep2 is not None else '')

                if 'TAIL' in ep1n and 'ARROW' in ep2n:      # i -> j
                    A[i, j] = 1
                elif 'TAIL' in ep2n and 'ARROW' in ep1n:    # j -> i
                    A[j, i] = 1
                else:
                    # undirected/unknown; be conservative
                    A[i, j] = 1
                    A[j, i] = 1

            return A
        except Exception as e:
            raise RuntimeError(
                "Failed to extract adjacency from CausalLearn result.\n"
                f"type(G)={type(G)}, has_graph={hasattr(G,'graph')}, "
                f"has_nodes={hasattr(G,'nodes') or hasattr(G,'get_nodes')}, "
                f"error={e!r}"
            )

    # If we get here, nothing matched
    raise RuntimeError(
        "CausalLearn GES returned an unrecognized structure and no graph/adjacency was extractable."
    )


def run_once(df, varnames, use_causallearn=True, score='bic', standardize=True, **_):
    X = df[varnames].to_numpy(dtype=float)
    if use_causallearn:
        return _run_ges_causallearn(X, varnames, score=score, standardize=standardize)
    else:
        return _run_ges_pgmpy(X, varnames, score=score)   # only if you keep the pgmpy fallback    


def save_adj_csv(A: np.ndarray, varnames, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfA = pd.DataFrame(A, index=varnames, columns=varnames)
    dfA.to_csv(out_path)
    print(f"[ok] wrote adjacency: {out_path}")

def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    sc = StandardScaler()
    Xs = sc.fit_transform(df.values)
    return pd.DataFrame(Xs, index=df.index, columns=df.columns)

def _align_to(names_ref, A, names):
    """Reindex A to names_ref order (fill missing with 0)."""
    dfA = pd.DataFrame(np.asarray(A, dtype=float), index=names, columns=names)
    dfA = dfA.reindex(index=names_ref, columns=names_ref).fillna(0.0)
    return dfA.values

def _run_one_env_ges(e, sub_df, *, env_col, score, standardize, use_pgmpy, outdir):

    from pathlib import Path

    def _run_one_env_ges(e, sub, env_col, score, standardize, use_pgmpy, outdir):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)   # safe in parallel (exist_ok=True)
        varnames = [c for c in sub.columns if c != env_col]
        A = run_once(sub, varnames, use_causallearn=not use_pgmpy,
                     score=score, standardize=standardize, verbose=False)
        out_csv = outdir / f"A_env_{e}.csv"
        pd.DataFrame(A, index=varnames, columns=varnames).to_csv(out_csv)
        return {"env": str(e), "A": A, "names": varnames, "path": str(out_csv)}
    """Run your existing run_once() on a single environment subset and write A_env_<e>.csv."""
    t0 = time.time()

    # keep only numeric variables; drop env label column if present
    sub = sub_df.drop(columns=[env_col], errors="ignore")
    sub = sub.select_dtypes(include=[np.number]).copy()
    varnames = list(sub.columns)

    if standardize:
        sub = _standardize_df(sub)

    # call your existing runner; it should return (A, names) or just A
    res = run_once(sub, varnames,
                   use_causallearn=not use_pgmpy,
                   score=score,
                   standardize=False)  # we've already standardized above if requested

    if isinstance(res, tuple):
        A, names = res
    else:
        A, names = res, varnames

    A = np.asarray(A, dtype=float)
    out_csv = os.path.join(outdir, f"A_env_{e}.csv")
    pd.DataFrame(A, index=names, columns=names).to_csv(out_csv)

    dt = time.time() - t0
    print(f"[ok|GES] env={e} wrote {out_csv}  (t={dt:.2f}s)")
    return {"env": str(e), "A": A, "names": names, "path": out_csv, "sec": dt}


def main():
    import argparse, pathlib, json, os


    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--env-col", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", default="bic")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--k-allow", type=int, default=None)
    ap.add_argument("--true", default=None)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--use-pgmpy", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # load
    df = pd.read_csv(args.data)
    print(f"[debug|GES] env_col={args.env_col} present_in_df={args.env_col in df.columns}")

    # ---------------- PER-ENV BRANCH ----------------
    if args.env_col:                            # <— begin per-env branch
        outdir = Path(args.outdir) if args.outdir else Path("results_ges_per_env")
        outdir.mkdir(parents=True, exist_ok=True)

        groups = list(df.groupby(args.env_col))
        print(f"[per-env|GES] {len(groups)} envs via '{args.env_col}' "
              f"(n_jobs={args.n_jobs}, score={args.score}, standardize={args.standardize})")

        results = joblib.Parallel(n_jobs=args.n_jobs, backend="loky")(
            joblib.delayed(_run_one_env_ges)(
                e, sub, env_col=args.env_col, score=args.score,
                standardize=args.standardize, use_pgmpy=args.use_pgmpy, outdir=str(outdir)
            )
            for e, sub in groups
        )
        if len(results) == 0:
            print("[warn] no per-env results produced; skipping aggregation.")
            return

        # align, stack, write union/intersection/support, optional k-allow, optional scoring...
        names_ref = results[0]["names"]
        A_stack = np.stack([_align_to(names_ref, r["A"], r["names"]) for r in results], axis=0)
        E, p, _ = A_stack.shape
        support = (A_stack > 0).sum(axis=0)
        A_union = (support > 0).astype(float)
        A_inter = (support == E).astype(float)

        pd.DataFrame(support, index=names_ref, columns=names_ref).to_csv(outdir / "support_counts.csv")
        pd.DataFrame(A_union, index=names_ref, columns=names_ref).to_csv(outdir / "A_union.csv")
        pd.DataFrame(A_inter, index=names_ref, columns=names_ref).to_csv(outdir / "A_Jstable_ges.csv")
        print(f"[ok|GES] wrote A_union.csv, A_Jstable_ges.csv, support_counts.csv  (E={E}, p={p})")

        if args.k_allow is not None and 0 <= args.k_allow < E:
            thresh = E - args.k_allow
            A_k = (support >= thresh).astype(float)
            pd.DataFrame(A_k, index=names_ref, columns=names_ref).to_csv(outdir / f"A_Jstable_k{args.k_allow}.csv")
            print(f"[ok|GES] wrote A_Jstable_k{args.k_allow}.csv (threshold={thresh})")

        if args.true:
            try:
                A_true = (pd.read_csv(args.true, index_col=0)
                            .reindex(index=names_ref, columns=names_ref)
                            .fillna(0).values.astype(int))
                rep_inter = {"score": args.score, "n_envs": E, "p": p, "names": names_ref}
                rep_inter.update(score_adj(A_inter, A_true, undirected=args.undirected))
                (outdir / "report_intersection.json").write_text(json.dumps(rep_inter, indent=2))
                print("[ok|GES] wrote report_intersection.json")
                if args.k_allow is not None and 0 <= args.k_allow < E:
                    rep_k = {"score": args.score, "n_envs": E, "p": p, "names": names_ref,
                             "k_allow": args.k_allow, "threshold": E-args.k_allow}
                    rep_k.update(score_adj(A_k, A_true, undirected=args.undirected))
                    (outdir / f"report_k{args.k_allow}.json").write_text(json.dumps(rep_k, indent=2))
                    print(f"[ok|GES] wrote report_k{args.k_allow}.json")
            except Exception as e:
                print(f"[warn|GES] per-env scoring failed: {e}")

        return                                  # <— IMPORTANT: return stays INSIDE per-env branch ONLY

    # ---------------- POOLED BRANCH -----------------
    print("[route|GES] pooled")                 # <— add this so you see it fire
    if args.out is None:
        raise ValueError("Please specify --out for pooled/single-env run.")

    # numeric columns only (drops any stray string cols like 'env')
    all_cols = [c for c in df.columns if c != args.env_col]
    varnames = [c for c in all_cols if np.issubdtype(df[c].dtype, np.number)]
    dropped = sorted(set(all_cols) - set(varnames))
    if dropped:
        print(f"[pooled] dropping non-numeric columns: {dropped}")

    print(f"[GES|pooled] n={len(df)} score={args.score} standardize={args.standardize}")
    A = run_once(df, varnames,
                 use_causallearn=not args.use_pgmpy,
                 score=args.score, standardize=args.standardize,
                 verbose=args.verbose)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_adj_csv(A, varnames, out_path)
    print(f"[ok|GES] wrote pooled adjacency to {out_path}")

    if args.true:
        try:
            A_true = (pd.read_csv(args.true, index_col=0)
                        .reindex(index=varnames, columns=varnames)
                        .fillna(0).values.astype(int))
            rep = {"score": args.score, "n_envs": 1, "p": len(varnames), "names": varnames}
            rep.update(score_adj(A, A_true, undirected=args.undirected))
            (out_path.parent / "report_pooled.json").write_text(json.dumps(rep, indent=2))
            print("[ok|GES] wrote report_pooled.json")
        except Exception as e:
            print(f"[warn|GES] pooled scoring failed: {e}")


if __name__ == "__main__":
    main()

