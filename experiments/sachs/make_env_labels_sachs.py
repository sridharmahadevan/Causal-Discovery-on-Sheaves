#!/usr/bin/env python3
# Add an 'env' column to a wide CSV via clustering (for j-stable runs).
# Example:
#   python make_env_labels_sachs.py \
#     --in-csv data/sachs.csv \
#     --out-csv data/sachs_with_env.csv \
#     --auto-k --k-min 3 --k-max 6 \
#     --min-env-size 50 \
#     --standardize --random-state 0

import argparse, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def log(msg: str): print(msg, file=sys.stderr)

def pick_k_by_silhouette(X, k_min, k_max, random_state):
    best_k, best_score = None, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:  # degenerate
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k if best_k is not None else max(2, k_min)

def reassign_small_clusters(X, labels, centroids, min_size):
    labels = labels.copy()
    k = centroids.shape[0]
    counts = np.bincount(labels, minlength=k)
    small = np.where(counts < min_size)[0]
    if len(small) == 0:
        return labels
    non_small = np.where(counts >= min_size)[0]
    if len(non_small) == 0:
        return labels  # nothing to do
    # squared Euclidean distances to non-small centroids
    ns_centroids = centroids[non_small]
    dists = ((X[:, None, :] - ns_centroids[None, :, :]) ** 2).sum(axis=2)
    for s in small:
        idx = np.where(labels == s)[0]
        if idx.size == 0:
            continue
        nearest = np.argmin(dists[idx], axis=1)   # index into non_small
        labels[idx] = non_small[nearest]
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--k", type=int, default=None,
                    help="Use this many clusters (overrides --auto-k).")
    ap.add_argument("--auto-k", action="store_true",
                    help="Select K by silhouette in [--k-min, --k-max].")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--min-env-size", type=int, default=30,
                    help="Min samples per env; smaller clusters are reassigned.")
    ap.add_argument("--standardize", action="store_true",
                    help="Z-score numeric features before clustering.")
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    in_path  = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Keep only numeric columns for clustering; pass others through unchanged
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric columns found.")
    X = df[num_cols].to_numpy(dtype=float, copy=True)

    if args.standardize:
        X = StandardScaler().fit_transform(X)

    if args.k is not None and not args.auto_k:
        k = args.k
    else:
        k = pick_k_by_silhouette(X, args.k_min, args.k_max, args.random_state)

    log(f"[make_env] n={len(df)} p={X.shape[1]}  k={k}  standardize={bool(args.standardize)}")

    km = KMeans(n_clusters=k, random_state=args.random_state, n_init="auto")
    labels = km.fit_predict(X)
    labels = reassign_small_clusters(X, labels, km.cluster_centers_, args.min_env_size)

    uniq = sorted(np.unique(labels).tolist())
    remap = {old: i for i, old in enumerate(uniq)}
    env_str = [f"e{remap[int(z)]}" for z in labels]

    df_out = df.copy()
    df_out["env"] = env_str
    df_out.to_csv(out_path, index=False)

    counts = pd.Series(env_str).value_counts().sort_index()
    rep = {
        "in_csv": str(in_path),
        "out_csv": str(out_path),
        "n": int(len(df)),
        "p": int(X.shape[1]),
        "k": int(len(uniq)),
        "min_env_size": int(args.min_env_size),
        "standardize": bool(args.standardize),
        "counts": counts.to_dict(),
        "numeric_columns_used": num_cols,
    }
    (out_path.parent / (out_path.stem + ".env_counts.txt")).write_text(
        "env counts:\n" + "\n".join(f"{k}: {v}" for k, v in counts.items())
    )
    (out_path.parent / (out_path.stem + ".env_report.json")).write_text(json.dumps(rep, indent=2))

    log("[ok] wrote:")
    log(f"  - {out_path}")
    log(f"  - {(out_path.parent / (out_path.stem + '.env_counts.txt'))}")
    log(f"  - {(out_path.parent / (out_path.stem + '.env_report.json'))}")

if __name__ == "__main__":
    main()
