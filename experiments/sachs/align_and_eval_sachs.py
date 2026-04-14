#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd

CANON_ORDER = ["Raf","Mek","Plcg","PIP2","PIP3","Erk","Akt","PKA","PKC","P38","Jnk"]
ALIASES = {k.lower():v for v in CANON_ORDER for k,v in [(v,v)]}  # simple identity alias

def norm(s): return re.sub(r"[^a-z0-9]+","",str(s).strip().lower())

def load_sq(path):
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"{path} not square: {df.shape}")
    if df.index.tolist() != df.columns.tolist():
        labs = sorted(set(df.index) | set(df.columns))
        df = df.reindex(index=labs, columns=labs).fillna(0)
    return df

def map_x_to_names(pred_cols, data_csv):
    names = [c for c in pd.read_csv(data_csv).columns if c.lower()!="env"]
    if len(names) != len(pred_cols):
        if len(CANON_ORDER)==len(pred_cols): return CANON_ORDER
        raise ValueError(f"Data columns ({len(names)}) != pred size ({len(pred_cols)})")
    return names

def rename_like_true(dfP, dfT, data_csv=None):
    pred_cols, true_cols = list(dfP.columns), list(dfT.columns)
    # case: X1..Xn
    if all(re.fullmatch(r"[Xx]\d+", str(c)) for c in pred_cols):
        if not data_csv: raise ValueError("Pred uses X1..Xn but --data not given")
        names = map_x_to_names(pred_cols, data_csv)
        dfP = dfP.rename(index=dict(zip(pred_cols,names)), columns=dict(zip(pred_cols,names)))
    # align order to truth
    return dfP.reindex(index=true_cols, columns=true_cols).fillna(0)

def binarize(df, tau=0.0, undirected=False):
    A = df.to_numpy(float)
    if undirected: A = ((A>tau)|(A.T>tau)).astype(int)
    else:
        A = (A>tau).astype(int); np.fill_diagonal(A,0)
    return A

def metrics(Ap, At):
    P,T = Ap.astype(bool), At.astype(bool)
    tp = int((P & T).sum()); fp = int((P & ~T).sum())
    fn = int((~P & T).sum()); tn = int((~P & ~T).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    Ud = lambda A: ((A>0)|(A.T>0)).astype(int)
    shd = int(np.abs(Ud(Ap)-Ud(At)).sum()//2)
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=prec,recall=rec,f1=f1,shd=shd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", required=True)
    ap.add_argument(
        "--pred",
        required=True,
        help=(
            "Square predicted adjacency CSV. Typical examples are "
            "artifacts/sachs/generated/pooled/fci_envpooled.csv or "
            "artifacts/sachs/generated/per_env/A_Jstable_fci.csv"
        ),
    )
    ap.add_argument(
        "--data",
        required=True,
        help="sachs_with_env.csv used to map X1..Xn column names back to protein names if needed",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--tau", type=float, default=0.0)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    dfT = load_sq(args.true)
    dfP = load_sq(args.pred)
    dfPa = rename_like_true(dfP, dfT, data_csv=args.data)

    (outdir/"pred_aligned.csv").write_text(dfPa.to_csv())
    (outdir/"debug_labels.json").write_text(json.dumps({
        "true": list(dfT.columns),
        "pred_aligned": list(dfPa.columns),
        "overlap": sorted(set(dfT.columns)&set(dfPa.columns))
    }, indent=2))

    Ap = binarize(dfPa, tau=args.tau, undirected=args.undirected)
    At = binarize(dfT, tau=args.tau, undirected=args.undirected)
    rep = metrics(Ap, At); rep["nvars"] = len(dfT)
    (outdir/"report.json").write_text(json.dumps(rep, indent=2))
    print("[ok] wrote", outdir, rep)

if __name__=="__main__":
    main()
