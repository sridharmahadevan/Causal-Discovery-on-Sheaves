#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust DCDI runner (patched v15)
- v14 + a tolerant metrics logger that writes metrics.csv (and an auto-scaled metrics_zoom.png)
"""

import argparse, importlib, importlib.util, json, os, re, sys, types, warnings, csv
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------- helpers ----------

def _argval(argv, name, cast=None, default=None):
    key = f"--{name}"
    try:
        i = argv.index(key)
    except ValueError:
        return default
    if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--"):
        v = argv[i + 1]
        return cast(v) if cast else v
    return True

def _argval_any(argv, keys, cast=int, default=None):
    for k in keys:
        v = _argval(argv, k, cast, None)
        if v is not None:
            return v
    return default

def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    s = str(x); parts = re.split(r"[,\s;]+", s.strip()); return [p for p in parts if p]


def _smart_import_dcdi_train():
    try: return importlib.import_module("dcdi.train")
    except Exception: pass
    try: return importlib.import_module("dcdi.dcdi.train")
    except Exception: pass
    for base in map(Path, sys.path):
        try:
            for c in (base/"dcdi"/"train.py", base/"dcdi"/"dcdi"/"train.py"):
                if c.exists():
                    spec = importlib.util.spec_from_file_location("dcdi_flex.train", str(c))
                    mod = importlib.util.module_from_spec(spec); assert spec and spec.loader
                    spec.loader.exec_module(mod)  # type: ignore
                    return mod
        except Exception: continue
    raise ModuleNotFoundError("Could not import dcdi.train")


class AutoOpt:
    def __init__(self, base=None, p=None, seed=123, outdir=".", device="cpu"):
        self._store = dict(base or {})
        self._p = p; self._seed = seed; self._outdir = str(outdir); self._device = device

    def __getattr__(self, name):
        if name in self._store: return self._store[name]
        val = self._default_for(name); self._store[name] = val
        print(f"[dcdi_run] opt.{name} defaulted to {val!r}")
        return val

    def __setattr__(self, name, value):
        if name.startswith("_"): return super().__setattr__(name, value)
        self._store[name] = value

    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): setattr(self, key, value)

    def _default_for(self, name):
        n = name.lower()
        if n in ("epochs","max_epochs"): return self._store.get("epochs", 200)
        if "patience" in n: return 50
        if n in ("val_every","eval_every"): return 10
        if n == "log_every": return 10
        if "checkpoint" in n: return max(50, int(self._store.get("epochs",200)//4))
        if n in ("batch","batch_size","train_batch_size","eval_batch_size"): return self._store.get("batch_size", 128)
        if "lr" in n: return float(self._store.get("lr", 1e-2))
        if n in ("hidden_dim","h_dim"): return 64
        if n in ("n_hidden","num_layers","layers"): return 2
        if n in ("nonlin","activation"): return "relu"
        if n.startswith("lambda_") or "weight_decay" in n: return 0.0
        if n == "optimizer": return "sgd"
        if n == "scheduler": return "none"
        if n in ("p","n_vars","num_vars","num_nodes"): return int(self._p) if self._p is not None else 0
        if n == "seed": return int(self._seed)
        if "device" in n: return self._device
        if n.endswith("dir") or n.endswith("path"): return self._outdir
        if n == "use_cuda": return False
        if n == "undirected": return bool(self._store.get("undirected", False))
        if n == "verbose": return True
        # safe non-zero defaults
        if n in ("stop_crit_win","stopcrit","stop_window"): return 100
        if n in ("snapshot_freq","snapshotfrequency","snapshotfrequency_steps"): return 100
        if n in ("plot_freq","plotfrequency"): return 100
        # AL defaults
        if n in ("mu_init","mu","mulagrange"): return 0.5
        if n in ("gamma_init","gamma0"): return 1e-2
        if n in ("omega_mu","omegamu"): return 1.0
        if n in ("omega_gamma","omegagamma"): return 1.0
        if n in ("h_threshold","hthresh","h_thresh"): return 1e-4
        return 0


class DCDISampler:
    def __init__(self, X, device="cpu", num_samples=None):
        self.device = torch.device(device)
        self.X = X.to(self.device) if hasattr(X, "to") else X
        self.N = int(self.X.shape[0]); self.p = int(self.X.shape[1])
        self.num_samples = int(max(1, min(int(num_samples) if num_samples is not None else self.N, self.N)))
        self._mask   = torch.zeros(self.N, self.p, dtype=torch.float32, device=self.device)
        self._regime = torch.zeros(self.N,           dtype=torch.long,   device=self.device)

    def sample(self, batch_size:int):
        B = int(max(1, min(batch_size, self.N)))
        idx = torch.randint(0, self.N, (B,), device=self.device)
        return self.X[idx], self._mask[idx], self._regime[idx]

    def __len__(self):
        return self.N

    def to(self, device):
        self.device = torch.device(device)
        self.X = self.X.to(self.device); self._mask = self._mask.to(self.device); self._regime = self._regime.to(self.device); return self
    def reset(self): return None


def _nonneg_adj_from_W(W):
    I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    return (W * W) * (1.0 - I)


def _build_dcdi_model(train_mod, p:int, opt):
    import inspect, types as _types
    device = getattr(opt, "device", "cpu")

    def _normalize(m):
        if not hasattr(m, "p"): m.p = p
        if not hasattr(m, "n_vars"): m.n_vars = p
        if not hasattr(m, "num_vars"): m.num_vars = p
        if not hasattr(m, "num_nodes"): m.num_nodes = p
        if hasattr(m, "W") and not hasattr(m, "get_W"): m.get_W = lambda: m.W
        if hasattr(m, "W") and not hasattr(m, "adjacency"): m.adjacency = m.W
        if not hasattr(m, "get_parameters"):
            def _gp(self, mode=None):
                ws, bs = [], []
                for _, param in self.named_parameters():
                    (bs if param.ndim == 1 else ws).append(param)
                return ws, bs, []
            m.get_parameters = _types.MethodType(_gp, m)
        if not hasattr(m, "compute_log_likelihood"):
            def _ll(self, x, *args, **kw):
                W = getattr(self, "W", None)
                if W is None and args and isinstance(args[0], (list, tuple)) and len(args[0])>0:
                    W = args[0][0]
                pred = x @ W; resid = x - pred
                return (-0.5 * (resid ** 2).sum(dim=1)).mean()
            m.compute_log_likelihood = _types.MethodType(_ll, m)
        if not hasattr(m, "get_w_adj"):
            def _gwa(self):
                W = getattr(self, "W", None) or getattr(self, "adjacency", None)
                return _nonneg_adj_from_W(W)
            m.get_w_adj = _types.MethodType(_gwa, m)
        if not hasattr(m, "get_w_adjs_log"):
            def _gwal(self):
                A = self.get_w_adj()
                return [A.detach().clone()] if hasattr(A, "detach") else [A]
            m.get_w_adjs_log = _types.MethodType(_gwal, m)
        if not hasattr(m, "get_grad_norm"):
            def _ggn(self, mode="wbx"):
                import math
                tot = 0.0
                for _, param in self.named_parameters():
                    if param.grad is None: continue
                    inc = (('w' in str(mode) and param.ndim > 1) or
                           ('b' in str(mode) and param.ndim == 1) or
                           ('x' in str(mode) and param.ndim not in (1,2)))
                    if inc:
                        val = torch.linalg.vector_norm(param.grad.detach()).item()
                        if math.isfinite(val): tot += val * val
                device0 = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
                return torch.tensor((tot ** 0.5) if tot > 0 else 0.0, device=device0)
            m.get_grad_norm = _types.MethodType(_ggn, m)
        try: m.to(device)
        except Exception: pass
        return m

    # Try classes
    for name in ("Model","DCDI","DAGLearner","Learner","DCDIModule","Net"):
        K = getattr(train_mod, name, None)
        if K is None: continue
        try:
            kwargs = {}
            if hasattr(K, "__init__"):
                iv = K.__init__.__code__.co_varnames
                for k in ("p","n_vars","num_vars","in_dim","input_dim","dim"):
                    if k in iv: kwargs[k] = p; break
                for k in ("hidden_dim","h_dim"):
                    if k in iv: kwargs[k] = getattr(opt, "hidden_dim", 64)
                for k in ("n_hidden","num_layers","layers"):
                    if k in iv: kwargs[k] = getattr(opt, "n_hidden", 2)
                if "nonlin" in iv: kwargs["nonlin"] = getattr(opt, "nonlin", "relu")
                if "opt" in iv: kwargs["opt"] = opt
            m = K(**kwargs) if kwargs else K()
            return _normalize(m)
        except Exception: pass

    # Try builder functions
    for build_name in ("build_model","make_model","create_model","init_model","get_model"):
        build_fn = getattr(train_mod, build_name, None)
        if build_fn is None: continue
        try: return _normalize(build_fn(p))
        except Exception:
            try: return _normalize(build_fn())
            except Exception: pass

    # Fallback minimal model
    class MinimalDCDI(nn.Module):
        def __init__(self, p):
            super().__init__()
            W = torch.randn(p, p) * 0.01
            W.fill_diagonal_(0.0)
            self.W = nn.Parameter(W)
            self.p=p; self.n_vars=p; self.num_vars=p; self.num_nodes=p
        def forward(self, x): return x @ self.W
        def get_W(self): return self.W
        @property
        def adjacency(self): return self.W
        def get_parameters(self, mode=None): return [self.W], [], []
        def compute_log_likelihood(self, x, *args, **kwargs):
            pred = x @ self.W; resid = x - pred
            return (-0.5 * (resid ** 2).sum(dim=1)).mean()
        def get_w_adj(self): return (self.W*self.W) * (1.0 - torch.eye(self.W.shape[0], device=self.W.device, dtype=self.W.dtype))
        def get_w_adjs_log(self): return [self.get_w_adj().detach().clone()]
        def get_grad_norm(self, mode="wbx"):
            import math
            tot=0.0
            for _, p in self.named_parameters():
                if p.grad is None: continue
                inc = (('w' in str(mode) and p.ndim > 1) or ('b' in str(mode) and p.ndim == 1) or ('x' in str(mode) and p.ndim not in (1,2)))
                if inc:
                    val = torch.linalg.vector_norm(p.grad.detach()).item()
                    if math.isfinite(val): tot += val*val
            return torch.tensor((tot**0.5) if tot>0 else 0.0, device=self.W.device)
    return _normalize(MinimalDCDI(p))


# ---------- writer ----------

def _save_graph_outputs(W_t, var_names, outdir: Path, tau: float):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    W = W_t.detach().cpu().numpy().astype(float) if hasattr(W_t,"detach") else np.asarray(W_t, dtype=float)
    p = W.shape[0]
    if not var_names or len(var_names)!=p: var_names=[f"x{idx}" for idx in range(p)]
    pd.DataFrame(W, index=var_names, columns=var_names).to_csv(outdir/"W.csv")
    G = (np.abs(W)>float(tau)).astype(int); np.fill_diagonal(G,0)
    pd.DataFrame(G, index=var_names, columns=var_names).to_csv(outdir/f"G_thresh_tau_{tau}.csv")
    edges=[(var_names[i],var_names[j],float(W[i,j])) for i in range(p) for j in range(p) if i!=j and abs(W[i,j])>float(tau)]
    pd.DataFrame(edges, columns=["src","dst","weight"]).to_csv(outdir/"edges.csv", index=False)
    (outdir/"summary.json").write_text(json.dumps({"p":p,"tau":float(tau),"num_edges_tau":int(G.sum()),"var_names":var_names,"files":{"W":"W.csv","G_tau":f"G_thresh_tau_{tau}.csv","edges":"edges.csv"}}, indent=2))


def _extract_W_from_model(model):
    if model is None: raise RuntimeError("Model is None; cannot extract W.")
    for name in ("W","W_est","A","adjacency","adj_matrix","B","weights","W_logits","A_logits"):
        if hasattr(model, name):
            W = getattr(model, name)
            if callable(W): W = W()
            if isinstance(W, np.ndarray): W = torch.from_numpy(W)
            if hasattr(W, "data") and not isinstance(W, torch.Tensor): W = W.data
            if hasattr(W,"ndim") and W.ndim==2 and W.shape[0]==W.shape[1]: return W
    for mname in ("get_W","get_W_est","get_adjacency","adjacency_matrix","get_w_adj"):
        if hasattr(model, mname):
            W = getattr(model, mname)()
            if isinstance(W, np.ndarray): W = torch.from_numpy(W)
            if hasattr(W,"ndim") and W.ndim==2 and W.shape[0]==W.shape[1]: return W
    raise AttributeError("No adjacency/weights found on model.")


def _load_W_from_artifacts(outdir: Path):
    patterns=["W.npy","W_est.npy","adjacency.npy","adj_matrix.npy","A.npy","graph.npy","B.npy","weights.npy",
              "W.csv","adjacency.csv","graph.csv","checkpoint*.pt","model*.pt","*.pt","*.pth"]
    cands=[]; [cands.extend(outdir.rglob(p)) for p in patterns]; cands.sort(key=lambda p:p.stat().st_mtime, reverse=True)
    for pth in cands:
        try:
            if pth.suffix==".npy":
                arr=np.load(pth); 
                if arr.ndim==1:
                    s=int(round(arr.size**0.5)); 
                    if s*s!=arr.size: continue
                    arr=arr.reshape(s,s)
                if arr.ndim==2 and arr.shape[0]==arr.shape[1] and arr.shape[0]>=2: return torch.from_numpy(arr.astype(float)), None, pth.parent
            elif pth.suffix==".csv":
                df=pd.read_csv(pth, index_col=0)
                if df.shape[0]==df.shape[1] and df.shape[0]>=2: return torch.from_numpy(df.values.astype(float)), list(df.columns), pth.parent
            elif pth.suffix in (".pt",".pth"):
                ckpt=torch.load(pth, map_location="cpu")
                def pick_sq(d):
                    best=None; best_sz=-1
                    if isinstance(d,dict):
                        stacks=[d]
                        if "state_dict" in d and isinstance(d["state_dict"],dict): stacks.append(d["state_dict"])
                        if "model" in d and isinstance(d["model"],dict): stacks.append(d["model"])
                        for sd in stacks:
                            for k,v in sd.items():
                                if hasattr(v,"ndim") and getattr(v,"ndim",0)==2 and v.shape[0]==v.shape[1]:
                                    sz=v.shape[0]
                                    if sz>best_sz: best, best_sz = v, sz
                    return best
                Wc=pick_sq(ckpt)
                if Wc is not None: return Wc.detach().clone().cpu(), None, pth.parent
        except Exception: continue
    return None, None, outdir


def save_outputs(*args_, **kw):
    args_ns=kw.pop("args", None); model=kw.pop("model", None); dataset_like=kw.pop("dataset_like", None); names_like=kw.pop("names_like", None)
    extra_dirs=kw.pop("extra_search_dirs", None) or []
    if len(args_)>=1 and args_ns is None: args_ns=args_[0]
    if len(args_)>=2 and model is None: model=args_[1]
    if len(args_)>=3 and dataset_like is None: dataset_like=args_[2]
    if args_ns is None:
        args_ns=SimpleNamespace(outdir=kw.get("outdir","./dcdi_out"), tau=kw.get("tau",0.25), data=kw.get("data") or kw.get("dataset_path"),
                                dataset_path=kw.get("dataset_path"), drop_cols=kw.get("drop_cols"), env_col=kw.get("env_col"))
    root_out=Path(args_ns.outdir).resolve(); root_out.mkdir(parents=True, exist_ok=True)
    tau=float(getattr(args_ns,"tau",0.25))

    def _extract_names_from(obj):
        if obj is None: return None
        if isinstance(obj,(list,tuple,pd.Index,np.ndarray)): return list(obj)
        if isinstance(obj,pd.DataFrame): return list(obj.columns)
        if isinstance(obj,(str,Path)) and str(obj).endswith((".csv",".tsv")):
            try: return list(pd.read_csv(obj, nrows=1).columns)
            except Exception: return None
        for attr in ("variable_names","var_names","vars","columns","features","names"):
            if hasattr(obj,attr):
                v=getattr(obj,attr); v=v() if callable(v) else v
                if isinstance(v,(list,tuple,pd.Index,np.ndarray)): return list(v)
        return None

    var_names=_extract_names_from(names_like) or _extract_names_from(dataset_like) or _extract_names_from(model)
    if not var_names:
        data_path=getattr(args_ns,"data",None) or getattr(args_ns,"dataset_path",None)
        try:
            if data_path: var_names=list(pd.read_csv(data_path, nrows=1).columns)
        except Exception: var_names=None

    W_t=None
    if model is not None:
        try: W_t=_extract_W_from_model(model)
        except Exception as e: print(f"[save_outputs] Could not extract W from live model: {e}")

    if W_t is None:
        search_roots=[root_out, root_out/"train"] + [Path(d) for d in extra_dirs if d]
        for r in search_roots:
            W_t, vn2, chosen=_load_W_from_artifacts(Path(r))
            if W_t is not None:
                if vn2 and not var_names: var_names=vn2
                root_out=Path(chosen); break

    if W_t is None: raise RuntimeError("Model is None and no adjacency matrix was found.")
    if not var_names: var_names=[f"x{i}" for i in range(W_t.shape[0])]

    _save_graph_outputs(W_t, var_names, root_out, tau)
    train_dir = Path(args_ns.outdir) / "train"
    try:
        train_dir.mkdir(parents=True, exist_ok=True)
        _save_graph_outputs(W_t, var_names, train_dir, tau)
    except Exception as e:
        print(f"[dcdi_run] note: could not mirror outputs to train/: {e}")
    print(f"[dcdi_run] W/edges written under: {root_out} and {train_dir}")


# ---------- metrics ----------

class MetricsLogger:
    """
    Tolerant logger for trainer metrics. Accepts any kwargs, finds numeric/0-d tensors,
    writes to metrics.csv. Keeps an internal step counter if trainer doesn't pass one.
    """
    def __init__(self, outdir: Path):
        self.path = Path(outdir) / "train" / "metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = set(["iter"])
        self._step = 0
        # create empty file with header later when first row arrives

    def __call__(self, *args, **kwargs):
        import math
        row = {}
        # iteration from kwargs or bump counter
        it = None
        for k in ("iter","iteration","step","t","i"):
            if k in kwargs:
                try: it = int(kwargs[k])
                except Exception: pass
                break
        if it is None:
            self._step += 1
            it = self._step
        row["iter"] = it

        # if args contains a dict, include it
        if args and isinstance(args[0], dict):
            kwargs = {**args[0], **kwargs}

        # extract numerics
        for k, v in kwargs.items():
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    if v.ndim == 0:
                        v = float(v.detach().cpu().item())
                    else:
                        continue
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    row[k] = float(v)
            except Exception:
                continue

        # init header and write
        self.fieldnames.update(row.keys())
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(self.fieldnames))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

def _postplot_metrics(outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        mpath = Path(outdir) / "train" / "metrics.csv"
        if not mpath.exists():
            return
        df = pd.read_csv(mpath)
        # pick common columns if present
        cols = [c for c in df.columns if c.lower() in ("iter","iteration","step")]
        x = df[cols[0]] if cols else pd.Series(range(len(df)))
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("nll","al","aug","loss","h"))]
        plt.figure(figsize=(7.5,4.8))
        for c in candidates:
            if c == cols[0]: continue
            plt.plot(x, df[c], label=c)
        plt.xlabel("Iterations")
        plt.ylabel("Value")
        if candidates:
            top = df[candidates].max().max()
            plt.ylim(0, top*1.1 if top>0 else 1)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(Path(outdir) / "train" / "metrics_zoom.png", dpi=140)
    except Exception as e:
        print(f"[dcdi_run] metrics plot skipped: {e}")


# ---------- training core ----------

def _call_dcdi_train_flex(argv_forward, args):
    # ---------- I/O, cleaning ----------
    outdir = Path(getattr(args, "outdir", "./dcdi_out")).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    train_root = outdir / "train"; train_root.mkdir(parents=True, exist_ok=True)

    data_path = getattr(args, "data", None) or getattr(args, "dataset_path", None)
    if not data_path:
        raise RuntimeError("No --data / --dataset-path given for DCDI.")
    df = pd.read_csv(data_path, low_memory=False)

    drop_cols = set(_as_list(getattr(args, "drop_cols", None)))
    env_col = getattr(args, "env_col", None)
    if env_col and env_col in df.columns:
        drop_cols.add(env_col)

    likely_env = {"env","environment","condition","cond","interv","intervention","batch","group"}
    for c in df.columns:
        if c.lower() in likely_env:
            drop_cols.add(c)

    # drop categorical regime-like columns with few distinct values
    for c in list(df.columns):
        if c in drop_cols: 
            continue
        ser = df[c]
        if ser.dtype == object:
            vals = ser.dropna().astype(str).unique()
            if 1 <= len(vals) <= max(10, int(0.02 * len(ser))):
                if all(re.match(r"^(e\d+|env\d+|control|treated|treatment|case|ctrl)$", v.lower()) for v in vals):
                    drop_cols.add(c)

    kept = [c for c in df.columns if c not in drop_cols]
    if len(kept) < 2:
        raise RuntimeError(f"After dropping env-like columns {sorted(drop_cols)}, <2 feature columns remain.")

    df_num = df[kept].apply(pd.to_numeric, errors="coerce")
    # factorize columns that were fully non-numeric
    all_nan = [c for c in df_num.columns if df_num[c].notna().sum() == 0]
    for c in all_nan:
        codes, _ = pd.factorize(df[c].astype(str), sort=True)
        if len(set(codes)) > 1:
            df_num[c] = codes.astype(float)
    # drop any still-all-nan columns
    df_num = df_num[[c for c in df_num.columns if df_num[c].notna().sum() > 0]].astype(float)
    df_num = df_num.fillna(df_num.mean())

    var_names = list(df_num.columns)
    if len(var_names) < 2:
        raise RuntimeError("Insufficient numeric features for DCDI after cleaning.")

    X = df_num.values.astype(np.float32)
    xmu = X.mean(axis=0, keepdims=True)
    xsd = X.std(axis=0, keepdims=True); xsd[xsd == 0] = 1.0
    Xz = (X - xmu) / xsd

    p = int(len(var_names))
    seed = _argval(argv_forward, "seed", int, 123)
    rng = np.random.RandomState(seed)
    n = Xz.shape[0]; idx = np.arange(n); rng.shuffle(idx)
    split = max(1, int(0.8 * n))
    tr = idx[:split]; te = idx[split:] if split < n else idx[:split]
    train_np = Xz[tr]; test_np = Xz[te] if split < n else None

    # ---------- hyperparams ----------
    train_batch_size = _argval_any(argv_forward, ["train-batch-size","train_batch_size","batch-size","batch_size","bs"], int, 128)
    num_train_iter   = _argval_any(argv_forward, ["num-train-iter","num_train_iter","train-iters","train_iters","n-iters","n_iters","steps","train_steps"], int, 20000)
    lr               = _argval(argv_forward, "lr", float, 1e-2)
    max_indegree     = _argval(argv_forward, "max-indegree", int, 3)
    device_cli       = _argval(argv_forward, "device", str, "cpu").lower()
    weight_decay     = _argval(argv_forward, "weight-decay", float, 0.0)
    optimizer_name   = (_argval(argv_forward, "optimizer", str, None) or "sgd").lower()
    stop_win         = _argval_any(argv_forward, ["stop-crit-win","stop_crit_win","stopwin","stop_window"], int, 100)
    snapshot_freq    = _argval_any(argv_forward, ["snapshot-freq","snapshot_freq","checkpoint-every","checkpoint_every"], int, 100)
    eval_num_samples = _argval_any(argv_forward, ["eval-num-samples","eval_num_samples","num-samples","num_samples","test-num-samples","test_num_samples"], int, min(2048, (0 if test_np is None else test_np.shape[0])))
    plot_freq        = _argval_any(argv_forward, ["plot-freq","plot_freq"], int, 100)

    # AL knobs
    mu_init     = _argval_any(argv_forward, ["mu-init","mu_init","mu"], float, 0.5)
    gamma_init  = _argval_any(argv_forward, ["gamma-init","gamma_init","gamma0"], float, 1e-2)
    omega_mu    = _argval_any(argv_forward, ["omega-mu","omega_mu"], float, 1.0)
    omega_gamma = _argval_any(argv_forward, ["omega-gamma","omega_gamma"], float, 1.0)
    h_threshold = _argval_any(argv_forward, ["h-threshold","h_threshold","h_thresh","hthresh"], float, 1e-4)

    print(f"[dcdi_run] using num_train_iter={num_train_iter}, train_batch_size={train_batch_size}, eval_num_samples={eval_num_samples}, plot_freq={plot_freq}")
    print(f"[dcdi_run] AL knobs: mu_init={mu_init}, gamma_init={gamma_init}, omega_mu={omega_mu}, omega_gamma={omega_gamma}, h_threshold={h_threshold}")

    opt = AutoOpt(
        base=dict(
            epochs=200, lr=lr, max_indegree=max_indegree, device=device_cli, weight_decay=weight_decay,
            undirected=bool(getattr(args, "undirected", False)), outdir=str(outdir), save_artifacts=True,
            verbose=True, early_stopping=True,
            optimizer=optimizer_name, momentum=(0.9 if optimizer_name == "sgd" else 0.0),
            train_batch_size=train_batch_size, batch_size=train_batch_size, eval_batch_size=train_batch_size,
            num_train_iter=num_train_iter, num_train_steps=num_train_iter, n_iters=num_train_iter, train_steps=num_train_iter,
            stop_crit_win=stop_win, snapshot_freq=snapshot_freq, plot_freq=plot_freq,
            mu_init=mu_init, gamma_init=gamma_init, omega_mu=omega_mu, omega_gamma=omega_gamma, h_threshold=h_threshold
        ),
        p=p, seed=seed, outdir=outdir, device=device_cli
    )

    # ---------- tensors & samplers (device-aware) ----------
    device_str = ("cuda"
                  if ((device_cli == "cuda") or getattr(opt, "use_cuda", False)) and torch.cuda.is_available()
                  else "cpu")

    def _to_tensor(x, dev=device_str):
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return t.to(dev, non_blocking=True)

    train_tensor = _to_tensor(train_np)
    test_tensor  = _to_tensor(test_np) if test_np is not None else None

    ds_train = DCDISampler(train_tensor, device=device_str, num_samples=train_batch_size)
    ds_test  = DCDISampler(test_tensor,  device=device_str, num_samples=eval_num_samples) if test_tensor is not None else None
    print(f"[dcdi_run] sampler ready: N={ds_train.N}, p={ds_train.p}, batch={opt.train_batch_size}")

    # ---------- import train(), paths, warnings ----------
    train_mod = _smart_import_dcdi_train()
    if not hasattr(train_mod, "train"):
        raise RuntimeError("dcdi.train has no train()")
    train_fn = getattr(train_mod, "train")

    opt.exp_path = str(train_root)
    opt.log_dir  = str(train_root / "logs")
    opt.ckpt_dir = str(train_root / "checkpoints")
    for d in (train_root, Path(opt.log_dir), Path(opt.ckpt_dir)):
        d.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="Data has no positive values, and therefore cannot be log-scaled.")

    # ---------- build model seed ----------
    model_seed = _build_dcdi_model(train_mod, p, opt)
    if device_str == "cuda" and model_seed is not None and hasattr(model_seed, "to"):
        model_seed = model_seed.to("cuda")

    # ---------- metrics logger ----------
    metrics = MetricsLogger(outdir)
    def _noop(*a, **k): return None
    gt_A = np.zeros((p, p), dtype=np.int64); gt_interv = None

    # ---------- force CUDA flags (forks gate on these) ----------
    opt.use_cuda = (device_str == "cuda")
    opt.device   = device_str
    if device_str == "cuda":
        torch.cuda.set_device(0)

    # ---------- train: capture return (positional → kwargs) ----------
    try:
        model = train_fn(model_seed, gt_A, gt_interv, ds_train, ds_test, opt, metrics, _noop)
    except TypeError:
        model = train_fn(
            model=model_seed,
            gt_adjacency=gt_A, gt_interv=gt_interv,
            train_data=ds_train, test_data=ds_test,
            opt=opt, metrics_callback=metrics, plotting_callback=_noop,
        )
    if isinstance(model, tuple) and len(model) > 0:
        model = model[0]

    # ---------- post plots ----------
    try:
        _postplot_metrics(outdir)
    except Exception as e:
        print(f"[dcdi_run] metrics postplot failed: {e}")

    dataset_like = SimpleNamespace(variable_names=var_names)
    extra_search_dirs = [outdir, train_root]
    return model, dataset_like, extra_search_dirs


# ---------- CLI ----------

def main():
    parser=argparse.ArgumentParser(description="Robust DCDI runner (patched v15)")
    parser.add_argument("--data","--dataset-path",dest="data",required=True,help="Path to CSV")
    parser.add_argument("--drop-cols",dest="drop_cols",default=None,help="Columns to drop (comma/space separated)")
    parser.add_argument("--env-col","--env_col",dest="env_col",default=None,help="Environment column name")
    parser.add_argument("--outdir",required=True,help="Output directory")
    parser.add_argument("--tau",type=float,default=0.25,help="Threshold for |W| -> adjacency")
    parser.add_argument("--undirected",action="store_true",help="Report undirected skeletons (thresholding still directional)")
    args, forward = parser.parse_known_args()
    model, dataset_like, extra_search_dirs = _call_dcdi_train_flex(forward, args)
    save_outputs(args, model, dataset_like, names_like=dataset_like, extra_search_dirs=extra_search_dirs)


if __name__=="__main__":
    main()
