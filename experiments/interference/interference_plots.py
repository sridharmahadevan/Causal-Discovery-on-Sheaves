#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_betas(outdir: Path, cover_name: str) -> bool:
    path = outdir / f"betas_{cover_name}.csv"
    if not path.exists():
        return False
    betas = pd.read_csv(path)
    plt.figure(figsize=(4.2, 4))
    plt.boxplot([betas["beta_E1"], betas["beta_E2"]], labels=["β(E1)", "β(E2)"])
    plt.axhline(0, color="#94a3b8", lw=1)
    plt.title(f"Per-chart betas on {cover_name}")
    plt.tight_layout()
    plt.savefig(outdir / f"betas_{cover_name}.png", dpi=200)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/interference")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    table = pd.read_csv(outdir / "stability_by_cover.csv")

    covers = table["cover"].tolist()
    f1 = table["freq_E1→Y"].values
    f2 = table["freq_E2→Y"].values

    plt.figure(figsize=(7, 4))
    x = range(len(covers))
    plt.bar(x, f1, width=0.4, label="E1→Y", color="#3b82f6")
    plt.bar([i + 0.4 for i in x], f2, width=0.4, label="E2→Y", color="#10b981")
    plt.xticks([i + 0.2 for i in x], covers, rotation=0)
    plt.ylim(0, 1.05)
    plt.ylabel("Edge frequency (charts K=10)")
    plt.title("Interference covers — edge frequencies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "freq_by_cover.png", dpi=200)
    plt.close()

    has_wl_lm = plot_betas(outdir, "WL∩LM")
    has_e_lm = plot_betas(outdir, "E∩LM")
    print("[ok] wrote:", outdir / "freq_by_cover.png")
    if has_wl_lm:
        print("[ok] wrote:", outdir / "betas_WL∩LM.png")
    if has_e_lm:
        print("[ok] wrote:", outdir / "betas_E∩LM.png")


if __name__ == "__main__":
    main()
