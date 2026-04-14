#!/usr/bin/env python3
"""Lightweight inventory checker for the trimmed paper-facing bundle."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "name": "Interference",
        "status": "verified",
        "paths": [
            "experiments/interference/interference_minimal_v2.py",
            "experiments/interference/interference_plots.py",
            "artifacts/interference/stability_by_cover.csv",
        ],
    },
    {
        "name": "Synthetic psi-FCI",
        "status": "verified",
        "paths": [
            "experiments/psifci/alpha_sweep_psifci.py",
            "experiments/psifci/run_psifci_jstable.py",
            "data/psifci/synth_data.csv",
            "data/psifci/A_true_Jstable.csv",
            "artifacts/psifci/grid_summary.csv",
            "artifacts/psifci/raw_alpha_0.005_agg_report.json",
        ],
    },
    {
        "name": "Synthetic GES",
        "status": "artifact-backed",
        "paths": [
            "experiments/sachs/run_ges.py",
            "artifacts/ges/synthetic_pooled_report.json",
            "artifacts/ges/synthetic_jstable_report.json",
        ],
    },
    {
        "name": "Synthetic DCDI",
        "status": "sample-plus-artifact",
        "paths": [
            "experiments/dcdi/dcdi_benchmark_val.py",
            "experiments/dcdi/dcdi_run_patched_v15.py",
            "experiments/dcdi/jstable_eval.py",
            "data/dcdi_sample/lin_perfect_d20_e1_g02/data.csv",
            "data/dcdi_sample/lin_perfect_d20_e1_g02/A_true.csv",
            "artifacts/dcdi/results_lin_perfect.csv",
        ],
    },
    {
        "name": "Sachs",
        "status": "verified",
        "paths": [
            "experiments/sachs/align_and_eval_sachs.py",
            "experiments/sachs/make_env_labels_sachs.py",
            "data/sachs/sachs.csv",
            "data/sachs/sachs_with_env.csv",
            "data/sachs/A_true_sachs_aligned.csv",
            "artifacts/sachs/sachs_grid_summary.csv",
            "artifacts/sachs/pooled_report.json",
            "artifacts/sachs/per_env_intersection_report.json",
        ],
    },
]


def rel_exists(relpath: str) -> bool:
    return (ROOT / relpath).exists()


def check_matrix_shape(path: Path) -> tuple[bool, str]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
    except Exception as exc:
        return False, f"unreadable: {exc}"
    if not rows:
        return False, "empty"
    width = len(rows[0])
    if len(rows) != width:
        return False, f"non-square csv ({len(rows)}x{width})"
    return True, f"square csv ({len(rows)}x{width})"


def synthetic_integrity_checks() -> Iterable[str]:
    checks = [
        ROOT / "data" / "psifci" / "A_true_Jstable.csv",
        ROOT / "data" / "sachs" / "A_true_sachs_aligned.csv",
        ROOT / "data" / "dcdi_sample" / "lin_perfect_d20_e1_g02" / "A_true.csv",
    ]
    for path in checks:
        if not path.exists():
            continue
        ok, detail = check_matrix_shape(path)
        if not ok:
            yield f"[warn] {path.relative_to(ROOT)}: {detail}"


def main() -> int:
    print("# Trimmed Paper Inventory\n")
    for experiment in EXPERIMENTS:
        name = experiment["name"]
        status = experiment["status"]
        print(f"## {name}")
        print(f"status: {status}")
        missing = []
        for relpath in experiment["paths"]:
            exists = rel_exists(relpath)
            marker = "ok" if exists else "missing"
            print(f"- [{marker}] {relpath}")
            if not exists:
                missing.append(relpath)
        if missing:
            print(f"summary: missing {len(missing)} artifact(s)")
        else:
            print("summary: all listed artifacts present")
        print()

    issues = list(synthetic_integrity_checks())
    print("## Integrity checks")
    if issues:
        for issue in issues:
            print(f"- {issue}")
    else:
        print("- no malformed square-adjacency files detected in retained sample data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
