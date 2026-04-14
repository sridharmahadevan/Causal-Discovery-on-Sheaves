# Artifact Map

| Paper item | Experiment family | Script(s) | Checked-in artifact(s) | Notes |
|---|---|---|---|---|
| Interference cover illustration | Interference | `experiments/interference/interference_minimal_v2.py`, `experiments/interference/interference_plots.py` | `artifacts/interference/stability_by_cover.csv` | Cleanest executable toy path in the trimmed repo |
| Synthetic `psi`-FCI alpha sweep | Synthetic `psi`-FCI | `experiments/psifci/alpha_sweep_psifci.py`, `experiments/psifci/run_psifci_jstable.py` | `artifacts/psifci/grid_summary.csv` | Main synthetic table in the paper-facing bundle |
| Single synthetic `psi`-FCI raw slice | Synthetic `psi`-FCI | `experiments/psifci/run_psifci_jstable.py` | `artifacts/psifci/raw_alpha_0.005_agg_report.json` | Supports the “raw counts” slice discussion |
| Supporting synthetic score-based result | GES | `experiments/sachs/run_ges.py` | `artifacts/ges/synthetic_pooled_report.json`, `artifacts/ges/synthetic_jstable_report.json` | Supporting evidence only, not the core story |
| Synthetic DCDI benchmark summary | DCDI | `experiments/dcdi/dcdi_benchmark_val.py`, `experiments/dcdi/dcdi_run_patched_v15.py`, `experiments/dcdi/jstable_eval.py` | `artifacts/dcdi/results_lin_perfect.csv` | Full heavy backend omitted from trimmed repo |
| Sachs pooled vs j-stable summary | Sachs | `experiments/psifci/run_psifci_jstable.py`, `experiments/sachs/align_and_eval_sachs.py` | `artifacts/sachs/sachs_grid_summary.csv` | Main Sachs summary retained from the paper bundle |
| Sachs pooled alignment report | Sachs | `experiments/sachs/align_and_eval_sachs.py` | `artifacts/sachs/pooled_report.json` | Validation report on pooled Sachs output |
| Sachs per-environment intersection report | Sachs | `experiments/sachs/align_and_eval_sachs.py` | `artifacts/sachs/per_env_intersection_report.json` | Validation report on strict intersection |

## Data map

- `data/psifci/`
  - synthetic `psi`-FCI benchmark data and ground truth
- `data/sachs/`
  - Sachs table, Sachs with environment labels, and aligned reference graph
- `data/dcdi_sample/lin_perfect_d20_e1_g02/`
  - one sample DCDI benchmark instance kept to document input format
