# Reproduce the Paper-Facing Experiments

This repository is intentionally smaller than the original internal codebase. The commands below focus on the clean paper-facing paths retained here.

## 1. Inventory check

Run this first:

```bash
python3 tools/check_trimmed_paper_inventory.py
```

## 2. Interference toy example

Generate the overlapping-cover simulation:

```bash
python3 experiments/interference/interference_minimal_v2.py \
  --outdir artifacts/interference
```

Render the paper-facing plots from the same output directory:

```bash
python3 experiments/interference/interference_plots.py \
  --outdir artifacts/interference
```

## 3. Synthetic psi-FCI sweep

Run the three-alpha sweep used in the paper-facing synthetic experiment:

```bash
python3 experiments/psifci/alpha_sweep_psifci.py \
  --data data/psifci/synth_data.csv \
  --env-col env \
  --true data/psifci/A_true_Jstable.csv \
  --alphas 0.005,0.01,0.02 \
  --outdir artifacts/psifci/generated \
  --depth 0 \
  --standardize \
  --undirected
```

The checked-in summary artifact retained from the original paper bundle is:

- `artifacts/psifci/grid_summary.csv`

## 4. Sachs j-stable evaluation

The `align_and_eval_sachs.py` script does not generate a graph by itself. Its `--pred` argument must point to a square adjacency CSV produced by another script, typically `experiments/psifci/run_psifci_jstable.py`.

The clean public path retained in this repository is:

1. generate a pooled Sachs adjacency,
2. generate a per-environment strict-intersection adjacency,
3. align either one to the Sachs reference graph and score it.

### 4a. Generate a pooled Sachs prediction

```bash
python3 experiments/psifci/run_psifci_jstable.py \
  --data data/sachs/sachs_with_env.csv \
  --alpha 0.001 \
  --outdir artifacts/sachs/generated/pooled
```

This writes the pooled adjacency to:

- `artifacts/sachs/generated/pooled/fci_envpooled.csv`

### 4b. Generate a per-environment strict-intersection Sachs prediction

```bash
python3 experiments/psifci/run_psifci_jstable.py \
  --data data/sachs/sachs_with_env.csv \
  --env-col env \
  --alpha 0.001 \
  --outdir artifacts/sachs/generated/per_env
```

This writes:

- per-environment adjacencies such as `fci_enve0.csv`, `fci_enve1.csv`, ...
- the strict j-stable intersection adjacency:
  - `artifacts/sachs/generated/per_env/A_Jstable_fci.csv`
- support diagnostics:
  - `artifacts/sachs/generated/per_env/support_counts.csv`

### 4c. Align and evaluate the pooled prediction

```bash
python3 experiments/sachs/align_and_eval_sachs.py \
  --true data/sachs/A_true_sachs_aligned.csv \
  --pred artifacts/sachs/generated/pooled/fci_envpooled.csv \
  --data data/sachs/sachs_with_env.csv \
  --outdir artifacts/sachs/eval_pooled
```

### 4d. Align and evaluate the strict-intersection prediction

```bash
python3 experiments/sachs/align_and_eval_sachs.py \
  --true data/sachs/A_true_sachs_aligned.csv \
  --pred artifacts/sachs/generated/per_env/A_Jstable_fci.csv \
  --data data/sachs/sachs_with_env.csv \
  --outdir artifacts/sachs/eval_intersection
```

The key thing to remember is that `--pred` must be a square adjacency CSV with row and column headers. In the retained public bundle, the two most important choices are:

- pooled: `fci_envpooled.csv`
- strict j-stable intersection: `A_Jstable_fci.csv`

The historical checked-in Sachs grid summary kept in `artifacts/sachs/sachs_grid_summary.csv` contains additional settings from the larger internal codebase. The lean public repository is clearer and more reliable for the two commands above than for reproducing the entire historical grid sweep end-to-end.

The canonical checked-in summary artifacts are:

- `artifacts/sachs/sachs_grid_summary.csv`
- `artifacts/sachs/pooled_report.json`
- `artifacts/sachs/per_env_intersection_report.json`

## 5. Sachs environment construction

The retained environment-label helper can rebuild a clustered `env` column if needed:

```bash
python3 experiments/sachs/make_env_labels_sachs.py \
  --in-csv data/sachs/sachs.csv \
  --out-csv data/sachs/sachs_with_env.generated.csv \
  --auto-k \
  --k-min 3 \
  --k-max 6 \
  --min-env-size 30 \
  --standardize
```

## 6. DCDI note

The trimmed repository keeps the benchmark wrapper code and one sample benchmark input:

- `experiments/dcdi/dcdi_benchmark_val.py`
- `experiments/dcdi/dcdi_run_patched_v15.py`
- `experiments/dcdi/jstable_eval.py`
- `data/dcdi_sample/lin_perfect_d20_e1_g02/`

The full vendored DCDI training backend from the internal repository is intentionally omitted here to keep the public bundle small. The paper-facing DCDI section is therefore validated in this repository mainly through the checked-in summary artifact:

- `artifacts/dcdi/results_lin_perfect.csv`

If you later decide to make the DCDI path fully runnable here, the next step is to vendor only the lightweight `dcdi` Python package code, not the historical data and experiment dumps.

## 7. Supporting GES summaries

The paper includes a supporting score-based example. The retained script is:

- `experiments/sachs/run_ges.py`

The checked-in summaries are:

- `artifacts/ges/synthetic_pooled_report.json`
- `artifacts/ges/synthetic_jstable_report.json`
