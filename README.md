## Causal Discovery on Sheaves

This repository is a lean software companion for work on causal discovery over regime covers, sheaf-style gluing, and judo-calculus-based causal inference.

The goal of this trimmed repository is simple:

- keep the canonical scripts for the experiments discussed in the paper,
- keep the small datasets needed for the clean paper-facing runs,
- keep checked-in summary artifacts that validate the reported results,
- leave out large legacy trees, virtual environments, and exploratory runs.

### Background papers

This repository is best read together with the following papers:

- Sridhar Mahadevan. [Decentralized Causal Discovery using Judo Calculus](https://arxiv.org/abs/2510.23942). CoRR, 2025.
- Sridhar Mahadevan. [Intuitionistic j-Do-Calculus in Topos Causal Models](https://arxiv.org/abs/2510.17944). CoRR, 2025.
- Sridhar Mahadevan. [Universal Causal Inference in a Topos](https://openreview.net/forum?id=TOhpnECT10). NeurIPS 2025 Spotlight.

### What is in scope

- `experiments/interference/`
  - minimal overlapping-cover interference simulation and plotting
- `experiments/psifci/`
  - synthetic `psi`-FCI sweep and the regime-wise j-stable wrapper
- `experiments/dcdi/`
  - DCDI benchmark wrappers and evaluation utilities
- `experiments/sachs/`
  - Sachs alignment, environment labeling, and supporting GES script
- `data/`
  - small paper-facing inputs for synthetic `psi`-FCI, Sachs, and one DCDI sample instance
- `artifacts/`
  - checked-in summary outputs used to validate the paper tables and figures
- `docs/`
  - reproduction notes and a paper-to-artifact map
- `tools/check_trimmed_paper_inventory.py`
  - a lightweight integrity checker for this trimmed bundle

### What is intentionally omitted

- large vendored trees and duplicate snapshots,
- internal virtual environments,
- LINCS and PISA pipelines,
- full historical run directories,
- the heavier private DCDI backend/data bundle used for the full benchmark sweep.

The DCDI wrappers are retained because they define the paper-facing interface, but this public bundle validates the DCDI section primarily through checked-in summary artifacts plus a small sample benchmark input.

### Quick start

1. Create an environment and install the lightweight dependencies in `requirements.txt`.
2. Run the bundle checker:

```bash
python3 tools/check_trimmed_paper_inventory.py
```

3. See [docs/REPRODUCE.md](/Users/sridharmahadevan/Downloads/mac-cech_homology_GT/Causal-Discovery-on-Sheaves/docs/REPRODUCE.md) for example commands.
4. See [docs/ARTIFACT_MAP.md](/Users/sridharmahadevan/Downloads/mac-cech_homology_GT/Causal-Discovery-on-Sheaves/docs/ARTIFACT_MAP.md) for the mapping from paper items to scripts and artifacts.

### Notes

- The cleanest fully executable paths in this trimmed repository are the interference and synthetic `psi`-FCI experiments.
- Sachs is supported by checked-in data and evaluation utilities.
- DCDI is represented by the benchmark wrappers, a sample input, and the checked-in summary table used in the paper.
