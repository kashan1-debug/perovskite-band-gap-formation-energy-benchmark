# Benchmarking the Generalization and Extrapolation Limits of Composition-Based Machine Learning Models for Perovskite Formation Energy and Band-Gap Prediction

> How far can a model trained only on chemical composition go — and where does it break when it meets a metal it has never seen?

This repository contains the complete, reproducible workflow behind a benchmark study of machine-learning models for perovskite formation energy and band-gap prediction on the Castelli grid of **18,928 ABX₃ perovskites** (52 A × 52 B × 7 X). Every experiment was run in Google Colab from the cells in [`code/`](code/), with stop gates at each stage; the full plan is in [`docs/workflow.md`](docs/workflow.md).

**Status:** manuscript in preparation (see [`paper/manuscript.md`](paper/manuscript.md)). Not yet submitted.

## Key findings

All formation-energy numbers below are **pooled R² on leave-one-B-out** — the model is trained without one B-site metal and tested on all 364 compounds containing it, pooled over all 52 metals. This is the hard test: generalisation to an unseen element, not interpolation between familiar crystals.

| Model | Pooled leave-one-B-out R² |
|---|---|
| Magpie-only XGBoost | 0.086 |
| Magpie-only Ridge | 0.206 |
| **Site-aware XGBoost** | **0.733** |
| No-B-information XGBoost (ablation) | 0.683 |
| Pretrained MEGNet, calibrated (zero-shot) | 0.561 |

Three more facts frame the result:

- **There is a ceiling.** Compounds with the same composition but swapped A/B sites share identical Magpie features yet differ in formation energy. The pooled R² ceiling from this composition degeneracy alone is **0.543** — so a composition-only model can never reach 1.0 on this grid.
- **Site resolution, not B-site physics, carries most of the gain.** Giving the model the A, B and X sites explicitly (site-aware features) lifts pooled R² from 0.086 to 0.733. Removing all B-site information only drops it to 0.683 — B-site physics adds just **+0.050**, and helps some metals (W, Re, Ta) while hurting others (Cr).
- **The graph model does not win.** A pretrained MEGNet (trained on 133,420 Materials Project structures), used zero-shot and calibrated, reaches 0.561 — below site-aware XGBoost at 0.733. Its raw predictions are off by a constant scale offset (raw R² −77.8), because the two datasets reference oxygen differently; calibration fixes the scale, not the ranking (Spearman 0.752). The 0.561 is an upper bound: ~6.2% of Castelli formulas also appear in Materials Project.

Uncertainty under element shift is metal-dependent. Split-conformal intervals (nominal 90%) hold on a random split (coverage **0.899**) and on held-out Te (**0.923**) and Mg (**0.945**), but **fail on W (0.690)** — coverage is not a property of the method, it is a property of the metal.

Band-gap screening on the same grid is a cautionary tale: only 735 of 18,928 compounds have a non-zero DFT gap. A classifier reaches ROC-AUC 0.919, but at the usual 0.5 threshold it finds only ~7% of the non-metals. Lowering the threshold to ~0.2 gives the best F1 (0.528). Grouped regression on the 735 non-zero-gap compounds gives R² 0.558.

## Dataset

- **Source:** Castelli perovskites via [matminer](https://hackingmaterials.lbl.gov/matminer/) (`matminer.datasets.load_dataset("castelli_perovskites")`), 18,928 rows. The dataset is a complete factorial grid: every B metal appears in exactly 364 compounds, every X anion in exactly 2,704.
- **Target:** formation energy per atom, `e_form / 5` (the stored `e_form` is for the whole 5-atom cell). Mean 0.294 eV/atom; only 74 rows are negative, so **no stability claim is made** — stability needs energy above the convex hull, which this dataset does not contain.
- **Band gap:** the DFT (GLLBSC) gap, not an experimental gap. Zero for 18,193 compounds (96.1%); non-zero for 735.
- **Features:** composition only — Magpie elemental descriptors (132), plus site-resolved per-site physics and derived textbook quantities (tolerance factor *t*, octahedral factor *μ*) in the site-aware experiment. No DFT-derived quantities are ever used as features.

## Repository layout

```
├── code/           # One pasteable Google Colab cell per file, run in order (00 → 12)
│   ├── 00_setup_colab.txt      # Colab setup: install (A), imports (B), Drive mount (C)
│   ├── 01_…_12_*.py            # Stages 1–7, 3b, 3c, 9a, 9a2, 9b
│   └── benchmark_v2.py         # Standalone audited script (same pipeline, local run)
├── data/           # Dataset provenance and how to fetch it (no data files in the repo)
├── docs/
│   ├── workflow.md             # Full pre-registered workflow: decisions, stages, stop gates
│   ├── literature_check.md     # Novelty assessment of the five claims (2 NEW, 3 PARTIAL)
│   └── understanding_the_paper.md  # Plain-language guide to the six key numbers
├── paper/
│   ├── manuscript.md           # Full manuscript draft (v2)
│   └── figures/                # Figures 1–7 (PNG, 300 dpi)
└── results/        # Result tables and reports produced by the cells
```

## Reproducing the results

1. Open Google Colab, create a blank notebook.
2. Paste `code/00_setup_colab.txt` — Setup A (install, once per runtime), restart the runtime, then Setup B and C. See [`code/README.md`](code/README.md) for the exact order.
3. Paste and run the stage cells in order, `01` → `12`. Each cell is safe to re-run and saves its outputs to `MyDrive/perovskite/results_v2/`.
4. Expected runtimes are noted in `docs/workflow.md` (Stage 3 ≈ 17 min; Stage 9b ≈ 3 min).

Environment: Python 3, scikit-learn 1.6.1, XGBoost 3.4.1, matminer 0.10.1, pymatgen 2026.10.24, SHAP, MEGNet 1.3.2 / TensorFlow 2.21.0 (Stage 9b only).

## What is new here

Per the literature check in [`docs/literature_check.md`](docs/literature_check.md), no prior study combined these five elements on the Castelli grid: the composition-degeneracy ceiling, pooled leave-one-B-out reporting, site-resolved features tested under element hold-out, the no-B-information ablation, and split-conformal uncertainty under element shift. The last two are new; the first three are partial — closely related work exists and is cited in the manuscript.

## Citation

```bibtex
@misc{ali2026perovskite,
  author       = {Muhammad Kashan Ali},
  title        = {Benchmarking the Generalization and Extrapolation Limits of
                  Composition-Based Machine Learning Models for Perovskite
                  Formation Energy and Band-Gap Prediction},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/kashan1-debug/perovskite-band-gap-stability-benchmark}
}
```

Also see [`CITATION.cff`](CITATION.cff).

## Author

**Muhammad Kashan Ali** — Department of Chemistry, Bahauddin Zakariya University, Multan, Pakistan.

## Acknowledgements

The author designed the study, ran all experiments in Google Colab, and verified all numbers. Muse (Meta's AI assistant) helped build and audit the code, run diagnostics, and draft the manuscript under the author's direction; this assistance is disclosed in the manuscript. The author takes full responsibility for the work.

## License

Code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)). The manuscript text is shared under CC BY 4.0.
