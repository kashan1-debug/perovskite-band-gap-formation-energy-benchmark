# Code — one pasteable Google Colab cell per file

Each file is **one Colab cell**: paste its whole contents into a single code
cell and run it. Run them in order, `00` → `12`. Every cell defines its own
output folder and fails loudly if Google Drive is not mounted; every cell is
safe to run twice (existing outputs are backed up, not overwritten silently).

`benchmark_v2.py` is the same pipeline as a standalone script for a local
run; the numbered cells are the authoritative, actually-run version.

## Setup — run first, in every fresh Colab session

| File | What it does |
|---|---|
| `00_setup_colab.txt` | **Setup A** — install the libraries (once per runtime, then restart). **Setup B** — imports and version checks. **Setup C** — mount Google Drive and set `OUT_DIR` to `MyDrive/perovskite/results_v2/`. |

Rules that cost a lost run if ignored: after a restart, run Setup B then
Setup C again — never Setup A. Installed packages survive a restart; the
Drive mount and `OUT_DIR` do not. A `requests` version warning
(google-colab wants 2.32.4, pymatgen-core wants ≥2.32.5) is harmless — keep
the 2.32.4 pin.

## Stages

| File | Stage | What it does | Main outputs |
|---|---|---|---|
| `01_dataset_audit.py` | 1 | Loads Castelli perovskites, checks shape, sites, A/B/X parsing, target definition, composition-degeneracy ceiling | `audited.parquet`, `audit_report.md` |
| `02_magpie_features.py` | 2 | Magpie featurisation (132 features) | `features.parquet` (18,928 × 150) |
| `03_random_grouped_cv.py` | 3 | Random + grouped-by-formula 5-fold CV, seeds 0–2 (Mean / Ridge / RF / tuned XGB) | `results_cv_v2.csv` |
| `04_leave_one_family_out.py` | 4 | Leave-one-B-out (52) and leave-one-X-out (7), Ridge + tuned XGB | `results_family_v2.csv`, `predictions_family_v2.parquet` |
| `05_band_gap.py` | 5 | Band-gap classification (grouped), threshold analysis, non-zero-gap regression (grouped) | `results_bandgap_v2.csv` |
| `06_conformal.py` | 6 | Exact split-conformal intervals: random split + held-out Te, W, Mg | `results_conformal_v2.csv` |
| `07_figures_tables.py` | 7 | Figures 1–5 and the results table, read from saved CSVs only | `figures/fig1–fig5`, `results_table_v2.md` |
| `08_site_aware.py` | 3b | Site-aware features (Magpie + site physics + t/μ + one-hots), leave-one-B-out | `results_siteaware_v2.csv`, `features_siteaware.parquet` |
| `09_no_b_ablation.py` | 3c | No-B-information ablation (drops all 69 B-site columns), leave-one-B-out | `results_siteaware_nob_v2.csv` |
| `10_shap.py` | 9a | SHAP on a random 80/20 split, site-aware vs Magpie-only | `results_shap_v2.csv`, `figures/shap_importance.png` |
| `11_shap_family_out.py` | 9a2 | SHAP on held-out Te, W, Mg (trained without that metal) | `results_shap_familyout_v2.csv`, `figures/shap_familyout.png` |
| `12_megnet.py` | 9b | Pretrained MEGNet (zero-shot), leave-one-B-out, per-family calibration | `results_megnet_v2.csv` |

`12_megnet.py` needs one extra install first: `!pip install -q megnet`
(TensorFlow is already in Colab), then the requests pin, restart, Setup B,
Setup C, then the cell. It works around two MEGNet 1.3.2 issues in-cell: the
renamed `BrunnerNN_real` import in pymatgen 2026, and the Keras-2 optimizer
that Keras 3 refuses (loads with `compile=False`).

## Environment

Python 3 · scikit-learn 1.6.1 · XGBoost 3.4.1 · matminer 0.10.1 ·
pymatgen 2026.10.24 · SHAP · MEGNet 1.3.2 / TensorFlow 2.21.0 (Stage 9b only).

## Rules (from `docs/workflow.md`)

- **Composition/structure only.** Never use DFT columns (`vbm`, `cbm`,
  `fermi level`, `mu_b`, `e_form`) as features.
- **Never merge on reduced formula.** A/B-swapped structures share a reduced
  formula but are different rows with different energies.
- **Report pooled R²** for family-out, not the mean of per-family R².
- **Never quote the random-split site-aware figure** (R² 0.90 with default
  XGBoost, 0.82 Ridge) as the result — the result is the family-out number.
