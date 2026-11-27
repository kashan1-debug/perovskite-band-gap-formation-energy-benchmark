# Results

Result tables produced by the cells in `code/`, saved to
`MyDrive/perovskite/results_v2/` when run in Colab. The small CSVs are
committed here; the large prediction files (`.parquet`) are not — regenerate
them with the corresponding stage cell, or ask the author.

## Files

| File | From | Contents |
|---|---|---|
| `results_cv_v2.csv` | Stage 3 | Random + grouped CV, 24 rows (4 models × 2 splits × 3 seeds) |
| `results_family_v2.csv` | Stage 4 | Leave-one-B-out (52) + leave-one-X-out (7), 118 rows |
| `results_bandgap_v2.csv` | Stage 5 | Band-gap classification, threshold analysis, non-zero-gap regression, 22 rows |
| `results_conformal_v2.csv` | Stage 6 | Split-conformal: random + held-out Te, W, Mg, 4 rows |
| `results_siteaware_v2.csv` | Stage 3b | Site-aware leave-one-B-out, 208 rows |
| `results_siteaware_nob_v2.csv` | Stage 3c | No-B-information ablation, 52 rows |
| `results_megnet_v2.csv` | Stage 9b | Pretrained MEGNet, raw + calibrated, 52 rows |
| `results_shap_v2.csv` | Stage 9a | SHAP values, random split, 364 rows |
| `results_shap_groups_v2.csv` | Stage 9a | SHAP group shares, 8 rows |
| `results_shap_familyout_v2.csv` | Stage 9a2 | SHAP values on held-out Te/W/Mg, 696 rows |
| `results_shap_familyout_groups_v2.csv` | Stage 9a2 | SHAP group shares under shift, 21 rows |
| `shap_report.md` | Stage 9a | SHAP report (included here) |

Headline numbers (pooled leave-one-B-out R², formation energy per atom):
Magpie-only XGBoost 0.086 · Magpie-only Ridge 0.206 · site-aware XGBoost
0.733 · no-B-information XGBoost 0.683 · calibrated MEGNet 0.561 ·
composition-degeneracy ceiling 0.543. See the root README for the full table.
