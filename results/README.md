# Results

Result tables produced by the cells in `code/`, saved to
`MyDrive/perovskite/results_v2/` when run in Colab. The small CSVs are
committed here; the large prediction files (`.parquet`) are not — regenerate
them with the corresponding stage cell, or ask the author.

## Files

| File | From | Contents | In repo |
|---|---|---|---|
| `results_cv_v2.csv` | Stage 3 | Random + grouped CV, 24 rows (4 models × 2 splits × 3 seeds) | yes |
| `results_family_v2.csv` | Stage 4 | Leave-one-B-out (52) + leave-one-X-out (7), 118 rows | yes |
| `results_bandgap_v2.csv` | Stage 5 | Band-gap classification, threshold analysis, non-zero-gap regression, 22 rows | yes |
| `results_conformal_v2.csv` | Stage 6 | Split-conformal: random + held-out Te, W, Mg, 4 rows | not yet |
| `results_siteaware_v2.csv` | Stage 3b | Site-aware leave-one-B-out, 208 rows | not yet |
| `results_siteaware_nob_v2.csv` | Stage 3c | No-B-information ablation, 52 rows | not yet |
| `results_megnet_v2.csv` | Stage 9b | Pretrained MEGNet, raw + calibrated, 52 rows | not yet |
| `results_shap_v2.csv` | Stage 9a | SHAP values, random split, 364 rows | not yet |
| `results_shap_groups_v2.csv` | Stage 9a | SHAP group shares, 8 rows | not yet |
| `results_shap_familyout_v2.csv` | Stage 9a2 | SHAP values on held-out Te/W/Mg, 696 rows | yes |
| `results_shap_familyout_groups_v2.csv` | Stage 9a2 | SHAP group shares under shift, 21 rows | yes |
| `shap_report.md` | Stage 9a | SHAP report (included here) | yes |

Files marked "not yet" are still on the author's Drive and are added as
they are sent. The large prediction files (`.parquet`) are not committed —
regenerate them with the corresponding stage cell, or ask the author.

Version-1 baseline results live in [`v1/`](v1/) with their own note —
they are the numbers v2 reproduces, kept separate so the two versions
are never mixed up.

Headline numbers (pooled leave-one-B-out R², formation energy per atom):
Magpie-only XGBoost 0.086 · Magpie-only Ridge 0.206 · site-aware XGBoost
0.733 · no-B-information XGBoost 0.683 · calibrated MEGNet 0.561 ·
composition-degeneracy ceiling 0.543. See the root README for the full table.
