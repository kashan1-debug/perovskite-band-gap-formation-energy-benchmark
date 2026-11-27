# Results

Result tables produced by the cells in `code/`, saved to
`MyDrive/perovskite/results_v2/` when run in Colab. The small CSVs are
committed here; the large prediction files (`.parquet`) are not — regenerate
them with the corresponding stage cell, or ask the author.

## Files

All v2 result tables and stage reports are now in this folder.

| File | From | Contents |
|---|---|---|
| `results_cv_v2.csv` | Stage 3 | Random + grouped CV, 24 rows (4 models × 2 splits × 3 seeds) |
| `results_family_v2.csv` | Stage 4 | Leave-one-B-out (52) + leave-one-X-out (7), 118 rows |
| `results_bandgap_v2.csv` | Stage 5 | Band-gap classification, threshold analysis, non-zero-gap regression, 22 rows |
| `results_conformal_v2.csv` | Stage 6 | Split-conformal: random + held-out Te, W, Mg, 4 rows |
| `results_siteaware_v2.csv` | Stage 3b | Site-aware leave-one-B-out, 208 rows (includes the Magpie control) |
| `results_siteaware_nob_v2.csv` | Stage 3c | No-B-information ablation, 52 rows |
| `results_megnet_v2.csv` | Stage 9b | Pretrained MEGNet, raw + calibrated, 52 rows |
| `results_shap_v2.csv` | Stage 9a | SHAP values, random split, 364 rows |
| `results_shap_groups_v2.csv` | Stage 9a | SHAP group shares, 8 rows |
| `results_shap_familyout_v2.csv` | Stage 9a2 | SHAP values on held-out Te/W/Mg, 696 rows |
| `results_shap_familyout_groups_v2.csv` | Stage 9a2 | SHAP group shares under shift, 21 rows |
| `shap_report.md` | Stage 9a | SHAP report (same file as `shap_report_v2.md` on Drive) |
| `results_table_v2.md` | Stage 7 | The results table from the manuscript, checked against the CSVs |

Stage reports (one per stage, written by the cell that ran it):
`audit_report.md` (Stage 1), `features_report.md` (Stage 2),
`cv_report_v2.md` (Stage 3), `family_report_v2.md` (Stage 4),
`bandgap_report_v2.md` (Stage 5), `conformal_report_v2.md` (Stage 6),
`siteaware_report_v2.md` (Stage 3b), `siteaware_nob_report_v2.md` (Stage 3c),
`shap_familyout_report_v2.md` (Stage 9a2).
`column_list.txt` and `column_list_features.txt` list the columns of the
audited dataset and the Magpie feature frame.

The large prediction and feature files (`.parquet`) are not committed —
regenerate them with the corresponding stage cell, or ask the author.

Version-1 baseline results live in [`v1/`](v1/) with their own note —
they are the numbers v2 reproduces, kept separate so the two versions
are never mixed up.

Headline numbers (pooled leave-one-B-out R², formation energy per atom):
Magpie-only XGBoost 0.086 · Magpie-only Ridge 0.206 · site-aware XGBoost
0.733 · no-B-information XGBoost 0.683 · calibrated MEGNet 0.561 ·
composition-degeneracy ceiling 0.543. See the root README for the full table.
