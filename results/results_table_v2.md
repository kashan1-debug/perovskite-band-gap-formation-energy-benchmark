# Perovskite v2 — Results table

Every number below comes from a saved v2 CSV read in Stage 7,
never typed by hand. N, split, model and seeds are stated
with each row. Magpie only (132 columns), e_form_per_atom,
unless stated otherwise.

## Study A — Random vs grouped CV (5-fold, seeds 0–2, n = 18,928)

| Split | Model | R² (mean ± std across seeds) |
|---|---|---|
| random | Mean | -0.000 ± 0.000 |
| random | Ridge | 0.385 ± 0.000 |
| random | RandomForest | 0.259 ± 0.002 |
| random | XGBoost | 0.431 ± 0.000 |
| grouped | Mean | -0.000 ± 0.000 |
| grouped | Ridge | 0.388 ± 0.000 |
| grouped | RandomForest | 0.475 ± 0.000 |
| grouped | XGBoost | 0.443 ± 0.000 |

Composition-only ceiling R² (pooled, Stage 1): 0.543.
Best CV R² (grouped Random Forest) is below the ceiling, as it must be.

## Study B — Leave-one-family-out, pooled headline (seed 0)

| Held out | Model | n_test | Pooled R² | Pooled RMSE | Pooled MAE |
|---|---|---|---|---|---|
| B-site | Ridge | 18928 | 0.206 | 0.132 | 0.102 |
| B-site | XGBoost | 18928 | 0.086 | 0.142 | 0.107 |
| X-anion | Ridge | 18928 | -9.958 | 0.492 | 0.148 |
| X-anion | XGBoost | 18928 | 0.375 | 0.117 | 0.090 |

X-out Ridge pooled R² is catastrophic (linear extrapolation to a
new anion fails); XGBoost survives X-out because trees do not
extrapolate. Mean per-family R² is NOT the headline — pooled is.

## Study C — Band gap (grouped 5-fold, tuned XGBoost)

- Classification, metal vs non-metal (n = 18,928, 735 non-metals): ROC-AUC 0.919, PR-AUC 0.457, at threshold 0.5 precision 0.704, recall 0.075.
- At threshold 0.2 (best F1): precision 0.462, recall 0.620, F1 0.529.
- Non-zero-gap regression (n = 735): R² 0.558, RMSE 0.996, MAE 0.763.
- WARNING only, plain all-rows regression: R² 0.288 — looks OK only because 96.1% of gaps are zero. Never a result.

## Study D — Conformal uncertainty (90% nominal, exact quantile)

| Test | n_test | q (eV/atom) | Coverage | Mean width | R² on test |
|---|---|---|---|---|---|
| random | 3787 | 0.185 | 0.899 | 0.370 | 0.423 |
| family-out Te | 364 | 0.187 | 0.923 | 0.374 | -0.142 |
| family-out W | 364 | 0.187 | 0.690 | 0.374 | -4.211 |
| family-out Mg | 364 | 0.190 | 0.945 | 0.381 | 0.423 |

Random coverage checks a 90% promise (exchangeable rows).
Family-out coverage is an observation under shift — no guarantee
applies. Coverage is NOT accuracy: W covers only 0.690 while
its point R² is -4.211; Te covers 0.923 while its point R² is
negative. Always report coverage AND width AND point R² together.

## Figures

- fig1_cv_r2.png — random vs grouped CV, ceiling line at 0.543.
- fig2_familyout_r2.png — pooled family-out; X-out Ridge (-9.96)
  is annotated, not plotted on the linear scale.
- fig3_parity_xgb.png — B-out parity, Ridge and XGBoost.
- fig4_distributions.png — e_form and band-gap distributions.
- fig5_bandgap_conformal.png — threshold curves and coverage.

