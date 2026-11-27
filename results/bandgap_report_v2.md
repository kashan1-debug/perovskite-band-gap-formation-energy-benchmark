# Perovskite v2 — Stage 5 band-gap report

Magpie only (132 columns), tuned XGBoost only.
Experiment: 'corrected' — v1 used random K-fold, v2 uses GROUPED 5-fold CV on composition_group, so v2 numbers are NOT expected to match v1 exactly. Do not force them to match.
Band gap is the DFT (GLLBSC) gap, not the experimental gap.

## C1 — Classification, metal vs non-metal (grouped, per seed)

                            task   model   split  seed     n  threshold  roc_auc  accuracy  precision   recall       f1   pr_auc      tn   fp    fn   tp  n_pred_nonmetal  r2  rmse  mae experiment
classification_metal_vs_nonmetal XGBoost grouped     0 18928        0.5 0.919193  0.962912   0.726027 0.072109 0.131188 0.457500 18173.0 20.0 682.0 53.0             73.0 NaN   NaN  NaN  corrected
classification_metal_vs_nonmetal XGBoost grouped     1 18928        0.5 0.919034  0.962859   0.710526 0.073469 0.133169 0.456239 18171.0 22.0 681.0 54.0             76.0 NaN   NaN  NaN  corrected
classification_metal_vs_nonmetal XGBoost grouped     2 18928        0.5 0.918984  0.962754   0.674419 0.078912 0.141291 0.456477 18165.0 28.0 677.0 58.0             86.0 NaN   NaN  NaN  corrected

Mean across seeds: ROC-AUC 0.919, accuracy 0.963 (majority baseline 0.961 — never report accuracy alone), precision 0.704, recall 0.075, F1 0.135, PR-AUC 0.457.
v1 (random): ROC-AUC 0.917, accuracy 0.962, recall 0.057 (42/735 caught), precision 0.656 at threshold 0.5.

## Threshold analysis — pooled across seeds (mean OOF prob per row)

  0.1 | 0.337 | 0.789 | 0.472 | 1722
  0.2 | 0.462 | 0.616 | 0.528 | 981
  0.3 | 0.524 | 0.355 | 0.423 | 498
  0.5 | 0.694 | 0.080 | 0.144 | 85
  0.7 | 0.000 | 0.000 | 0.000 | 0
  0.9 | 0.000 | 0.000 | 0.000 | 0

## C2 — Regression, non-zero gaps only, n = 735 (grouped, per seed)

                  task   model   split  seed   n  threshold  roc_auc  accuracy  precision  recall  f1  pr_auc  tn  fp  fn  tp  n_pred_nonmetal       r2     rmse      mae experiment
regression_nonzero_gap XGBoost grouped     0 735        NaN      NaN       NaN        NaN     NaN NaN     NaN NaN NaN NaN NaN              NaN 0.560943 0.992824 0.761143  corrected
regression_nonzero_gap XGBoost grouped     1 735        NaN      NaN       NaN        NaN     NaN NaN     NaN NaN NaN NaN NaN              NaN 0.557128 0.997128 0.762530  corrected
regression_nonzero_gap XGBoost grouped     2 735        NaN      NaN       NaN        NaN     NaN NaN     NaN NaN NaN NaN NaN              NaN 0.555115 0.999391 0.765998  corrected

Mean across seeds: R2 0.558 +- 0.003, RMSE 0.996 +- 0.003, MAE 0.763 +- 0.003. n = 735 is small and grouped folds are noisy — report mean +- std.
v1 (random): R2 0.505, RMSE 1.054, MAE 0.802 (provisional).

## C0 — WARNING only: plain all-rows regression (never the headline)

                  task   model  split  seed     n  threshold  roc_auc  accuracy  precision  recall  f1  pr_auc  tn  fp  fn  tp  n_pred_nonmetal       r2     rmse      mae experiment
warning_plain_all_rows XGBoost random     0 18928        NaN      NaN       NaN        NaN     NaN NaN     NaN NaN NaN NaN NaN              NaN 0.287881 0.387904 0.105863  corrected

This looks OK only because 96.1% of gaps are zero. v1 tuned: R2 0.288. Never report it as a band-gap result.

## GATE 5 — corrected-experiment sanity checks

  [PASS] C1 ROC-AUC mean 0.919 (expected in [0.80, 0.97])
  [PASS] C1 recall at 0.5 mean 0.075 (expected < 0.20, v1 random was 0.057 — the model ranks well but misses most non-metals at 0.5)
  [PASS] C2 non-zero R2 mean 0.558 (expected in [0.0, 0.70])
  [PASS] C2 non-zero RMSE mean 0.996 (expected in [0.7, 1.4])
  [PASS] Headline is C1/C2 only — warning row exists, labelled 'warning_plain_all_rows', and was not used in any gate check

## Notes

- experiment column: 'corrected' for the full run, 'smoke' for test runs.
- Grouped CV is stricter than random CV: related compositions cannot share a fold, so grouped numbers are the honest ones.
- If GATE 5 failed, STOP — do not proceed to Stage 6. Find why the v2 numbers are outside the sanity ranges first.

