# Perovskite v2 — Stage 3 random + grouped CV report

Magpie only (132 columns), e_form_per_atom, 5-fold, seeds 0, 1, 2.
Models are tuned configs only — no defaults.
RandomForest here: n_estimators=100, max_depth=10 (the v1 Cell 10 tuned config — random R2 0.258, grouped 0.475).
Grouped folds are identical across seeds (GroupKFold has no shuffle) — expected.

## Results (mean +- std across folds, per seed)

  split        model  seed     n   r2_mean   r2_std  rmse_mean  rmse_std  mae_mean  mae_std   experiment
 random         Mean     0 18928 -0.000048 0.000061   0.148491  0.001393  0.113199 0.001153 reproduction
 random         Mean     1 18928 -0.000111 0.000109   0.148488  0.001819  0.113202 0.001480 reproduction
 random         Mean     2 18928 -0.000153 0.000137   0.148494  0.001318  0.113199 0.000741 reproduction
 random        Ridge     0 18928  0.384584 0.014737   0.116465  0.000539  0.088901 0.000643 reproduction
 random        Ridge     1 18928  0.384607 0.012963   0.116465  0.001453  0.088956 0.001064 reproduction
 random        Ridge     2 18928  0.384748 0.008029   0.116460  0.000664  0.088931 0.000387 reproduction
 random RandomForest     0 18928  0.261269 0.020138   0.127601  0.001195  0.093649 0.000723 reproduction
 random RandomForest     1 18928  0.260148 0.028756   0.127670  0.001805  0.093687 0.001143 reproduction
 random RandomForest     2 18928  0.256546 0.030206   0.127987  0.002051  0.093851 0.001024 reproduction
 random      XGBoost     0 18928  0.431392 0.011686   0.111953  0.000515  0.085437 0.000411 reproduction
 random      XGBoost     1 18928  0.431076 0.015303   0.111974  0.001401  0.085506 0.001061 reproduction
 random      XGBoost     2 18928  0.431149 0.009589   0.111981  0.000969  0.085473 0.000672 reproduction
grouped         Mean     0 18928 -0.000121 0.000166   0.148494  0.001313  0.113200 0.001295 reproduction
grouped         Mean     1 18928 -0.000121 0.000166   0.148494  0.001313  0.113200 0.001295 reproduction
grouped         Mean     2 18928 -0.000121 0.000166   0.148494  0.001313  0.113200 0.001295 reproduction
grouped        Ridge     0 18928  0.388026 0.008811   0.116150  0.000766  0.088652 0.000578 reproduction
grouped        Ridge     1 18928  0.388026 0.008811   0.116150  0.000766  0.088652 0.000578 reproduction
grouped        Ridge     2 18928  0.388026 0.008811   0.116150  0.000766  0.088652 0.000578 reproduction
grouped RandomForest     0 18928  0.474996 0.009620   0.107578  0.000909  0.081842 0.000859 reproduction
grouped RandomForest     1 18928  0.474155 0.008988   0.107666  0.000958  0.081880 0.000857 reproduction
grouped RandomForest     2 18928  0.474653 0.008161   0.107616  0.000889  0.081826 0.000819 reproduction
grouped      XGBoost     0 18928  0.443268 0.008657   0.110784  0.000979  0.084696 0.000846 reproduction
grouped      XGBoost     1 18928  0.443703 0.008365   0.110741  0.000869  0.084660 0.000732 reproduction
grouped      XGBoost     2 18928  0.442935 0.008310   0.110817  0.000893  0.084713 0.000775 reproduction

## GATE 3 — reproduction vs v1 Cell 10 tuned

  [PASS] random Mean: v2 -0.000 vs v1 0.000 (diff -0.000)
  [PASS] random Ridge: v2 0.385 vs v1 0.385 (diff -0.000)
  [PASS] random RandomForest: v2 0.259 vs v1 0.258 (diff +0.001)
  [PASS] random XGBoost: v2 0.431 vs v1 0.432 (diff -0.001)
  [PASS] grouped Mean: v2 -0.000 vs v1 0.000 (diff -0.000)
  [PASS] grouped Ridge: v2 0.388 vs v1 0.388 (diff +0.000)
  [PASS] grouped RandomForest: v2 0.475 vs v1 0.475 (diff -0.000)
  [PASS] grouped XGBoost: v2 0.443 vs v1 0.443 (diff +0.000)

## Notes

- experiment column: 'reproduction' for the full run, 'smoke' for test runs.
- Pooled composition-only ceiling R2 = 0.543 (Stage 1). No Magpie-only model should beat it.
- If GATE 3 failed, STOP — do not proceed to Stage 4. Find why v2 differs from v1 first.

