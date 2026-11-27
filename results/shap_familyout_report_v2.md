# Perovskite v2 — Stage 9a2 SHAP on held-out B metals report

SHAP (SHapley Additive exPlanations) gives every feature a fair share of the credit for each prediction. Mean |SHAP| is how much a feature moves predictions on average, in eV/atom. For each of Te, W and Mg, one tuned XGBoost model (n_estimators=100, max_depth=3, lr=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, seed 0) was trained ONLY on rows with B != that metal, and SHAP was computed with shap.TreeExplainer on all 364 held-out rows of that metal (100 in SMOKE).

## Why B one-hot and B-physics are not leakage here

They describe the TEST crystals, whose B identity is known at prediction time — exactly as in Stage 3b. The held-out metal never appears in the training rows (asserted in code); its B one-hot column is all zero in training, so the model cannot have memorised that metal.

## Held-out transfer scores (sanity only, NOT gated)

- Te: held-out R2 0.529, RMSE 0.065, MAE 0.051, n_test 364
- W: held-out R2 0.226, RMSE 0.069, MAE 0.056, n_test 364
- Mg: held-out R2 0.512, RMSE 0.091, MAE 0.073, n_test 364

## SHAP identity check (base_value + sum(SHAP) == prediction)

- Te: base_value 0.294509, max error over 5 rows 1.36e-07 (tolerance 1e-06) — PASS
- W: base_value 0.297013, max error over 5 rows 6.15e-08 (tolerance 1e-06) — PASS
- Mg: base_value 0.293636, max error over 5 rows 8.82e-08 (tolerance 1e-06) — PASS

## Per-group SHAP totals (share = group total / family total)

family_B     group  n_features  total_mean_abs_shap  mean_mean_abs_shap    share       r2     rmse      mae  n_test
      Te    magpie         132             0.085430            0.000647 0.447986 0.528972 0.065122 0.050900     364
      Te B_physics          12             0.053571            0.004464 0.280921 0.528972 0.065122 0.050900     364
      Te A_physics          12             0.026210            0.002184 0.137445 0.528972 0.065122 0.050900     364
      Te   derived           5             0.024704            0.004941 0.129544 0.528972 0.065122 0.050900     364
      Te  B_onehot          52             0.000560            0.000011 0.002934 0.528972 0.065122 0.050900     364
      Te  X_onehot           7             0.000223            0.000032 0.001169 0.528972 0.065122 0.050900     364
      Te X_physics          12             0.000000            0.000000 0.000000 0.528972 0.065122 0.050900     364
       W B_physics          12             0.079134            0.006594 0.399176 0.226143 0.069066 0.055866     364
       W    magpie         132             0.067291            0.000510 0.339440 0.226143 0.069066 0.055866     364
       W A_physics          12             0.025361            0.002113 0.127928 0.226143 0.069066 0.055866     364
       W   derived           5             0.024727            0.004945 0.124730 0.226143 0.069066 0.055866     364
       W  B_onehot          52             0.000801            0.000015 0.004039 0.226143 0.069066 0.055866     364
       W  X_onehot           7             0.000716            0.000102 0.003613 0.226143 0.069066 0.055866     364
       W X_physics          12             0.000213            0.000018 0.001074 0.226143 0.069066 0.055866     364
      Mg    magpie         132             0.080814            0.000612 0.468496 0.512294 0.091216 0.072668     364
      Mg B_physics          12             0.044444            0.003704 0.257653 0.512294 0.091216 0.072668     364
      Mg A_physics          12             0.024898            0.002075 0.144340 0.512294 0.091216 0.072668     364
      Mg   derived           5             0.021430            0.004286 0.124232 0.512294 0.091216 0.072668     364
      Mg  B_onehot          52             0.000504            0.000010 0.002919 0.512294 0.091216 0.072668     364
      Mg  X_onehot           7             0.000407            0.000058 0.002360 0.512294 0.091216 0.072668     364
      Mg X_physics          12             0.000000            0.000000 0.000000 0.512294 0.091216 0.072668     364

## Top 10 features — held-out Te

 rank                          feature     group  mean_abs_shap
    1        MagpieData mean GSbandgap    magpie       0.015575
    2                      rA_minus_rB   derived       0.012298
    3                      B_mendeleev B_physics       0.012241
    4                          B_r_ion B_physics       0.009689
    5 MagpieData mean SpaceGroupNumber    magpie       0.009449
    6                             B_EN B_physics       0.009384
    7        MagpieData mean NpValence    magpie       0.008563
    8                 tolerance_factor   derived       0.008429
    9                           B_ionE B_physics       0.007122
   10                         A_r_atom A_physics       0.006431

## Top 10 features — held-out W

 rank                          feature     group  mean_abs_shap
    1                      B_mendeleev B_physics       0.037611
    2        MagpieData mean GSbandgap    magpie       0.015883
    3                      rA_minus_rB   derived       0.014552
    4                             B_EN B_physics       0.011369
    5                         B_r_atom B_physics       0.009938
    6                          B_r_ion B_physics       0.009652
    7                 tolerance_factor   derived       0.007732
    8                         A_r_atom A_physics       0.007707
    9 MagpieData mean SpaceGroupNumber    magpie       0.007221
   10     MagpieData maximum NUnfilled    magpie       0.006503

## Top 10 features — held-out Mg

 rank                          feature     group  mean_abs_shap
    1        MagpieData mean GSbandgap    magpie       0.017623
    2                      B_mendeleev B_physics       0.015294
    3                      rA_minus_rB   derived       0.013857
    4                          B_r_ion B_physics       0.011814
    5 MagpieData mean SpaceGroupNumber    magpie       0.011452
    6                             B_EN B_physics       0.009374
    7     MagpieData maximum NUnfilled    magpie       0.008050
    8                         A_r_atom A_physics       0.005862
    9                          A_r_ion A_physics       0.005178
   10                 tolerance_factor   derived       0.005093

## Mechanism — which group carries the transfer prediction?

Magpie still carries the transfer prediction on Te (held-out R2 0.529): Magpie takes 45% of the SHAP credit, vs 27% for site-resolution columns and 28% for B-site columns — the site-aware features add little even on a metal the model never saw, and the Stage 3b/3c family-out gain is NOT explained by which features the model leans on.

B-site columns carry the transfer prediction on W (held-out R2 0.226): B physics + B one-hot take 40% of the SHAP credit, vs 26% for site-resolution columns and 34% for Magpie — this DISAGREES with the Stage 3c ablation (no-B-info 0.683 vs site-aware 0.733), so the two tests must be reconciled before the paper claims a mechanism.

Magpie still carries the transfer prediction on Mg (held-out R2 0.512): Magpie takes 47% of the SHAP credit, vs 27% for site-resolution columns and 26% for B-site columns — the site-aware features add little even on a metal the model never saw, and the Stage 3b/3c family-out gain is NOT explained by which features the model leans on.

OVERALL: Magpie still dominates even under shift — averaged over Te, W, Mg, Magpie takes 42% of the SHAP credit (vs 40% on the Stage 9a random split) and site resolution only 27% (vs 28% on 9a). The Stage 3b/3c family-out gain is NOT explained by which features the model leans on — the paper must say so, and must not claim SHAP proves the mechanism.

## Comparison with Stage 9a (random split, site-aware model)

- Stage 9a shares: site-resolution 27.8%, B-site 32.4%, Magpie 39.8%.
- This stage, averaged over Te, W, Mg: site-resolution 26.5%, B-site 31.6%, Magpie 41.9%.
- Stage 3b/3c reference: site-aware pooled B-out R2 0.733, no-B-info 0.683, Magpie-only 0.086.

## Notes

- experiment: leave-one-B-out SHAP, tuned XGBoost only, seed 0.
- 'share' is the group's total mean |SHAP| divided by the family's total mean |SHAP|; 'mean' divides the total by the number of features in the group, so large groups are not flattered for size.
- Held-out R2 values are printed for context only and are not pass/fail: W is expected to be low (Stage 4 Magpie B-out W R2 -5.10; site-aware 0.226 in Stage 3b).
- Figure: figures/shap_familyout.png (group shares per family).

