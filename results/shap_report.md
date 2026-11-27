# Perovskite v2 — Stage 9a SHAP feature-importance report

SHAP (SHapley Additive exPlanations) gives every feature a fair share of the credit for each prediction. Mean |SHAP| is how much a feature moves predictions on average, in eV/atom. Two tuned XGBoost models (n_estimators=100, max_depth=3, lr=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, seed 0), random 80/20 split seed 0: site-aware X (232 cols) and Magpie-only control (132 cols). SHAP computed with shap.TreeExplainer on a subsample of the test set (2000 rows, random_state=0).

## Honest limit

SHAP here is computed on a RANDOM split (interpolation), not on leave-one-B-out. It shows which features the model uses when it can interpolate. It is supporting evidence for the mechanism, not a direct test of transfer — the direct test stays Stages 3b/3c (site-aware pooled B-out R2 0.733, no-B-info 0.683, Magpie-only 0.086). Do not write 'SHAP proves transfer'.

## Test scores (random split — sanity only, not gated)

- site-aware: test R2 0.812, RMSE 0.064, MAE 0.050
- Magpie-only: test R2 0.440, RMSE 0.111, MAE 0.086

## SHAP identity check (base_value + sum(SHAP) == prediction)

- siteaware: base_value 0.294644, max error over 5 rows 4.65e-08 (tolerance 1e-06) — PASS
- magpie: base_value 0.294533, max error over 5 rows 1.49e-07 (tolerance 1e-06) — PASS

## Top 10 features — site-aware model

 rank                          feature     group  mean_abs_shap
    1                      B_mendeleev B_physics       0.019116
    2        MagpieData mean GSbandgap    magpie       0.018285
    3                      rA_minus_rB   derived       0.017314
    4                             B_EN B_physics       0.015719
    5 MagpieData mean SpaceGroupNumber    magpie       0.013983
    6                          B_r_ion B_physics       0.013303
    7                 tolerance_factor   derived       0.012180
    8                         B_r_atom B_physics       0.008032
    9                         A_r_atom A_physics       0.006793
   10     MagpieData maximum NUnfilled    magpie       0.005377

## Top 10 features — Magpie-only model

 rank                              feature  group  mean_abs_shap
    1            MagpieData mean GSbandgap magpie       0.019089
    2 MagpieData minimum Electronegativity magpie       0.011029
    3         MagpieData range GSvolume_pa magpie       0.010239
    4        MagpieData maximum NdUnfilled magpie       0.009622
    5            MagpieData minimum Number magpie       0.009272
    6         MagpieData minimum NUnfilled magpie       0.007724
    7            MagpieData maximum Number magpie       0.007673
    8            MagpieData mean NpValence magpie       0.007360
    9     MagpieData mean SpaceGroupNumber magpie       0.006897
   10         MagpieData maximum NUnfilled magpie       0.006677

## Per-group SHAP totals

    model     group  n_features  total_mean_abs_shap  mean_mean_abs_shap     share
   magpie    magpie         132             0.158343            0.001200       1.0
siteaware    magpie         132             0.083697            0.000634  0.398014
siteaware B_physics          12             0.067085            0.005590  0.319019
siteaware   derived           5             0.033914            0.006783  0.161273
siteaware A_physics          12             0.024379            0.002032   0.11593
siteaware  B_onehot          52             0.000995            0.000019  0.004731
siteaware  X_onehot           7             0.000217            0.000031  0.001032
siteaware X_physics          12             0.000000            0.000000       0.0

## Mechanism — which group carries the site-aware gain?

Magpie still carries the gain: Magpie takes 40% of the site-aware model's SHAP credit, vs 28% for site-resolution columns and 32% for B-site columns — the site-aware features add little on this random split, and the Stage 3b/3c family-out story is not explained by SHAP here.

Top group (site-aware model): magpie. Top feature: B_mendeleev.

## Notes

- experiment: random 80/20 split, seed 0, tuned XGBoost only.
- 'share' is the group's total mean |SHAP| divided by the model's total mean |SHAP|; 'mean' divides the total by the number of features in the group, so large groups are not flattered for size.
- Figure: figures/shap_importance.png (top 20 features + group totals).

