# Perovskite v2 — Stage 6 conformal uncertainty report

Exact split conformal, 90% nominal, tuned XGBoost, Magpie only.
Quantile is the exact order statistic: sort calibration residuals,
k = ceil((n_cal + 1) * 0.90), q = r[k-1]. No np.quantile.

## Results

         test   model  n_train  n_cal  n_test        q  coverage  mean_width   r2_test  rmse_test  mae_test  metal_in_train  metal_in_cal   experiment
       random XGBoost    11356   3785    3787 0.184789  0.898600    0.369579  0.423064   0.111660  0.085078              -1            -1 reproduction
family-out Te XGBoost    11138   3712     364 0.187040  0.923077    0.374081 -0.142449   0.101420  0.079275               0             0    corrected
 family-out W XGBoost    11138   3712     364 0.186762  0.689560    0.373525 -4.210520   0.179214  0.152408               0             0    corrected
family-out Mg XGBoost    11138   3712     364 0.190296  0.945055    0.380592  0.423077   0.099209  0.079100               0             0    corrected

## Framing

Random coverage is an empirical check of a 90% promise under
exchangeability. Family-out coverage is an empirical observation
under domain shift, where the guarantee does not apply. Coverage
is NOT accuracy: a wide interval can cover while point
predictions are worse than the mean (v1 Te: coverage 0.948,
R2 on test -0.086). Always report coverage AND width AND
point R2 together.

## v1 comparison

- Random (reproduction): v1 coverage 0.900, mean width 0.368,
  R2 on test 0.438, N_test 3,787.
- Te (corrected, NOT a reproduction): v1 coverage 0.948, mean
  width 0.381, R2 on test -0.086 — but v1 Te was LEAKED
  (trained on ~218 Te rows, tested on all 364 Te). v2 trains
  and calibrates on B != Te only, so do not expect an exact
  match. W (hard failure) and Mg (good case) are new in v2.

## GATE 6

  [PASS] random coverage 0.899 in [0.88, 0.92] (v1: 0.900)
  [PASS] random mean width 0.370 in [0.25, 0.50] (v1: 0.368)
  [PASS] family-out Te: metal-in-train=0, metal-in-cal=0 (both must be 0)
  [PASS] family-out W: metal-in-train=0, metal-in-cal=0 (both must be 0)
  [PASS] family-out Mg: metal-in-train=0, metal-in-cal=0 (both must be 0)
  [PASS] columns unified (one fixed column set)

## Notes

- experiment column: 'reproduction' for the random row,
  'corrected' for the family-out rows.
- metal_in_train / metal_in_cal are -1 for the random row
  (not applicable — no metal is held out); 0 for every
  family-out row (asserted in code).
- If GATE 6 failed on random coverage, the quantile code is
  wrong — stop. Family-out coverages are reported, not gated.

