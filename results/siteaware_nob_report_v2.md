# Perovskite v2 — Stage 3c no-B-information ablation report

No-B-info features = Magpie (132) + A-site physics (12) + X-site physics (12) + X one-hot (7) = 163 columns. Dropped: B one-hot (52) + B-site physics (12) + all 5 derived columns (tolerance_factor, octahedral_factor, rA_minus_rB, EN_A_minus_EN_B, EN_B_minus_EN_X) = 69 columns. No Ridge, tuned XGBoost only, leave-one-B-out only, seed 0.

Why it can still work: Magpie is composition-only, so it still carries B's contribution in aggregate, and with A and X given explicitly the model can work out B's part by subtraction.

## Feature groups

- Kept: Magpie 132, A physics 12, X physics 12, X one-hot 7 — total 163
- Dropped: B one-hot 52, B physics 12, derived 5 — total 69
- B-site markers left in kept set: [] (asserted in code)
- Leakage columns in no-B-info X: [] (asserted in code)

## Pooled headline — no-B-info XGBoost

- n_test=18928, R2=0.683, RMSE=0.084, MAE=0.064
- Per-family R2: mean=0.335, median=0.431, min=-0.632 (Ta), max=0.747 (Ni)

## Comparison — pooled B-out XGBoost R2 (all numbers read from disk)

feature_set   model  n_test  r2_pooled  rmse_pooled
     magpie XGBoost   18928   0.085721     0.141989
  siteaware XGBoost   18928   0.732794     0.076761
        nob XGBoost   18928   0.682593     0.083661

- source: magpie XGBoost pooled from predictions_siteaware_v2.parquet
- source: site-aware XGBoost pooled from predictions_siteaware_v2.parquet
- source: no-B-info XGBoost pooled from predictions_siteaware_nob_v2.parquet (this run)

## Interpretation

- B-physics adds on B-out

## GATE 3c

  [PASS] results_siteaware_nob_v2.csv: 52 rows (expected 52)
  [PASS] predictions_siteaware_nob_v2.parquet: 18928 predictions (expected 18928)
  [PASS] comparison table present: magpie, site-aware and no-B-info pooled values all read from disk
  [PASS] interpretation printed: B-physics adds on B-out

## Notes

- experiment column: 'siteaware_nob_b_out' for the full run, 'smoke' for test runs.
- Pooled R2 is computed once on all pooled test predictions; it is not the mean of per-family R2s.
- Stage 3b reference numbers (full Colab run): site-aware pooled B-out XGBoost 0.733, Ridge -0.193; Magpie-only 0.206 / 0.086. These are quoted here for context only — the comparison table above recomputes them from the files on disk.
- The 8-family diagnostic (Mg/Te/W/Cr/Ni/Al/B/Re) gave no-B-info pooled 0.586 vs site-aware 0.484 on those 8 families only. Do NOT quote the diagnostic as the Stage 3c result.

