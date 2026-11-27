# Perovskite v2 — Stage 3b site-aware leave-one-B-out report

Site-aware features = Magpie (132) + one-hot B (52) + one-hot X (7) + per-site physics (36) + derived t/mu/differences (5) = 232 columns. No A_site one-hot. No DFT columns in X. No random CV in this stage — leave-one-B-out only, the hard test.

## Feature groups

- Magpie: 132 columns
- One-hot B: 52, one-hot X: 7
- Physics: 41 columns (12 per site x 3 + 5 derived: tolerance_factor, octahedral_factor, rA_minus_rB, EN_A_minus_EN_B, EN_B_minus_EN_X)
- Physics NaN cells: 0; ionic-radius fallbacks used: 0
- Ionic radius: audit charge convention (A smallest positive, B largest positive, X most negative, majority X element). atomic_radius used in place of covalent_radius (pymatgen has no covalent_radius attribute in this version).

## Pooled headline (lead with this, not mean per-family R2)

feature_set   model  n_test  r2_pooled  rmse_pooled  mae_pooled
     magpie   Ridge   18928   0.205610     0.132353    0.101563
     magpie XGBoost   18928   0.085721     0.141989    0.107224
  siteaware   Ridge   18928  -0.192716     0.162175    0.128273
  siteaware XGBoost   18928   0.732794     0.076761    0.060283

## Per-family R2 summary (secondary — context only)

feature_set   model   mean_r2  median_r2     min_r2   max_r2 worst_family best_family
     magpie   Ridge -0.774352  -0.469321  -5.103253 0.307712            W          Mg
     magpie XGBoost -0.941065  -0.606987  -4.641594 0.431287           Re          Mg
  siteaware   Ridge -1.592085  -0.704146 -12.252296 0.651110           Hg          Mg
  siteaware XGBoost  0.386570   0.496942  -1.516669 0.747087           Cr          Ni

## Control check — Magpie-only B-out vs Stage 4

  [PASS] magpie control Ridge: pooled 0.206 vs Stage 4 0.206 (diff -0.000)
  [PASS] magpie control XGBoost: pooled 0.086 vs Stage 4 0.086 (diff -0.000)

## GATE 3b — site-aware pooled B-out R2 > 0.25

  [INFO] siteaware Ridge: pooled B-out R2 -0.193 (criterion: > 0.25 to break the Magpie-only failure)
  [FAIL] siteaware XGBoost: pooled R2 0.733 — above 0.60, suspicious: a B identity column may have leaked.
  [FAIL] The 0.25 criterion cannot be judged — a leakage alarm fired above. Fix the leak first, do not claim a pass.

## Notes

- experiment column: 'siteaware_b_out' for the full run, 'smoke' for test runs.
- Pooled R2 is computed once on all pooled test predictions per (feature_set, model); it is not the mean of per-family R2s.
- Random-CV site-aware numbers from the audit (Ridge 0.82 / tuned XGBoost 0.76, default XGBoost 0.90) are INTERPOLATION only and must never be quoted as this stage's result.
- If GATE 3b failed, that is a result: even site-aware features do not transfer to a new B metal. Report it, do not hide it.

