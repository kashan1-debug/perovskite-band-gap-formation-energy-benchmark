# Perovskite results — final table

Dataset: Castelli perovskites, N = 18928, 52 A x 52 B x 7 X = 18,928. Magpie-only features: 132.

## Formation energy per atom (eV/atom)

|                             |     R2 |   RMSE |   MAE |
|:----------------------------|-------:|-------:|------:|
| ('grouped', 'Mean')         | -0     |  0.148 | 0.113 |
| ('grouped', 'RandomForest') |  0.475 |  0.108 | 0.082 |
| ('grouped', 'Ridge')        |  0.388 |  0.116 | 0.089 |
| ('grouped', 'XGBoost')      |  0.443 |  0.111 | 0.085 |
| ('random', 'Mean')          | -0     |  0.148 | 0.113 |
| ('random', 'RandomForest')  |  0.258 |  0.128 | 0.094 |
| ('random', 'Ridge')         |  0.385 |  0.116 | 0.089 |
| ('random', 'XGBoost')       |  0.432 |  0.112 | 0.085 |

## Family-out (mean per-family R2)

|                                      |      R2 |
|:-------------------------------------|--------:|
| ('leave-one-B_site-out', 'Ridge')    |  -0.774 |
| ('leave-one-B_site-out', 'XGBoost')  |  -0.941 |
| ('leave-one-X_anion-out', 'Ridge')   | -12.112 |
| ('leave-one-X_anion-out', 'XGBoost') |   0.283 |

## Band gap

                             Task         Model     N                 R2               RMSE                 MAE            ROC_AUC           Accuracy
 classification_metal_vs_nonmetal XGBoost_tuned 18928                                                           0.9170790813989532 0.9622252747252747
      regression_nonzero_gap_only XGBoost_tuned   735 0.5054157197718878   1.05373645367208  0.8023539431322189                                      
WARNING_plain_all_rows_regression XGBoost_tuned 18928 0.2878814415171653 0.3879040989110763 0.10586289267999309                                      

## Key claims (do not overstate)
- Magpie-only R2 ceiling ~0.46 (within-group std 0.109 / overall std 0.149).
- Best random: tuned XGBoost R2 0.432. Best grouped: tuned Random Forest R2 0.475.
- Leave-one-B-out pooled R2: Ridge 0.206, XGBoost 0.086 — model does NOT generalise to a new B metal.
- Leave-one-X-out pooled R2: Ridge -9.63 (linear extrapolation fails), XGBoost 0.375.
- Band gap: classification ROC-AUC 0.917, but recall only 5.7% (42/735 non-metals found at 0.5 threshold). Accuracy 0.962 is misleading (baseline 0.961).
- Non-zero-gap regression (n=735): R2 0.505, RMSE 1.05 eV — provisional, small sample.
- Plain all-rows band-gap regression R2 0.288 is NOT a band-gap result — 96% of gaps are zero.