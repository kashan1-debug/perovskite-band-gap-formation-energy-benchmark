# Version-1 results (baselines that version 2 reproduces)

These two files are the original v1 results. They are kept here so that
anyone can check the v2 reproduction against them; the manuscript reports
the v2 numbers.

| File | Contents |
|---|---|
| `results_cv_tuned.csv` | v1 random + grouped 5-fold CV, 24 rows (Cell 10). Random — Ridge 0.385, RF 0.259, XGB 0.431; grouped — Ridge 0.388, RF 0.475, XGB 0.443. Reproduced by v2 Stage 3 (`results_cv_v2.csv`) to ±0.001. |
| `results_bandgap.csv` | v1 band-gap results, 3 rows (Cell 12, repaired Cell 14). Classification ROC-AUC 0.917; non-zero-gap regression R² 0.505 (random K-fold, n=735); plain all-rows regression R² 0.288 (warning only — never report as a band-gap success). |

Do not mix these with the v2 files in `results/`: the v1 band-gap
regression used random splits, while v2 uses grouped splits (R² 0.558),
and the v1 classification had no threshold or precision–recall analysis.
