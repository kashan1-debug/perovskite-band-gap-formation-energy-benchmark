# Perovskite v2 — Complete Workflow (write this FIRST, build it SECOND)

**Purpose of this file:** the whole workflow, locked before any code is
rebuilt, so we do not make the same mistakes again. If a step is not in
this file, do not add it mid-run. If a check in this file fails, STOP
at that gate — do not go on to modelling.

**v1 is locked and kept as-is.** The original notebook is
`Untitled11.ipynb` (the v1 Colab run, 2026-11-22). v2 is a clean
rebuild. Do not patch v1. Do not mix v1 numbers into v2 tables
without saying which version they came from.

**Working title:** Benchmarking the Generalization and Extrapolation
Limits of Composition-Based Machine Learning Models for Perovskite
Formation Energy and Band-Gap Prediction

**One-sentence question:** If a model has never seen a given B-site
metal or X anion, can it still predict a perovskite's formation
energy and band gap — and can it tell us when it should not be
trusted?

---

## 0. Locked decisions — do not change these mid-run

These are decisions, not guesses. Changing one changes the results,
so any change needs a new version note and a re-run from the stage
it affects.

| # | Decision | Why, in one line |
|---|---|---|
| D1 | Dataset is **Castelli only**, `castelli_perovskites` from matminer, N = 18,928. No MatBench merge, ever. | Merging on reduced formula collapses A/B-swapped crystals and silently assigns wrong gaps. MatBench `e_form` is identical, same row order, so it adds nothing. |
| D2 | Target is **e_form_per_atom = e_form / nsites**, nsites = 5 for every row. Report in **eV/atom**. Original `e_form` is per 5-atom cell. **Verify first in Stage 1:** check the source definition and that every structure really has 5 sites before dividing. Do not lock the division until that check passes. | Matbench docs: "Heat of formation of the entire 5-atom perovskite cell, in eV" (RPBE GGA-DFT). Castelli is the source of that dataset, same 18,928 rows in the same order, head `e_form` values matched row by row in Cell 5. Live browser check 2026-11-25 confirmed the Matbench wording on the official matminer page; the Castelli entry alone just says "heat of formation in eV" and does not state per-cell. Cite this in the paper. Oxygen reference state is water vapour, not O2 — say so in Methods. |
| D3 | Keep **all 18,928 rows**. Do NOT deduplicate to 9,646. | Deduplicating would delete the polymorph story and break comparability with MatBench. Duplicates are the finding, not dirt to clean. Subject to Stage 1 source-integrity check: if rows are corrupt/missing, that is reported, not silently dropped. |
| D4 | Three IDs, always: `formula_ordered` (Castelli formula, A-B-X order), `formula_reduced` (reduced formula, alphabetical), `composition_group` (= `formula_reduced`, the grouping key). | Ordered formula identifies one crystal. Reduced formula identifies one composition, which may have several crystals with different energies. |
| D5 | A/B/X come from **structure site order** only: site 0 = A, site 1 = B, sites 2–4 = X. Never from atomic radius. **Validate against the structures in Stage 1 before trusting it:** check site counts, site species, and ordered-formula agreement on all rows, not just the first 20. | Radius rule agreed with site order on only 14.0% of rows. Site order: 0 unparseable, 0 true mismatches in v1. |
| D6 | Features, Stage 1: **Magpie only, 132 columns**. No DFT columns in X, ever. | Magpie is composition-only. That is the point of Study A. Site-aware features are a *separate* experiment (Stage 3b), never mixed into the Magpie-only tables. |
| D7 | **No "stable" claim.** No energy-above-hull data exists here. | Only 74/18,928 rows have negative e_form_per_atom. Formation energy alone is not stability. |
| D8 | Band gap is **DFT (GLLBSC/PBE) gap, not experimental gap**. Say so every time. | PBE/GLLBSC gaps are systematically biased vs experiment. Every claim is "predicting the DFT target". |
| D9 | Seeds are **0, 1, 2**. Folds = 5. Every split is saved to disk. Acceptable as initial seeds; this is limited statistical evidence — report mean ± std and say so, do not claim tight uncertainty from 3 seeds. | A result that cannot be re-run with the same split is not a result. |
| D10 | Every results file has **one fixed set of columns, set before the first row is written**. Never append rows of different shapes to one CSV. | This exact bug broke Cell 14 (5-column classification rows + 6-column regression rows in one file). |
| D11 | Every cell is **safe to run twice** (idempotent). Re-running must keep the same rows, and must save a backup before overwriting a file. | The morning Cell 14 repair would have wiped `results_bandgap.csv` to 0 rows on a second run. Patched version saves `results_bandgap_broken_backup.csv` first and keeps 8-column rows as-is. |
| D12 | Every Colab session starts with **Setup B, then Setup C**. Never Setup A after a restart. | Setup A installs + pins requests, then a restart is needed. Fresh sessions forget `OUT_DIR`, `df`, `magpie_cols` — that caused the 2026-11-25 `NameError: OUT_DIR is not defined`. All v2 cells reload from Drive and must not depend on memory from an old session. |

### Safeguard — reproduction vs corrected experiment

The ±0.02–0.03 tolerance in the gates below applies **only to a
controlled reproduction**: same data, same target, same split
method, same model config. If we fix leakage, duplicate handling,
data splitting or the target, a metric can legitimately change by
more than that — that is a corrected experiment, not a failed
reproduction. Every changed number must be labelled in the log
as either `reproduction` (same method, should match v1) or
`corrected` (method changed, new number expected), with the
reason for the change written next to it. Never force a corrected
number back to the v1 value to "pass" a gate.

---

## 1. The four studies — keep them separate

Do not let one study's numbers leak into another study's claim.

- **Study A — Composition-level prediction.** Can Magpie-only models
  predict e_form_per_atom for familiar compositions? Splits: random
  5-fold, grouped-by-composition 5-fold. This is interpolation.
- **Study B — Chemical extrapolation.** Can the same models predict a
  B metal or X anion they never saw? Splits: leave-one-B-out (52),
  leave-one-X-out (7). This is the paper's main claim.
- **Study C — Band-gap screening.** (1) Metal vs non-metal
  classification, gap > 0. (2) Regression only on the 735 non-zero
  gaps. Both with grouped validation. Secondary analysis, not the
  headline.
- **Study D — Uncertainty and reliability.** Split conformal, 90%
  nominal. Random test (does the promise hold?) and Te family-out
  test (does it hold under shift?). Report coverage AND width AND
  point R² together — coverage alone is not accuracy.

---

## 2. Files and folders (Drive, so a disconnect loses nothing)

```
perovskite/results_v2/
  raw_castelli.parquet        # Castelli only, structure dropped before save
  audited.parquet             # + formula_ordered / formula_reduced /
                              #   composition_group / A_site / B_site /
                              #   X_anion / A_family / e_form_per_atom
  features.parquet            # audited + 132 Magpie columns (check shape!)
  splits/                     # saved split indices, one file per split
    random_seed0.json, grouped_seed0.json, ...
  results_cv.csv              # Stage 3, one fixed column set
  results_family.csv          # Stage 4, one fixed column set
  results_bandgap.csv         # Stage 5, one fixed column set
  results_conformal.csv       # Stage 6, one fixed column set
  results_table.md            # generated from the CSVs, never typed by hand
  figures/                    # fig1..fig6, 300 dpi PNG + PDF
  column_list.txt             # every column of features.parquet, one per line
```

**Parquet rule:** drop `structure` and `composition` (object columns)
before every `to_parquet`. PyArrow cannot save pymatgen objects.
Check the saved file opens again before going on.

**CSV rule:** one `DataFrame`, one `to_csv`, unified columns.
`append_row` must assert the row matches the file's columns.

---

## 3. Stage 0 — Setup (every fresh session)

1. Open a blank Colab notebook.
2. Setup A — install only if this runtime has never had it:
   `matminer pymatgen xgboost scikit-learn shap pandas numpy
   matplotlib`, then `requests==2.32.4`, then **Runtime → Restart**.
3. Setup B — imports + versions (sklearn 1.6.1, xgboost 3.4.1,
   matminer 0.10.1, pymatgen via `importlib.metadata`, never
   `pymatgen.__version__`) + SrTiO3 test + Magpie ready test.
4. Setup C — mount Drive, `OUT_DIR =
   /content/drive/MyDrive/perovskite/results_v2`, test-save file.

**Gate 0:** Setup B prints the four versions and both tests pass.
Setup C prints `OUT_DIR` and the test file reads back. If either
fails, stop. Do not run later cells.

---

## 4. Stage 1 — Dataset audit (the foundation — nothing else runs first)

**Purpose in plain words:** prove the data, the target and the
row alignment are right, before any model touches them.

Steps:
1. Load `castelli_perovskites` only. Print shape, columns, dtypes.
   Expected: (18,928, 10), columns include `formula`, `e_form`,
   `gap gllbsc`, `structure`, `vbm`, `cbm`.
2. Derive: `nsites` from structure (expected: all 5),
   `e_form_per_atom = e_form / nsites`,
   `band_gap` = rename of `gap gllbsc`,
   `formula_ordered` = Castelli `formula`,
   `formula_reduced` = `structure.composition.reduced_formula`,
   `composition_group` = `formula_reduced`.
3. Parse A/B/X from site order. Print unparseable count
   (expected 0) and ordered-formula mismatch count (expected 0).
   A==B diagonal rows (e.g. BiBiON2) are not mismatches — note them.
4. EDA: missing values per column (expected 0), e_form_per_atom
   mean/std/median/min/max, band_gap == 0 count (expected 18,193),
   > 0 count (expected 735), in 1.3–1.6 eV (expected 77),
   `gap is direct` True/False counts.
5. Duplicate audit — the key table:
   - unique ordered formulas (expected 18,928),
   - unique reduced formulas (expected 9,646),
   - groups with >1 row (expected 9,282),
   - mean within-group std of e_form_per_atom (v1: 0.109),
   - overall std (v1: 0.149),
   - composition-only ceiling R² = 1 − (pooled within-group
     variance / total variance), pooled variance = sum of
     squared deviations from group means / N (all rows).
     **Headline ceiling is the pooled value, ≈ 0.543** — that is
     the true group-mean ceiling. The mean-std cross-check
     1 − (mean within-group std / overall std)² ≈ 0.46
     reproduces the Colab Diagnostic 2 number, but it is NOT
     the ceiling: it squares the mean of the stds instead of
     averaging the variances. Report both, headline pooled.
     (Corrected 2026-11-25 during Stage 1 testing: the earlier
     workflow wording headlined 0.46; the Stage 1 cell's pooled
     formula is the correct one. Do NOT divide pooled variance
     by sum(n−1) — that gives a wrong ceiling of ~0.07.)
   - max within-group range (v1 audit: 0.728 eV/atom).
6. Alignment check: formulas, e_form and indices match row by row
   between the loaded frame and the saved frame. Print True/False.
   If False, stop — this was the Cell 10 suspect, and it was
   ruled out in v1, but it must be re-checked in v2.
7. Save `audited.parquet` (structure dropped) and `column_list.txt`.

**Gate 1 — STOP if any of these fail:**
- shape is not (18,928, …), nsites not all 5, missing > 0,
- band_gap zero count is not 18,193 / non-zero not 735,
- unique ordered formulas ≠ 18,928 or unique reduced ≠ 9,646,
- A/B/X unparseable > 0 or true mismatch > 0,
- alignment check is False.
- If the gate fails, do not featurise. Fix the load first.

**Expected numbers to reproduce (v1):** e_form_per_atom mean
0.294, std 0.149, median 0.272, min −0.128, max 1.032; non-zero
gap mean 1.824, median 1.4, max 7.0. If v2 differs by more than
rounding, the load is wrong, not the model.

---

## 5. Stage 2 — Features (Magpie only)

1. Featurise with `ElementProperty.from_preset('magpie')` on the
   composition. Expected: 132 columns, 0 NaN, ~2–3 min in Colab.
2. **Leakage assert, before any model:** X must be Magpie columns
   only. Assert none of these is in X: `vbm`, `cbm`, `fermi level`,
   `fermi width`, `mu_b`, `e_form`, `band_gap`, `e_form_per_atom`.
   `vbm`/`cbm` define the gap directly — including them is cheating.
3. Save `features.parquet` (structure/composition dropped) and
   print its shape + column list.
4. **Investigate the 148 vs 150 discrepancy NOW, not in the paper
   draft:** v1 expected (18,928, 150), Colab printed
   (18,928, 148), Magpie still 132. Print every non-Magpie column,
   compare with the expected 18 original columns, name the 2
   missing columns in the log, and save `column_list.txt`. If a
   needed column (B_site, e_form_per_atom) is missing, stop.
   If the 2 missing columns are harmless duplicates, record that
   and update the expected shape to 148 in this file — do not
   silently change comments to match the file.

**Gate 2:** Magpie = 132, NaN = 0, leakage list = [], B_site and
e_form_per_atom present, saved file re-opens with the same shape.

---

## 6. Stage 3 — Study A: random + grouped CV (reproduce v1 first)

**Models (tuned configs only — defaults scored below the mean in v1
and must never be shipped again):**
- Mean (DummyRegressor) — baseline, R² ≈ 0.
- Ridge — with StandardScaler, alpha as in v1.
- Random Forest — depth 10, 100 trees (the v1 Cell 10 TUNED
  config: random R² 0.258, grouped R² 0.475. Corrected
  2026-11-25: the workflow first said depth 8 /
  min_samples_leaf 5, which was a separate VM audit diagnostic,
  not the v1 Cell 10 config — Stage 3 now uses depth 10, 100 trees).
- XGBoost — depth 3, 100 trees, lr 0.05, subsample 0.8,
  colsample_bytree 0.8, reg_lambda 5.

**Tuning rule (fixes the review's Issue 6):** the configs above
were chosen in v1 after seeing a diagnostic fold. In v2, either
(a) re-select them by **inner 3-fold CV on training data only**,
inside each outer fold, and report the selected values, or
(b) keep them fixed and state in the paper, in Methods, that they
were tuned on a diagnostic fold in v1 and are *not* re-tuned on
the v2 test data. Option (a) is stricter and preferred for
XGBoost/Ridge; for Random Forest, option (b) is acceptable *if*
stated, because a full nested RF run is 1+ hour in Colab.
Do not select a config by looking at the final test score and
then reporting that score as if it were untouched.

**Splits:** random 5-fold and grouped-by-`composition_group`
5-fold, seeds 0, 1, 2. Save every split index to `splits/`.
Grouped seeds will be identical across seeds (GroupKFold has no
shuffle) — that is expected, not a bug.

**Metrics per row:** Split, Model, Seed, Fold, N_train, N_test,
R², RMSE, MAE. Report mean ± std across seeds/folds. Save one
row per fold as soon as it finishes (flush to Drive).

**Gate 3 — reproduction check:** v2 Magpie-only means must be
close to v1 (tolerance ±0.02 R², same ordering):
random — Ridge 0.385, RF 0.258, XGB 0.432; grouped — Ridge 0.388,
RF 0.475, XGB 0.443. If v2 is far off, the data or features
changed — stop and find why, do not explain it away.

### Stage 3b — Site-aware experiment (SEPARATE, after 3 passes)

Only after Stage 3 reproduces v1. Features: Magpie + one-hot
A/B/X + tolerance factor t + octahedral factor μ (from
`Element.ionic_radii`, charge convention documented in the
audit: A = smallest positive charge, B = largest positive
charge, X = most negative; 0 NaN expected). **Never** one-hot
alone as a headline — one-hot is identity, it cannot transfer
to a new B. Report random CV only as interpolation, headline
v1-audit numbers for orientation: Ridge 0.82 / tuned XGB 0.76,
**never** default XGB 0.90. Then run Stage 4 on the site-aware
features too — that is the real test. If site-aware B-out does
not beat Magpie-only B-out, say so.

**Result recorded 2026-11-25 (full Colab run, 208 rows):**
site-aware pooled B-out XGBoost R² **0.733** (RMSE 0.077,
MAE 0.060; per-family mean 0.387, median 0.497, 46/52
positive, best Ni 0.747, worst Cr −1.517) vs Magpie-only
0.086; site-aware Ridge **−0.193**, worse than Magpie Ridge
0.206 — the gain is XGBoost-specific. The cell's pre-set
leak alarm (>0.60) fired and GATE 3b first printed FAIL.
**Correction recorded 2026-11-25 (verified in VM, no re-run
needed — the 208 rows stand):** four diagnostics on 8
families (Mg, Te, W, Cr, Ni, Al, B, Re) found no leak —
baseline reproduced Colab exactly; dropping the B one-hot
barely moved the score (0.484 → 0.460 pooled on the 8);
leave-one-element-out (also dropping A == held-out from
training) held at 0.413; feeding *wrong* B-physics values
destroyed the score (0.251, W → −2.112). The 0.60 alarm
was set before seeing data and was too low — it is retired.
**GATE 3b corrected to PASS: 0.733 > 0.25.** Open caveat
before the paper claims B-physics is the key: on the
8-family subset a no-B-info variant scored 0.586 vs 0.484
for full site-aware — B-physics helps some families (W, Re)
and hurts others (Cr). Run that variant on all 52 families
before writing the mechanism story.

---

## 7. Stage 4 — Study B: leave-one-family-out

**Splits:** leave-one-B-out (52 families, 364 rows each),
leave-one-X-out (7 families, 2,704 rows each). Models: Ridge +
tuned XGBoost only (RF would take >1 hour — state that, do not
run it half-finished). A_family "other" (14,560 rows) is too
large/coarse to leave out as one family — do not do it.

**Report BOTH, every time — this fixes the review's Issue 4:**
- **Pooled R² / RMSE / MAE** across all held-out predictions
  (the headline number),
- mean per-family R², median per-family R², min, max, N per family.
Per-family R² alone is misleading: within-family variance is
small, so a small error gives a huge negative R².

**Expected v1 pooled numbers to reproduce:** B-out — Ridge
R² 0.206, RMSE 0.132; XGB R² 0.086, RMSE 0.142. X-out — Ridge
R² −9.63, RMSE 0.484; XGB R² 0.375, RMSE 0.117.
Per-family means (for the table, not the headline): B-out
Ridge −0.77, XGB −0.94; X-out Ridge −12.11, XGB 0.28.

**Gate 4:** pooled numbers within ±0.03 of v1, Te/B families
present with N = 364 / 2,704, no family missing. **Correction
recorded 2026-11-25 (user approved):** the ±0.03 absolute
tolerance applies only when |v1 pooled R²| ≤ 1. For a pooled
R² below −1 — only X-out Ridge, v1 −9.63 — the tolerance is
relative: |v2 − v1| ≤ max(0.03, 0.05 × |v1|). v2 gave −9.958,
diff −0.328, relative diff 3.4%, so it PASSES under the
corrected gate. The conclusion is identical in v1 and v2:
Ridge catastrophically fails X-out, XGBoost survives at 0.375.
Do not re-run to chase that number — the run is deterministic.
If pooled B-out suddenly looks good (>0.3) with Magpie only, suspect
leakage — a B identity column slipped into X. Check the leakage
assert again.

---

## 8. Stage 5 — Study C: band gap (secondary)

**Do not do plain all-rows regression.** 96.1% of gaps are zero;
all-rows R² (v1: 0.288 tuned, −0.204 default) is not a band-gap
result. Never report it as one. It may appear in the paper only
as a labelled warning row.

**C1 — Classification, metal (gap == 0) vs non-metal (gap > 0):**
- Model: tuned XGBoost, Magpie only (Stage 5 first; site-aware
  as a separate row later).
- Splits: grouped-by-composition 5-fold only in v2 (corrected
  2026-11-25: the workflow first said random AND grouped, but
  v1's random numbers already exist — ROC-AUC 0.917, recall
  5.7% — and the grouped number is the honest headline;
  re-running random in v2 adds no decision value, so the
  comparison in the report is v1 random vs v2 grouped).
- Report: ROC-AUC, **recall for non-metals**, precision,
  F1, balanced accuracy, confusion matrix. **Never accuracy
  alone** — baseline is 0.961.
- **Threshold analysis (new in v2):** precision–recall curve,
  ROC curve, and metrics at thresholds 0.3, 0.5, 0.7 and at the
  threshold that maximises F1 on validation data only.
  v1 at 0.5: ROC-AUC 0.917, accuracy 0.962, recall 5.7%
  (42/735 caught, 693 missed), precision 65.6%.

**C2 — Regression, non-zero gaps only (n = 735):**
- Splits: grouped-by-composition 5-fold only in v2 (same
  correction as C1 — v1's random R² 0.505 is the comparison
  number; the grouped number is the honest headline).
- Report: R², RMSE, MAE, mean ± std, N = 735.
  v1 random: R² 0.505, RMSE 1.054, MAE 0.802 (provisional,
  small sample — say so).

**Gate 5:** grouped band-gap R² is reported next to random.
If grouped is much lower, that is the honest number — use it.
If classification recall is still <10% at every threshold,
say the model ranks well but does not screen — do not tune
the threshold on the test set to make recall look better.

---

## 9. Stage 6 — Study D: conformal uncertainty

**Method — exact split conformal (fixes the review's Issue 5):**
- Split: train 60% / calibrate 20% / test 20%, seed 0.
- Model: tuned XGBoost, Magpie only.
- Residuals on calibration: r = |y − ŷ|.
- **Exact quantile, not interpolated `np.quantile`:**
  sort r ascending; rank k = ceil((n_cal + 1) × 0.90);
  q = r[k−1] if k ≤ n_cal, else r[n_cal−1] (max residual).
  Mean width = 2q.
- Two tests, in the same cell, with the Te-leak fixed:
  (1) random test, (2) family-out test: train + calibrate ONLY
  on B ≠ Te, test ONLY on Te (364 rows). Print Te-in-train and
  Te-in-cal — both must be 0. (v2/v4's old conformal trained on
  ~218 Te rows and tested on all 364 — that leak must never
  return.)
- Save `results_conformal.csv` as one DataFrame, one write,
  unified columns.

**Framing — write this in the paper, word for word in spirit:**
random coverage is an empirical check of a 90% promise under
exchangeability; Te coverage is an empirical observation under
domain shift, where the guarantee does not apply. Report
coverage, mean width, q, R² on test, N_test, N_cal together.

**Expected v1 numbers to reproduce:** random coverage 0.900,
mean width 0.368, R² on test 0.438, N_test 3,787;
Te coverage 0.948, mean width 0.381, R² on test −0.086,
N_test 364. Te coverage did NOT fall — but only because the
intervals are wide (±0.19 eV/atom) while point predictions are
worse than the mean. Coverage ≠ accuracy. Say that.

**Gate 6:** random coverage 0.88–0.92, Te-in-train = Te-in-cal
= 0, file has 2 rows, columns unified. If random coverage is
far from 0.90, the quantile code is wrong — stop.

---

## 10. Stage 7 — Figures and tables (generated, never typed)

Figures, 300 dpi PNG + PDF, all generated from the saved CSVs:
1. Target distributions (e_form_per_atom + band gap, median marked).
2. CV comparison: random vs grouped R², by model, with the
   ~0.46 ceiling line.
3. Parity: tuned XGBoost, random split (R² ≈ 0.44) — one 80/20
   split, matching Stage 3.
4. Per-family R²: leave-one-B-out, sorted, N under bars,
   pooled R² in the caption, not mean per-family R².
5. Conformal: coverage vs interval width, random vs Te,
   with nominal 0.90 line and R² on test annotated.
6. Band gap: ROC + precision–recall curves, confusion matrix,
   and non-zero-gap parity (n = 735).

Tables, generated from CSVs into `results_table.md`:
- CV means (random + grouped), family-out pooled + per-family
  summary, band-gap classification + non-zero regression
  (random + grouped), conformal (random + Te).
- Every table states N, split, model, and seed handling.

**Gate 7:** every number in a figure/table can be traced to a
CSV row. If a number is typed by hand, delete it and regenerate.

---

## 11. Stage 8 — Paper draft (only after Gates 1–7 pass)

Outline (short benchmark paper, not a "we used XGBoost" paper):
1. Introduction — the question is transfer, not prediction.
2. Methods — Castelli only, target definition + citation,
   three IDs, site-order parsing, Magpie only, leakage rule,
   splits (random/grouped/family-out), nested/fixed tuning
   disclosure, metrics (pooled first), conformal (exact
   quantile, empirical framing).
3. Results — Study A, B, C, D in that order, with the ceiling.
4. Discussion — why Magpie fails on new B (composition-only,
   A/B swaps), why Ridge fails on X-out (linear extrapolation),
   why Te coverage stays high (wide intervals), limits:
   synthetic factorial grid, no hull stability, DFT gaps,
   n = 735 for band-gap regression, tuning disclosure.
5. Conclusion — one paragraph, no new numbers.
6. Supplement — family table (all 52 B + 7 X, N, pooled and
   per-family R²), split files, versions, column list.

**Success criteria (pre-registered, so we do not move the
goalposts):**
- v2 reproduces v1 Magpie-only numbers within ±0.02–0.03.
- B-out pooled R² is reported as the headline extrapolation
  number, whatever it is — even if it stays near 0.09.
- Site-aware B-out is a *separate* claim: it only counts as an
  improvement if pooled B-out R² clearly beats Magpie-only
  B-out (e.g. > 0.25) *and* the features are transferable
  (no B-identity one-hot).
- No claim of a new solar material, no "stable" claim, no
  accuracy-alone claim, no all-rows band-gap R² claim.

---

## 12. Never-do list (every bug that cost us time in v1)

1. Never merge on reduced formula. Never deduplicate to 9,646.
2. Never parquet a frame with Structure/Composition columns.
3. Never put vbm/cbm/fermi/mu_b/e_form/band_gap in X.
4. Never use default XGBoost/RF (depth 6, 300 trees) — R² −0.27.
5. Never use the radius rule for A/B — site order only.
6. Never paste .ipynb JSON into a Colab cell.
7. Never run Setup A after a restart; never assume OUT_DIR/df
   exist in a fresh session.
8. Never write rows of different shapes to one CSV.
9. Never run a cell twice without a backup + idempotency check.
10. Never quote site-aware random R² 0.90 as a result (tuned
    headline is Ridge 0.82 / XGB 0.76, random only).
11. Never quote classification accuracy 0.962 alone, or all-rows
    band-gap R² 0.288, as successes.
12. Never train conformal on Te rows and test on Te.
13. Never use mean per-family R² as the headline family-out number.
14. Never claim stability without energy-above-hull.
15. Never select a model on the test set and report that score
    as untouched.

---

## 13. Colab run order and honest times

```
Fresh session:  Setup B → Setup C
Stage 1 audit:  ~2 min
Stage 2 Magpie: ~3 min (2 min 20 s in v1)
Stage 3 CV:     1+ hour (RF is the bottleneck; XGB/Ridge are minutes)
Stage 4 family: ~20–30 min (Ridge + XGB, 59 families)
Stage 5 band gap: ~10 min
Stage 6 conformal: ~2 min
Stage 7 figures: ~5 min
```
Run one stage, check its gate, save, then the next. If the
runtime dies, re-run Setup B + C only, then the same stage —
finished folds/families are skipped, not redone.

**Version note:** Colab sklearn 1.6.1 vs this VM 1.9.1.
Key v1 numbers reproduced exactly across versions, but v2
numbers must be quoted from the Colab run, not from this VM.

## 12. Stage 9 — Journal upgrades (only after draft v1; added 2026-11-25)

Muhammad decided: upgrade first, then aim for a journal. Draft v1 is good
enough for a preprint, not yet for a journal. Order: 9a SHAP (fastest),
9b graph-model comparison, 9c robustness, 9d literature check, 9e candidate
screen. Draft v2 follows the upgrades.

### Stage 9a — SHAP feature importance (WHY the site-aware gain)

File: `v2_stage9a_shap_paste.txt` (538 lines, one pasteable cell, no module).
Inputs: `features_siteaware.parquet` (18,928 x 250) + `audited.parquet`.
Two tuned XGBoost models, seed 0, random 80/20 split: site-aware (232 cols)
vs Magpie-only control (132 cols). `shap.TreeExplainer` on a 2,000-row test
subsample. Feature groups asserted in code: Magpie 132, A-physics 12,
B-physics 12, X-physics 12, derived 5, B one-hot 52, X one-hot 7.
Outputs: `results_shap_v2.csv` (364 rows), `results_shap_groups_v2.csv`,
`shap_report_v2.md`, `figures/shap_importance.png` (300 dpi).
Correctness: SHAP identity (base + sum SHAP == prediction within 1e-6)
asserted on 5 rows per model. VM smoke: exit 0 twice, identity max error
~1e-7, on synthetic data only (real parquet is on Drive, not in the VM).
Colab runtime ~2-5 min. GATE 9a: files exist, identity passed, both models
trained; test R2 printed as sanity only, NOT gated.
CAVEAT (honest limit, stated in the cell and report): SHAP here is on a
random split — interpolation. It is supporting evidence for the mechanism.
The direct transfer test stays Stages 3b/3c. The mechanism sentence has
three branches: site-resolution carries the gain (agrees with 3c) /
B-site carries it (disagrees with 3c — reconcile before the paper) /
Magpie still carries it.

### Stage 9a2 — SHAP on held-out B metals (the direct mechanism test)

File: `v2_stage9a2_shap_familyout_paste.txt` (689 lines, one pasteable
cell, no module). For each of Te, W, Mg: train site-aware XGBoost ONLY on
B != that metal (held-out absence asserted in code), SHAP on all 364
held-out rows, held-out R2/RMSE/MAE stored with the shares. B one-hot /
B-physics for the held-out metal are legitimate test inputs (they describe
the test crystals; the held-out one-hot column is all zero in training) —
stated in the header, not leakage. Outputs: `results_shap_familyout_v2.csv`
(696 rows), `results_shap_familyout_groups_v2.csv` (21 rows),
`shap_familyout_report_v2.md`, `figures/shap_familyout.png` (300 dpi).
Identity asserted within 1e-6 per family. GATE 9a2: files/columns/rows,
identity, figure; held-out R2 printed as INFO only, NOT gated (W expected
low). VM smoke: exit 0 twice on synthetic data, identity max error 3.32e-07,
resume verified; two bugs found and fixed in smoke (missing () in
int(tr_mask.sum), pandas FutureWarning on empty-frame concat). Full Colab
run ~2-4 min. Same limitation as 9a: full path untested in VM (real parquet
on Drive). NOT yet run in Colab by Muhammad.

---

*End of workflow. Build only what is in this file, in this order,
and stop at any failed gate.*
