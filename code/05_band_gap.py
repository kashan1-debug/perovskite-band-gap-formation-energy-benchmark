# ============================================================
# v2 — STAGE 5: Band gap (ONE cell, paste into Colab)
# Run AFTER Stage 4 (GATE 4 PASS, corrected) in the same session,
# or after Setup B then Setup C in a fresh session. It reloads
# features.parquet from Drive — it does NOT redo Stages 1-4.
# Full run takes about 15-20 minutes in Colab. Every finished
# task + seed is saved to Drive at once, so a disconnect loses
# nothing already done. If the run stops, just paste this cell
# again: finished rows are skipped, not redone.
# Model: tuned XGBoost ONLY, Magpie only (132 columns).
# C1 classification (metal vs non-metal) and C2 regression on the
# 735 non-zero gaps use GROUPED 5-fold CV on composition_group —
# that is the v2 fix; v1 used random K-fold only, so v2 is a
# CORRECTED experiment, not a reproduction, and its numbers are
# NOT expected to match v1 exactly. Do not force them to match.
# C0 plain all-rows regression is a WARNING row only — it looks
# OK only because 96% of gaps are zero. Never use it as a result.
# No family-out, no conformal, no formation-energy CV in this cell.
# Paste back: the GATE 5 line, the C1 table and the C2 table.
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs C1 on a stratified subsample of 4,000 rows and
# C2 on all 735 non-zero rows, 3-fold, seed 0 only, and writes
# results_bandgap_v2_smoke.csv instead of results_bandgap_v2.csv.
# Use it only to check the code runs.
SMOKE = False

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from xgboost import XGBClassifier, XGBRegressor

# --- OUT_DIR: fresh-session safe. Define it HERE, do not assume it exists. ---
# In Colab, Setup C mounts Drive; this is the results_v2 folder it makes.
# If OUT_DIR was already set by Setup C, keep that value as-is.
try:
    OUT_DIR
except NameError:
    OUT_DIR = Path("/content/drive/MyDrive/perovskite/results_v2")
OUT_DIR = Path(OUT_DIR)
if str(OUT_DIR).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
    raise RuntimeError(
        "Drive is not mounted. Run Setup C (mount Drive), then this cell again. "
        "Do NOT re-run Setup A after a restart."
    )
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR =", OUT_DIR)
print("SMOKE =", SMOKE, "(True = small test only, False = full run)")

# --- 1. Reload features.parquet from Stage 2 (do NOT redo Stages 1-4) ---
feat_path = OUT_DIR / "features.parquet"
if not feat_path.exists():
    raise FileNotFoundError(
        "features.parquet not found in OUT_DIR. Run Stage 2 first, "
        "then this cell again."
    )
df = pd.read_parquet(feat_path)
print("features.parquet shape:", df.shape, "(expected (18928, 150))")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
print("Magpie columns:", len(magpie_cols), "(expected 132)")
for needed in ["band_gap", "composition_group"]:
    if needed not in df.columns:
        raise AssertionError(f"STOP: needed column {needed} is missing from features.parquet.")
if df.shape != (18928, 150) or len(magpie_cols) != 132:
    raise AssertionError(
        "STOP: features.parquet is not (18928, 150) with 132 Magpie columns. "
        "Re-run Stage 2 before modelling."
    )

# --- 2. X and y. Magpie ONLY. Never vbm/cbm/fermi/mu_b/e_form/band_gap. ---
# vbm and cbm define the band gap directly (gap = cbm - vbm), so
# including them would be cheating: the model could just subtract.
X = df[magpie_cols].values
gap = df["band_gap"].values
groups = df["composition_group"].values
is_nonmetal = (gap > 0).astype(int)  # 1 = non-metal (gap > 0), 0 = metal (gap == 0)
print("X shape:", X.shape, "| band_gap == 0:", int((gap == 0).sum()),
      "(expected 18193) | > 0:", int((gap > 0).sum()), "(expected 735)")
print("Majority baseline accuracy (always guess metal):",
      f"{(gap == 0).mean():.3f} — accuracy near this is NOT a success, "
      "it is just the baseline. Report recall and ROC-AUC instead.")
print("Band gap is the DFT (GLLBSC) gap, NOT the experimental gap — say so always.")
print("X is Magpie only. Forbidden columns are NOT in X by construction.")

# --- 3. Model — tuned XGBoost only, same config as Stages 3-4. ---
# Default XGBoost scored below the mean on formation energy in v1.
# Never ship defaults again.
def make_clf(seed):
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
        tree_method="hist", n_jobs=-1, random_state=seed)

def make_reg(seed):
    return XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
        tree_method="hist", n_jobs=-1, random_state=seed)

print("Model (tuned only): XGBoost n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5")
THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
if SMOKE:
    N_FOLDS = 3
    SEEDS = [0]
    print("SMOKE: C1 on a stratified subsample of 4,000 rows, C2 on all "
          "735 non-zero rows, 3-fold, seed 0 only.")
else:
    N_FOLDS = 5
    SEEDS = [0, 1, 2]
print(f"Splits: grouped GroupKFold on composition_group, {N_FOLDS}-fold, "
      f"seeds {SEEDS} (C1 and C2). C0 warning: random KFold, seed 0 only.")
print("Experiment label: 'corrected' — v1 used random K-fold, v2 uses "
      "grouped, so v2 numbers are NOT expected to match v1 exactly.")

# --- 4. Results file: ONE fixed column set, checkpoint after every row. ---
# Resume-by-skipping: if results_bandgap_v2.csv already exists (a
# re-run after a disconnect), rows already in it are skipped, not
# redone. The file is never wiped on a re-run. In SMOKE mode a
# separate file is used so test rows can never mix into real results.
# Out-of-fold predictions are also saved, one row per test crystal,
# to predictions_bandgap_v2.parquet — the threshold analysis and
# the pooled metrics are computed from these, not from per-fold
# means. A task + seed counts as DONE only if it is in BOTH files:
# if a re-run finds CSV rows whose predictions are missing, those
# rows are dropped and the task is re-run, so the files agree.
CSV_COLS = ["task", "model", "split", "seed", "n", "threshold",
            "roc_auc", "accuracy", "precision", "recall", "f1", "pr_auc",
            "tn", "fp", "fn", "tp", "n_pred_nonmetal",
            "r2", "rmse", "mae", "experiment"]
csv_name = "results_bandgap_v2_smoke.csv" if SMOKE else "results_bandgap_v2.csv"
csv_path = OUT_DIR / csv_name
pred_name = ("predictions_bandgap_v2_smoke.parquet" if SMOKE
             else "predictions_bandgap_v2.parquet")
pred_path = OUT_DIR / pred_name
EXPERIMENT = "smoke" if SMOKE else "corrected"
MAIN_TASKS = ["classification_metal_vs_nonmetal",
              "regression_nonzero_gap", "warning_plain_all_rows"]

done = set()  # (task, seed) for main rows; threshold rows follow their seed
if csv_path.exists():
    old = pd.read_csv(csv_path)
    if list(old.columns) != CSV_COLS:
        backup = OUT_DIR / (csv_name + ".broken_backup.csv")
        old.to_csv(backup, index=False)
        print(f"WARNING: old {csv_name} had wrong columns — backed up to "
              f"{backup.name}, starting fresh.")
        csv_path.unlink()
        old = pd.DataFrame(columns=CSV_COLS)
    pred_done = set()
    if pred_path.exists():
        old_pred = pd.read_parquet(pred_path)
        pred_done = {(str(r.task), int(r.seed)) for r in old_pred.itertuples()}
    done_csv = {(str(r.task), int(r.seed)) for r in old.itertuples()
                if str(r.task) in MAIN_TASKS}
    # Drop CSV rows whose predictions are missing — they will be re-run.
    orphaned = done_csv - pred_done
    if orphaned:
        print(f"WARNING: {len(orphaned)} CSV rows have no saved predictions "
              "— dropping them so the task is re-run cleanly.")
        keep = old[~old.apply(
            lambda r: (str(r.task), int(r.seed)) in orphaned
            or (str(r.task) == "classification_threshold"
                and ("classification_metal_vs_nonmetal", int(r.seed)) in orphaned),
            axis=1)]
        keep.to_csv(csv_path, index=False)
        old = keep
    done = done_csv - orphaned
    print(f"Resuming: {len(done)} task+seed rows already in {csv_name} "
          "with predictions saved, they will be skipped.")
else:
    print(f"No existing {csv_name} — starting fresh.")


def save_rows(rows, pred_df_new):
    """Append rows to the CSV and their predictions to the parquet file,
    at once, so a disconnect cannot lose either. Empty strings are
    written for metrics that do not apply to a row (e.g. R2 on a
    classification row) — the column set never changes."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in CSV_COLS})
    if pred_path.exists():
        old_pred = pd.read_parquet(pred_path)
        pred_df_new = pd.concat([old_pred, pred_df_new], ignore_index=True)
    pred_df_new.to_parquet(pred_path, index=False)


def empty_row(task, model, split, seed, n):
    return {"task": task, "model": model, "split": split, "seed": seed,
            "n": n, "threshold": "", "roc_auc": "", "accuracy": "",
            "precision": "", "recall": "", "f1": "", "pr_auc": "",
            "tn": "", "fp": "", "fn": "", "tp": "", "n_pred_nonmetal": "",
            "r2": "", "rmse": "", "mae": "", "experiment": EXPERIMENT}


def threshold_metrics(y_true, y_prob, thr):
    """Precision / recall / F1 / confusion at one threshold."""
    pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_pred_nonmetal": int(tp + fp),
    }


# --- 5. C1 — Classification, metal vs non-metal, GROUPED CV. ---
# v1 (random K-fold): ROC-AUC 0.917, accuracy 0.962, recall 5.7%
# (42/735 caught, 693 missed), precision 65.6% at threshold 0.5.
# v2 is grouped, so expect recall / ROC-AUC to move — that is the
# corrected number, not a failed reproduction.
oof_probs = {}  # seed -> out-of-fold probabilities, pooled across folds
for seed in SEEDS:
    key = ("classification_metal_vs_nonmetal", seed)
    if key in done:
        print(f"SKIP C1 classification seed {seed} — already in {csv_name}.")
        # Reload its OOF probs for the pooled threshold table.
        _p = pd.read_parquet(pred_path)
        _s = _p[(_p["task"] == "classification_metal_vs_nonmetal")
                & (_p["seed"] == seed)].sort_values("row_idx")
        oof_probs[seed] = _s["y_prob"].values
        continue
    t0 = time.time()
    if SMOKE:
        idx, _ = train_test_split(np.arange(len(df)), train_size=4000,
                                  stratify=is_nonmetal, random_state=0)
        idx = np.sort(idx)
        Xs, ys, gs = X[idx], is_nonmetal[idx], groups[idx]
    else:
        idx, Xs, ys, gs = np.arange(len(df)), X, is_nonmetal, groups
    oof = np.zeros(len(idx))
    cv = GroupKFold(n_splits=N_FOLDS)
    for tr, te in cv.split(Xs, ys, gs):
        m = make_clf(seed)
        m.fit(Xs[tr], ys[tr])
        oof[te] = m.predict_proba(Xs[te])[:, 1]
    oof_probs[seed] = oof
    roc = float(roc_auc_score(ys, oof))
    pr_auc = float(average_precision_score(ys, oof))
    tm = threshold_metrics(ys, oof, 0.5)
    row = empty_row("classification_metal_vs_nonmetal", "XGBoost",
                    "grouped", seed, len(idx))
    row.update({"threshold": 0.5, "roc_auc": roc, "pr_auc": pr_auc, **tm})
    thr_rows = []
    for thr in THRESHOLDS:
        if thr == 0.5:
            continue  # already the main row
        tr_ = threshold_metrics(ys, oof, thr)
        r_ = empty_row("classification_threshold", "XGBoost",
                       "grouped", seed, len(idx))
        r_.update(tr_)
        thr_rows.append(r_)
    pred_df = pd.DataFrame({
        "task": "classification_metal_vs_nonmetal", "seed": seed,
        "row_idx": idx, "y_true": ys, "y_pred": (oof >= 0.5).astype(int),
        "y_prob": oof,
    })
    save_rows([row] + thr_rows, pred_df)
    print(f"[C1 seed {seed}] done in {time.time() - t0:.0f} s — "
          f"ROC-AUC {roc:.3f}, PR-AUC {pr_auc:.3f}, at 0.5: "
          f"accuracy {tm['accuracy']:.3f} (baseline {(ys == 0).mean():.3f}), "
          f"precision {tm['precision']:.3f}, recall {tm['recall']:.3f}, "
          f"F1 {tm['f1']:.3f}, caught {tm['tp']}/{int(ys.sum())} non-metals")
    print(f"  SAVED C1 seed {seed}: {len(thr_rows) + 1} rows "
          "(main + threshold rows)")

# --- 6. C2 — Regression, non-zero gaps only (n = 735), GROUPED CV. ---
# v1 (random K-fold): R2 0.505, RMSE 1.054, MAE 0.802 — provisional,
# small sample. v2 is grouped, so expect a lower, noisier number.
# Report mean +- std across seeds; n = 735 is small, say so.
nz_mask = gap > 0
X2, y2, g2 = X[nz_mask], gap[nz_mask], groups[nz_mask]
print(f"C2: non-zero gaps n={len(y2)} (expected 735), "
      f"mean {y2.mean():.3f}, median {np.median(y2):.3f}, max {y2.max():.3f}")
oof_preds_c2 = {}
for seed in SEEDS:
    key = ("regression_nonzero_gap", seed)
    if key in done:
        print(f"SKIP C2 non-zero regression seed {seed} — already in {csv_name}.")
        _p = pd.read_parquet(pred_path)
        _s = _p[(_p["task"] == "regression_nonzero_gap")
                & (_p["seed"] == seed)].sort_values("row_idx")
        oof_preds_c2[seed] = _s["y_pred"].values
        continue
    t0 = time.time()
    oof = np.zeros(len(y2))
    cv = GroupKFold(n_splits=N_FOLDS)
    fold_r2 = []
    for tr, te in cv.split(X2, y2, g2):
        m = make_reg(seed)
        m.fit(X2[tr], y2[tr])
        oof[te] = m.predict(X2[te])
        fold_r2.append(float(r2_score(y2[te], oof[te])))
    oof_preds_c2[seed] = oof
    row = empty_row("regression_nonzero_gap", "XGBoost", "grouped",
                    seed, len(y2))
    row.update({
        "r2": float(r2_score(y2, oof)),
        "rmse": float(np.sqrt(mean_squared_error(y2, oof))),
        "mae": float(mean_absolute_error(y2, oof)),
    })
    pred_df = pd.DataFrame({
        "task": "regression_nonzero_gap", "seed": seed,
        "row_idx": np.where(nz_mask)[0], "y_true": y2, "y_pred": oof,
        "y_prob": np.nan,
    })
    save_rows([row], pred_df)
    print(f"[C2 seed {seed}] done in {time.time() - t0:.0f} s — "
          f"pooled R2 {row['r2']:.3f}, RMSE {row['rmse']:.3f}, "
          f"MAE {row['mae']:.3f} | per-fold R2: "
          f"{[round(v, 3) for v in fold_r2]} (n=735, noisy — report mean +- std)")

# --- 7. C0 — WARNING row only: plain all-rows regression. ---
# 96.1% of gaps are zero, so this looks OK only because the model
# mostly predicts zero. v1 tuned: R2 0.288. NEVER the headline.
# Random 5-fold, seed 0 only, tuned XGBoost.
if ("warning_plain_all_rows", 0) in done:
    print("SKIP C0 warning row — already in results file.")
else:
    t0 = time.time()
    if SMOKE:
        idx0, _ = train_test_split(np.arange(len(df)), train_size=4000,
                                   stratify=is_nonmetal, random_state=0)
        idx0 = np.sort(idx0)
        X0, y0 = X[idx0], gap[idx0]
    else:
        idx0, X0, y0 = np.arange(len(df)), X, gap
    oof0 = np.zeros(len(y0))
    cv0 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tr, te in cv0.split(X0):
        m = make_reg(0)
        m.fit(X0[tr], y0[tr])
        oof0[te] = m.predict(X0[te])
    row0 = empty_row("warning_plain_all_rows", "XGBoost", "random", 0, len(y0))
    row0.update({
        "r2": float(r2_score(y0, oof0)),
        "rmse": float(np.sqrt(mean_squared_error(y0, oof0))),
        "mae": float(mean_absolute_error(y0, oof0)),
    })
    pred0 = pd.DataFrame({
        "task": "warning_plain_all_rows", "seed": 0,
        "row_idx": idx0, "y_true": y0, "y_pred": oof0, "y_prob": np.nan,
    })
    save_rows([row0], pred0)
    print(f"[C0 WARNING seed 0] done in {time.time() - t0:.0f} s — "
          f"R2 {row0['r2']:.3f}, RMSE {row0['rmse']:.3f}, MAE {row0['mae']:.3f} "
          "— WARNING ONLY: looks OK only because 96% of gaps are zero. "
          "Never report this as a band-gap result.")

# --- 8. Threshold analysis, pooled across seeds (mean OOF prob per row). ---
# Mean OOF probability across seeds, one value per crystal — this is
# the pooled threshold table. It is NOT tuned on the test set: the
# thresholds are fixed in advance (0.1 … 0.9).
results = pd.read_csv(csv_path)
c1_main = results[results["task"] == "classification_metal_vs_nonmetal"]
c2_rows = results[results["task"] == "regression_nonzero_gap"]
c0_rows = results[results["task"] == "warning_plain_all_rows"]
print("C1 classification (grouped, per seed, pooled across folds):")
print(c1_main.to_string(index=False))
print("C2 non-zero-gap regression (grouped, per seed, pooled across folds):")
print(c2_rows.to_string(index=False))
print("C0 WARNING (plain all-rows regression — never the headline):")
print(c0_rows.to_string(index=False))
pooled_thr_lines = []
if oof_probs:
    mean_prob = np.mean([oof_probs[s] for s in SEEDS if s in oof_probs], axis=0)
    # In SMOKE the C1 subsample is the same 4,000 rows for seed 0; in the
    # full run every seed covers all rows, so the mean prob aligns.
    print("Threshold analysis — pooled across seeds (mean OOF prob per row):")
    print("  thr | precision | recall | F1 | predicted non-metal")
    for thr in THRESHOLDS:
        tm = threshold_metrics(is_nonmetal if not SMOKE else
                               is_nonmetal[np.sort(train_test_split(
                                   np.arange(len(df)), train_size=4000,
                                   stratify=is_nonmetal,
                                   random_state=0)[0])], mean_prob, thr)
        line = (f"  {thr:.1f} | {tm['precision']:.3f} | {tm['recall']:.3f} | "
                f"{tm['f1']:.3f} | {tm['n_pred_nonmetal']}")
        print(line)
        pooled_thr_lines.append(line)
    print("  NOTE: accuracy at any threshold is near the 0.961 majority "
          "baseline — never report accuracy alone.")

# --- 9. GATE 5 — corrected-experiment checks (full run only). ---
# v1 was RANDOM-split, v2 is GROUPED, so v2 is NOT expected to
# match v1. The gate checks the v2 numbers are sane and the
# honest pattern holds — it does NOT fail for differing from v1.
def gate5_check():
    lines, failures = [], []
    roc_mean = float(c1_main["roc_auc"].astype(float).mean())
    rec_mean = float(c1_main["recall"].astype(float).mean())
    ok = 0.80 <= roc_mean <= 0.97
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] C1 ROC-AUC mean {roc_mean:.3f} "
                 f"(expected in [0.80, 0.97])")
    if not ok:
        failures.append(f"C1 ROC-AUC {roc_mean:.3f} outside [0.80, 0.97]")
    ok = rec_mean < 0.20
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] C1 recall at 0.5 mean "
                 f"{rec_mean:.3f} (expected < 0.20, v1 random was 0.057 — "
                 "the model ranks well but misses most non-metals at 0.5)")
    if not ok:
        failures.append(f"C1 recall at 0.5 {rec_mean:.3f} not < 0.20")
    r2_mean = float(c2_rows["r2"].astype(float).mean())
    rmse_mean = float(c2_rows["rmse"].astype(float).mean())
    ok = 0.0 <= r2_mean <= 0.70
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] C2 non-zero R2 mean "
                 f"{r2_mean:.3f} (expected in [0.0, 0.70])")
    if not ok:
        failures.append(f"C2 R2 {r2_mean:.3f} outside [0.0, 0.70]")
    ok = 0.7 <= rmse_mean <= 1.4
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] C2 non-zero RMSE mean "
                 f"{rmse_mean:.3f} (expected in [0.7, 1.4])")
    if not ok:
        failures.append(f"C2 RMSE {rmse_mean:.3f} outside [0.7, 1.4]")
    # Headline check: the gate and the headline use C1/C2 only.
    # The warning row must exist, be labelled 'warning_plain_all_rows',
    # and must not be part of any gate check above.
    ok = (len(c0_rows) >= 1
          and set(c0_rows["task"].unique()) == {"warning_plain_all_rows"})
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] Headline is C1/C2 only — "
                 "warning row exists, labelled 'warning_plain_all_rows', "
                 "and was not used in any gate check")
    if not ok:
        failures.append("warning row missing or mislabelled")
    return lines, failures


gate_lines = []
gate_failures = []
if SMOKE:
    print("GATE 5: SKIPPED in SMOKE mode — smoke uses a 4,000-row subsample "
          "and 3 folds, so the sanity ranges do not apply.")
    gate_lines.append("GATE 5: SKIPPED (SMOKE mode — no sanity-range check).")
    # Informational only: print what the checks would say, without failing.
    info_lines, _ = gate5_check()
    print("SMOKE-INFO (not a verdict):")
    for line in info_lines:
        print("   " + line)
else:
    print("GATE 5 — corrected-experiment sanity checks "
          "(v1 was random-split; v2 is grouped, so no v1 match is expected):")
    gate_lines, gate_failures = gate5_check()
    for line in gate_lines:
        print(line)

# --- 10. Save the Stage 5 report. ---
report_name = "bandgap_report_v2_smoke.md" if SMOKE else "bandgap_report_v2.md"
c1_mean = c1_main[["roc_auc", "accuracy", "precision", "recall", "f1",
                   "pr_auc"]].astype(float).mean()
c2_mean = c2_rows[["r2", "rmse", "mae"]].astype(float).mean()
c2_std = c2_rows[["r2", "rmse", "mae"]].astype(float).std()
report_lines = [
    "# Perovskite v2 — Stage 5 band-gap report",
    "",
    "Magpie only (132 columns), tuned XGBoost only.",
    "Experiment: 'corrected' — v1 used random K-fold, v2 uses GROUPED "
    "5-fold CV on composition_group, so v2 numbers are NOT expected "
    "to match v1 exactly. Do not force them to match.",
    "Band gap is the DFT (GLLBSC) gap, not the experimental gap.",
    "",
    "## C1 — Classification, metal vs non-metal (grouped, per seed)",
    "",
    c1_main.to_string(index=False),
    "",
    f"Mean across seeds: ROC-AUC {c1_mean['roc_auc']:.3f}, "
    f"accuracy {c1_mean['accuracy']:.3f} "
    f"(majority baseline {(gap == 0).mean():.3f} — never report accuracy alone), "
    f"precision {c1_mean['precision']:.3f}, recall {c1_mean['recall']:.3f}, "
    f"F1 {c1_mean['f1']:.3f}, PR-AUC {c1_mean['pr_auc']:.3f}.",
    "v1 (random): ROC-AUC 0.917, accuracy 0.962, recall 0.057 "
    "(42/735 caught), precision 0.656 at threshold 0.5.",
    "",
    "## Threshold analysis — pooled across seeds (mean OOF prob per row)",
    "",
]
report_lines += pooled_thr_lines
report_lines += [
    "",
    "## C2 — Regression, non-zero gaps only, n = 735 (grouped, per seed)",
    "",
    c2_rows.to_string(index=False),
    "",
    f"Mean across seeds: R2 {c2_mean['r2']:.3f} +- {c2_std['r2']:.3f}, "
    f"RMSE {c2_mean['rmse']:.3f} +- {c2_std['rmse']:.3f}, "
    f"MAE {c2_mean['mae']:.3f} +- {c2_std['mae']:.3f}. "
    "n = 735 is small and grouped folds are noisy — report mean +- std.",
    "v1 (random): R2 0.505, RMSE 1.054, MAE 0.802 (provisional).",
    "",
    "## C0 — WARNING only: plain all-rows regression (never the headline)",
    "",
    c0_rows.to_string(index=False),
    "",
    "This looks OK only because 96.1% of gaps are zero. "
    "v1 tuned: R2 0.288. Never report it as a band-gap result.",
    "",
    "## GATE 5 — corrected-experiment sanity checks",
    "",
]
report_lines += gate_lines
report_lines += [
    "",
    "## Notes",
    "",
    "- experiment column: 'corrected' for the full run, 'smoke' for test runs.",
    "- Grouped CV is stricter than random CV: related compositions "
    "cannot share a fold, so grouped numbers are the honest ones.",
    "- If GATE 5 failed, STOP — do not proceed to Stage 6. "
    "Find why the v2 numbers are outside the sanity ranges first.",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")
print(f"saved {csv_name}: {len(results)} rows")

# --- GATE 5 verdict. ---
if SMOKE:
    print("STAGE 5 SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 5: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 5 FAILED: " + "; ".join(gate_failures))
else:
    print("GATE 5: PASS — band-gap numbers are sane, grouped, and honest. "
          "Next: Stage 6 conformal.")
    print("STAGE 5 done — paste back the GATE 5 line, the C1 table "
          "and the C2 table.")
