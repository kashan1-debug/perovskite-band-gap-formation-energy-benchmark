# ============================================================
# v2 — STAGE 3: Random + grouped CV (ONE cell, paste into Colab)
# Run AFTER Stage 2 (GATE 2 PASS) in the same session, or after
# Setup B then Setup C in a fresh session. It reloads
# features.parquet from Drive — it does NOT redo Stages 1 or 2.
# Full run takes 1+ hour in Colab (Random Forest is the slow part;
# Ridge and XGBoost are minutes). Every finished row is saved to
# Drive at once, so a disconnect loses nothing already done.
# If the run stops, just paste this cell again: rows already in
# results_cv_v2.csv are skipped, not redone.
# No family-out, no band gap, no conformal in this cell.
# Paste back: the GATE 3 line and the 24-row table (or the last
# rows printed, if it stopped part-way).
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs only the first 2,000 rows, 3-fold, seed 0,
# all 4 models, both splits, and writes results_cv_v2_smoke.csv
# instead of results_cv_v2.csv. Use it only to check the code runs.
SMOKE = False

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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

# --- 1. Reload features.parquet from Stage 2 (do NOT redo Stages 1 or 2) ---
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
for needed in ["B_site", "e_form_per_atom", "composition_group"]:
    if needed not in df.columns:
        raise AssertionError(f"STOP: needed column {needed} is missing from features.parquet.")
if df.shape != (18928, 150) or len(magpie_cols) != 132:
    raise AssertionError(
        "STOP: features.parquet is not (18928, 150) with 132 Magpie columns. "
        "Re-run Stage 2 before modelling."
    )

# --- 2. X and y. Magpie ONLY. Never vbm/cbm/fermi/mu_b/e_form/band_gap. ---
# vbm and cbm define the band gap directly (gap = cbm - vbm), fermi
# level / fermi width / mu_b are DFT outputs, e_form / band_gap /
# e_form_per_atom are targets. Including any of them is cheating.
X = df[magpie_cols].values
y = df["e_form_per_atom"].values
groups = df["composition_group"].values
print("X shape:", X.shape, "| y shape:", y.shape, "| groups:", len(np.unique(groups)))
print("X is Magpie only. Forbidden columns are NOT in X by construction.")

# --- 3. Models — TUNED configs only, never defaults. ---
# Default XGBoost (depth 6, 300 trees) scored R2 -0.272 in v1 and
# default Random Forest scored -0.084 — worse than the mean. Never
# ship defaults again. Ridge is standardised inside a pipeline, so
# the scaling is fitted on training folds only (no leakage).
# NOTE on Random Forest: this cell uses n_estimators=100,
# max_depth=10 (shallow). That is the v1 Cell 10 TUNED config
# (glossary: "tuned forest: depth 10, 100 trees", random R2 0.258,
# grouped R2 0.475) — the config the GATE 3 numbers come from.
# The audit config (depth 8, min_samples_leaf 5, 100 trees) was a
# separate VM diagnostic (single fold R2 0.372), NOT the v1 Cell 10
# config, so it must not be used for a reproduction.
def make_model(name, seed):
    if name == "Mean":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
            tree_method="hist", n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown model {name}")


MODELS = ["Mean", "Ridge", "RandomForest", "XGBoost"]
print("Models (tuned only):")
print("  Mean: DummyRegressor, strategy=mean")
print("  Ridge: StandardScaler + Ridge(alpha=1.0)")
print("  RandomForest: n_estimators=100, max_depth=10, n_jobs=-1")
print("  XGBoost: n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5")

# --- 4. Splits, seeds, folds. ---
# Random: KFold, shuffle, one seed per run. Grouped: GroupKFold on
# composition_group, NO shuffle — so grouped folds are IDENTICAL
# for seeds 0, 1 and 2. That is expected, not a bug: only the
# models' own randomness (RandomForest, XGBoost) differs between
# grouped seeds, so Ridge and Mean grouped rows will be identical.
if SMOKE:
    df = df.iloc[:2000].reset_index(drop=True)
    X = df[magpie_cols].values
    y = df["e_form_per_atom"].values
    groups = df["composition_group"].values
    N_FOLDS = 3
    SEEDS = [0]
    print("SMOKE: first 2,000 rows only, 3-fold, seed 0 only.")
else:
    N_FOLDS = 5
    SEEDS = [0, 1, 2]
print(f"Splits: random KFold + grouped GroupKFold, {N_FOLDS}-fold, seeds {SEEDS}")
print("Full run = 2 schemes x 4 models x 3 seeds = 24 rows "
      f"(SMOKE = 2 x 4 x 1 = 8 rows).")

# --- 5. Results file: ONE fixed column set, checkpoint after every row. ---
# Resume-by-skipping: if results_cv_v2.csv already exists (a
# re-run after a disconnect), rows already in it are skipped, not
# redone. The file is never wiped on a re-run. In SMOKE mode a
# separate file is used so test rows can never mix into real results.
CSV_COLS = ["split", "model", "seed", "n", "r2_mean", "r2_std",
            "rmse_mean", "rmse_std", "mae_mean", "mae_std", "experiment"]
csv_name = "results_cv_v2_smoke.csv" if SMOKE else "results_cv_v2.csv"
csv_path = OUT_DIR / csv_name
done = set()
if csv_path.exists():
    old = pd.read_csv(csv_path)
    if list(old.columns) != CSV_COLS:
        backup = OUT_DIR / (csv_name + ".broken_backup.csv")
        old.to_csv(backup, index=False)
        print(f"WARNING: old {csv_name} had wrong columns — backed up to "
              f"{backup.name}, starting fresh.")
        csv_path.unlink()
    else:
        done = {(str(r.split), str(r.model), int(r.seed))
                for r in old.itertuples()}
        print(f"Resuming: {len(done)} rows already in {csv_name}, "
              "they will be skipped.")
else:
    print(f"No existing {csv_name} — starting fresh.")
EXPERIMENT = "smoke" if SMOKE else "reproduction"


def save_row(row):
    """Append one row to the CSV at once, so a disconnect cannot lose it."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"  SAVED {row['split']} {row['model']} seed {row['seed']}: "
          f"R2 {row['r2_mean']:.3f} +- {row['r2_std']:.3f}, "
          f"RMSE {row['rmse_mean']:.3f}, MAE {row['mae_mean']:.3f}")


def run_one(split, model_name, seed):
    """One (split, model, seed): 5 folds, mean +- std across folds."""
    t0 = time.time()
    if split == "random":
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        splits = cv.split(X)
    else:
        cv = GroupKFold(n_splits=N_FOLDS)
        splits = cv.split(X, y, groups)
    r2s, rmses, maes = [], [], []
    for tr_idx, te_idx in splits:
        m = make_model(model_name, seed)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        r2s.append(r2_score(y[te_idx], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y[te_idx], pred))))
        maes.append(float(mean_absolute_error(y[te_idx], pred)))
    row = {
        "split": split,
        "model": model_name,
        "seed": seed,
        "n": len(df),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "experiment": EXPERIMENT,
    }
    print(f"[{split} {model_name} seed {seed}] done in {time.time() - t0:.0f} s — "
          f"per-fold R2: {[round(v, 3) for v in r2s]}")
    save_row(row)
    return row


# --- 6. Run everything, checkpointing after every row. ---
all_rows = []
for split in ["random", "grouped"]:
    for model_name in MODELS:
        for seed in SEEDS:
            key = (split, model_name, seed)
            if key in done:
                print(f"SKIP {split} {model_name} seed {seed} — already in {csv_name}.")
                continue
            all_rows.append(run_one(split, model_name, seed))
print(f"Rows finished in this run: {len(all_rows)} "
      f"(total in {csv_name} after this run: {len(done) + len(all_rows)}).")

# --- 7. GATE 3 — reproduction check (full run only). ---
# v1 Cell 10 TUNED numbers (Magpie only, e_form_per_atom, mean across
# seeds). Tolerance +-0.03 R2. Labelled 'reproduction': same data,
# same target, same split method, same model config — so v2 should
# match v1. If a number is outside tolerance, that is a failed
# reproduction, not a "corrected" number: STOP, do not go to Stage 4.
# Ceiling alarm: pooled composition-only ceiling from Stage 1 is
# R2 0.543. No Magpie-only model can beat it — if one does, that is
# a leakage alarm, not a success.
V1 = {
    ("random", "Mean"): 0.000, ("random", "Ridge"): 0.385,
    ("random", "RandomForest"): 0.258, ("random", "XGBoost"): 0.432,
    ("grouped", "Mean"): 0.000, ("grouped", "Ridge"): 0.388,
    ("grouped", "RandomForest"): 0.475, ("grouped", "XGBoost"): 0.443,
}
CEILING = 0.543
results = pd.read_csv(csv_path)
gate_lines = []
gate_failures = []
if SMOKE:
    print("GATE 3: SKIPPED in SMOKE mode — smoke R2s are on 2,000 rows "
          "and 3 folds, so they cannot be compared with v1 full-run numbers.")
    gate_lines.append("GATE 3: SKIPPED (SMOKE mode — no v1 comparison).")
else:
    print("GATE 3 — reproduction vs v1 Cell 10 tuned (tolerance +-0.03 R2):")
    for (split, model_name), v1_r2 in V1.items():
        sub = results[(results["split"] == split) & (results["model"] == model_name)]
        v2_r2 = float(sub["r2_mean"].mean())
        ok = abs(v2_r2 - v1_r2) <= 0.03
        status = "PASS" if ok else "FAIL"
        line = (f"  [{status}] {split} {model_name}: v2 {v2_r2:.3f} "
                f"vs v1 {v1_r2:.3f} (diff {v2_r2 - v1_r2:+.3f})")
        print(line)
        gate_lines.append(line)
        if not ok:
            gate_failures.append(f"{split} {model_name}: v2 {v2_r2:.3f} vs v1 {v1_r2:.3f}")
    print(f"Ceiling reminder: pooled composition-only ceiling R2 = {CEILING:.3f}.")
    best = float(results["r2_mean"].max())
    print(f"Best v2 mean R2 in this file: {best:.3f}.")
    if best > CEILING + 0.03:
        msg = (f"LEAKAGE ALARM: best R2 {best:.3f} is above the ceiling "
               f"{CEILING:.3f} — a forbidden column may have slipped into X.")
        print(msg)
        gate_lines.append(msg)
        gate_failures.append(msg)

# --- 8. Save the Stage 3 report. ---
report_name = "cv_report_v2_smoke.md" if SMOKE else "cv_report_v2.md"
report_lines = [
    "# Perovskite v2 — Stage 3 random + grouped CV report",
    "",
    "Magpie only (132 columns), e_form_per_atom, 5-fold, seeds 0, 1, 2.",
    "Models are tuned configs only — no defaults.",
    "RandomForest here: n_estimators=100, max_depth=10 "
    "(the v1 Cell 10 tuned config — random R2 0.258, grouped 0.475).",
    "Grouped folds are identical across seeds (GroupKFold has no shuffle) — expected.",
    "",
    "## Results (mean +- std across folds, per seed)",
    "",
    results.to_string(index=False),
    "",
    "## GATE 3 — reproduction vs v1 Cell 10 tuned",
    "",
]
report_lines += gate_lines
report_lines += [
    "",
    "## Notes",
    "",
    "- experiment column: 'reproduction' for the full run, 'smoke' for test runs.",
    "- Pooled composition-only ceiling R2 = 0.543 (Stage 1). "
    "No Magpie-only model should beat it.",
    "- If GATE 3 failed, STOP — do not proceed to Stage 4. "
    "Find why v2 differs from v1 first.",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")
print(f"saved {csv_name}: {len(results)} rows")

# --- GATE 3 verdict. ---
if SMOKE:
    print("STAGE 3 SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 3: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 3 FAILED: " + "; ".join(gate_failures))
else:
    print("GATE 3: PASS — v2 reproduces v1 within +-0.03 R2. Next: Stage 4 family-out.")
    print("STAGE 3 done — paste back the GATE 3 line and the results table.")
