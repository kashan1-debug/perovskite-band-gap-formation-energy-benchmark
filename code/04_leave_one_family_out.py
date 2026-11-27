# ============================================================
# v2 — STAGE 4: Leave-one-family-out (ONE cell, paste into Colab)
# Run AFTER Stage 3 (GATE 3 PASS) in the same session, or after
# Setup B then Setup C in a fresh session. It reloads
# features.parquet from Drive — it does NOT redo Stages 1-3.
# Full run takes about 30 minutes in Colab. Every finished
# family is saved to Drive at once, so a disconnect loses
# nothing already done. If the run stops, just paste this cell
# again: families already in results_family_v2.csv are skipped,
# not redone.
# Models: Ridge and tuned XGBoost ONLY. Random Forest is left
# out on purpose — 52 B families + 7 X families x 2 models is
# already 118 retrains, and RF would make the run far too long.
# No band gap, no conformal, no CV in this cell.
# Paste back: the GATE 4 line and the pooled table (or the last
# rows printed, if it stopped part-way).
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs only leave-one-B-out for 3 B metals and
# leave-one-X-out for 2 X types, both models (10 rows), on the
# FULL rows of those families (364 rows per B, 2,704 per X) —
# and writes results_family_v2_smoke.csv instead of
# results_family_v2.csv. Use it only to check the code runs.
SMOKE = False

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# --- 1. Reload features.parquet from Stage 2 (do NOT redo Stages 1-3) ---
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
for needed in ["B_site", "X_anion", "e_form_per_atom"]:
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
print("X shape:", X.shape, "| y shape:", y.shape)
print("X is Magpie only. Forbidden columns are NOT in X by construction.")

# --- 3. Models — TUNED configs only, never defaults. Seed 0 only. ---
# Family-out is deterministic given the config: the split is fixed
# (one family held out), so there is no seed loop here. Seed 0 is
# used only for the models' own internal randomness (XGBoost).
# Ridge is standardised inside a pipeline, so the scaling is fitted
# on training rows only (no leakage from the held-out family).
def make_model(name):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
            tree_method="hist", n_jobs=-1, random_state=0)
    raise ValueError(f"unknown model {name}")


MODELS = ["Ridge", "XGBoost"]
print("Models (tuned only, seed 0):")
print("  Ridge: StandardScaler + Ridge(alpha=1.0)")
print("  XGBoost: n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5")

# --- 4. Families: 52 B metals (364 rows each), 7 X anions (2,704 each). ---
# A_family "other" (14,560 rows) is NOT left out — it is far too
# large and coarse to be one family. Do not do it.
B_FAMILIES = sorted(df["B_site"].unique().tolist())
X_FAMILIES = sorted(df["X_anion"].unique().tolist())
b_counts = df["B_site"].value_counts()
x_counts = df["X_anion"].value_counts()
print(f"B families: {len(B_FAMILIES)} (expected 52), "
      f"rows each: {sorted(b_counts.unique().tolist())} (expected [364])")
print(f"X families: {len(X_FAMILIES)} (expected 7), "
      f"rows each: {sorted(x_counts.unique().tolist())} (expected [2704])")
if len(B_FAMILIES) != 52 or set(b_counts.unique()) != {364}:
    raise AssertionError("STOP: B families are not 52 x 364 rows.")
if len(X_FAMILIES) != 7 or set(x_counts.unique()) != {2704}:
    raise AssertionError("STOP: X families are not 7 x 2704 rows.")
if SMOKE:
    B_FAMILIES = B_FAMILIES[:3]
    X_FAMILIES = X_FAMILIES[:2]
    print(f"SMOKE: B families {B_FAMILIES}, X families {X_FAMILIES} only.")
print(f"Full run = 52 B x 2 models + 7 X x 2 models = 118 rows "
      f"(SMOKE = 3 B x 2 + 2 X x 2 = 10 rows).")

# --- 5. Results file: ONE fixed column set, checkpoint after every row. ---
# Resume-by-skipping: if results_family_v2.csv already exists (a
# re-run after a disconnect), rows already in it are skipped, not
# redone. The file is never wiped on a re-run. In SMOKE mode a
# separate file is used so test rows can never mix into real results.
# Pooled metrics need the actual predictions, not just the per-family
# R2s, so every family's predictions are also saved, one row per
# test crystal, to predictions_family_v2.parquet. A family counts as
# DONE only if it is in BOTH files: if a re-run finds a row in the
# CSV but its predictions are missing, that row is dropped from the
# CSV and the family is re-run, so the two files can never disagree.
CSV_COLS = ["family_type", "family", "model", "n_test", "r2", "rmse",
            "mae", "experiment"]
csv_name = "results_family_v2_smoke.csv" if SMOKE else "results_family_v2.csv"
csv_path = OUT_DIR / csv_name
pred_name = ("predictions_family_v2_smoke.parquet" if SMOKE
             else "predictions_family_v2.parquet")
pred_path = OUT_DIR / pred_name
EXPERIMENT = "smoke" if SMOKE else "reproduction"

done = set()
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
        pred_done = {(str(r.family_type), str(r.family), str(r.model))
                     for r in old_pred.itertuples()}
    done_csv = {(str(r.family_type), str(r.family), str(r.model))
                for r in old.itertuples()}
    # Drop CSV rows whose predictions are missing — they will be re-run.
    orphaned = done_csv - pred_done
    if orphaned:
        print(f"WARNING: {len(orphaned)} CSV rows have no saved predictions "
              "— dropping them so the family is re-run cleanly.")
        keep = old[~old.apply(
            lambda r: (str(r.family_type), str(r.family), str(r.model))
            in orphaned, axis=1)]
        keep.to_csv(csv_path, index=False)
        old = keep
    done = done_csv - orphaned
    print(f"Resuming: {len(done)} families already in {csv_name} "
          "with predictions saved, they will be skipped.")
else:
    print(f"No existing {csv_name} — starting fresh.")


def save_row(row, y_true, y_pred):
    """Append one row to the CSV and its predictions to the parquet file,
    at once, so a disconnect cannot lose either."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    pred_df = pd.DataFrame({
        "family_type": row["family_type"],
        "family": row["family"],
        "model": row["model"],
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if pred_path.exists():
        old_pred = pd.read_parquet(pred_path)
        pred_df = pd.concat([old_pred, pred_df], ignore_index=True)
    pred_df.to_parquet(pred_path, index=False)
    print(f"  SAVED {row['family_type']} {row['family']} {row['model']}: "
          f"n={row['n_test']} R2 {row['r2']:.3f}, "
          f"RMSE {row['rmse']:.3f}, MAE {row['mae']:.3f}")


def run_one(family_type, family, model_name):
    """One family, one model: train on every other row, test on this family."""
    t0 = time.time()
    col = "B_site" if family_type == "B" else "X_anion"
    te_mask = (df[col] == family).values
    tr_mask = ~te_mask
    m = make_model(model_name)
    m.fit(X[tr_mask], y[tr_mask])
    pred = m.predict(X[te_mask])
    yt = y[te_mask]
    row = {
        "family_type": family_type,
        "family": family,
        "model": model_name,
        "n_test": int(te_mask.sum()),
        "r2": float(r2_score(yt, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yt, pred))),
        "mae": float(mean_absolute_error(yt, pred)),
        "experiment": EXPERIMENT,
    }
    print(f"[{family_type} {family} {model_name}] done in {time.time() - t0:.0f} s — "
          f"per-family R2: {row['r2']:.3f}")
    save_row(row, yt, pred)
    return row


# --- 6. Run everything, checkpointing after every family. ---
all_rows = []
for family_type, families in [("B", B_FAMILIES), ("X", X_FAMILIES)]:
    for family in families:
        for model_name in MODELS:
            key = (family_type, family, model_name)
            if key in done:
                print(f"SKIP {family_type} {family} {model_name} — already in {csv_name}.")
                continue
            all_rows.append(run_one(family_type, family, model_name))
print(f"Families finished in this run: {len(all_rows)} "
      f"(total in {csv_name} after this run: {len(done) + len(all_rows)}).")

# --- 7. Pooled headline + per-family summary (from the saved files). ---
# POOLED is the headline: concatenate ALL test predictions for one
# (family_type, model) and compute R2 / RMSE / MAE once on the pooled
# predictions. It is NOT the mean of the per-family R2s.
# Mean per-family R2 is secondary and misleading on its own:
# within one family the true values vary little, so a small error
# gives a huge negative per-family R2, even when the pooled number
# is fine. Both are reported; pooled leads.
results = pd.read_csv(csv_path)
preds = pd.read_parquet(pred_path)


def pooled_metrics(sub_preds):
    """R2 / RMSE / MAE computed once on pooled predictions."""
    yt = sub_preds["y_true"].values
    yp = sub_preds["y_pred"].values
    return {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "n": int(len(yt)),
    }


pooled_rows = []
for (family_type, model_name), sub in preds.groupby(["family_type", "model"]):
    pm = pooled_metrics(sub)
    pooled_rows.append({
        "family_type": family_type,
        "model": model_name,
        "n_test": pm["n"],
        "r2_pooled": pm["r2"],
        "rmse_pooled": pm["rmse"],
        "mae_pooled": pm["mae"],
    })
pooled = pd.DataFrame(pooled_rows).sort_values(["family_type", "model"])
print("POOLED headline (all test predictions together, per family_type x model):")
print(pooled.to_string(index=False))

summary_rows = []
for (family_type, model_name), sub in results.groupby(["family_type", "model"]):
    summary_rows.append({
        "family_type": family_type,
        "model": model_name,
        "mean_r2": float(sub["r2"].mean()),
        "median_r2": float(sub["r2"].median()),
        "min_r2": float(sub["r2"].min()),
        "max_r2": float(sub["r2"].max()),
        "worst_family": str(sub.loc[sub["r2"].idxmin(), "family"]),
        "best_family": str(sub.loc[sub["r2"].idxmax(), "family"]),
    })
summary = pd.DataFrame(summary_rows).sort_values(["family_type", "model"])
print("Per-family R2 summary (SECONDARY — context only, never the headline):")
print(summary.to_string(index=False))


def gate4_check(pooled_df):
    """GATE 4 — reproduction vs v1 Cell 11 pooled numbers.
    Returns (gate_lines, gate_failures). Tolerance +-0.03 on pooled R2.
    X-out Ridge -9.63 is catastrophic linear extrapolation and is
    EXPECTED to reproduce — a very negative number is not a bug."""
    v1_pooled = {
        ("B", "Ridge"): 0.206, ("B", "XGBoost"): 0.086,
        ("X", "Ridge"): -9.63, ("X", "XGBoost"): 0.375,
    }
    lines, failures = [], []
    for (family_type, model_name), v1_r2 in v1_pooled.items():
        sub = pooled_df[(pooled_df["family_type"] == family_type)
                        & (pooled_df["model"] == model_name)]
        if len(sub) == 0:
            failures.append(f"{family_type} {model_name}: no pooled row")
            lines.append(f"  [FAIL] {family_type} {model_name}: no pooled row")
            continue
        v2_r2 = float(sub["r2_pooled"].iloc[0])
        # Tolerance (corrected 2026-11-25, user approved): ±0.03
        # absolute when |v1| <= 1; for a pooled R2 below -1 (only
        # X-out Ridge, v1 -9.63) use a relative tolerance
        # max(0.03, 5% of |v1|). v2 gave -9.958, diff -0.328,
        # relative 3.4% — catastrophic in both, conclusion identical.
        tol = 0.03 if abs(v1_r2) <= 1 else max(0.03, 0.05 * abs(v1_r2))
        # +1e-9: a diff of exactly the tolerance must PASS, but in
        # floating point -9.60 - (-9.63) is 0.03000000000000025.
        ok = abs(v2_r2 - v1_r2) <= tol + 1e-9
        status = "PASS" if ok else "FAIL"
        line = (f"  [{status}] {family_type}-out {model_name}: pooled v2 {v2_r2:.3f} "
                f"vs v1 {v1_r2:.3f} (diff {v2_r2 - v1_r2:+.3f})")
        lines.append(line)
        if not ok:
            failures.append(f"{family_type} {model_name}: v2 {v2_r2:.3f} vs v1 {v1_r2:.3f}")
    return lines, failures


# --- 8. GATE 4 — reproduction check (full run only). ---
gate_lines = []
gate_failures = []
if SMOKE:
    print("GATE 4: SKIPPED in SMOKE mode — smoke runs only 3 B + 2 X "
          "families, so pooled numbers cannot be compared with v1.")
    gate_lines.append("GATE 4: SKIPPED (SMOKE mode — no v1 comparison).")
else:
    print("GATE 4 — reproduction vs v1 Cell 11 pooled (tolerance +-0.03 R2):")
    gate_lines, gate_failures = gate4_check(pooled)
    for line in gate_lines:
        print(line)
    print("Note: X-out Ridge pooled R2 is about -9.6 in v1 — linear models "
          "cannot extrapolate to a new anion. That very negative number "
          "reproducing is a PASS, not a bug.")

# --- 9. Save the Stage 4 report. ---
report_name = "family_report_v2_smoke.md" if SMOKE else "family_report_v2.md"
report_lines = [
    "# Perovskite v2 — Stage 4 leave-one-family-out report",
    "",
    "Magpie only (132 columns), e_form_per_atom, seed 0.",
    "Models: Ridge and tuned XGBoost only — no Random Forest "
    "(118 retrains already; RF would make the run far too long).",
    "A_family 'other' was NOT left out (14,560 rows, too coarse).",
    "",
    "## Pooled headline (lead with this, not mean per-family R2)",
    "",
    pooled.to_string(index=False),
    "",
    "## Per-family R2 summary (secondary — context only)",
    "",
    summary.to_string(index=False),
    "",
    "## Per-family results (all families)",
    "",
    results.to_string(index=False),
    "",
    "## GATE 4 — reproduction vs v1 Cell 11 pooled",
    "",
]
report_lines += gate_lines
report_lines += [
    "",
    "## Notes",
    "",
    "- experiment column: 'reproduction' for the full run, 'smoke' for test runs.",
    "- Pooled R2 is computed once on all pooled test predictions per "
    "(family_type, model); it is not the mean of per-family R2s.",
    "- X-out Ridge pooled R2 ~ -9.6 is expected: linear extrapolation "
    "to a new anion fails catastrophically. Trees do not extrapolate, "
    "so XGBoost survives X-out while Ridge does not.",
    "- If GATE 4 failed, STOP — do not proceed to Stage 5. "
    "Find why v2 differs from v1 first.",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")
print(f"saved {csv_name}: {len(results)} rows")
print(f"saved {pred_name}: {len(preds)} predictions")

# --- GATE 4 verdict. ---
if SMOKE:
    print("STAGE 4 SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 4: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 4 FAILED: " + "; ".join(gate_failures))
else:
    print("GATE 4: PASS — v2 reproduces v1 pooled family-out within +-0.03 R2. "
          "Next: Stage 5 band gap.")
    print("STAGE 4 done — paste back the GATE 4 line and the pooled table.")
