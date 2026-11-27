# ============================================================
# v2 — STAGE 3c: No-B-information ablation, leave-one-B-out ONLY
# (ONE cell, paste into Colab)
# Run AFTER Stage 3b (GATE 3b PASS, corrected — site-aware pooled
# B-out XGBoost R2 0.733). It does NOT need Stages 4-8, only
# features.parquet + audited.parquet + features_siteaware.parquet
# from Stages 1-2 and 3b. Run in the same session, or after
# Setup B then Setup C in a fresh session. It reloads the three
# parquet files from Drive — it does NOT rebuild features.
# Full run takes about 5 minutes in Colab. Every finished family
# is saved to Drive at once, so a disconnect loses nothing
# already done. If the run stops, just paste this cell again:
# families already in results_siteaware_nob_v2.csv are skipped,
# not redone.
# WHY THIS VARIANT CAN STILL WORK AT ALL: Magpie is
# composition-only, so it still carries B's contribution in
# aggregate — the average over the whole formula includes the B
# element. And with A and X given explicitly (their own physics
# numbers), the model can work out B's part by subtraction:
# what is left of the composition after A and X are accounted
# for must be B. So "no B information" really means "no
# site-resolved B information" — the B element is still inside
# the Magpie averages, just never named or measured on its own
# site. That is exactly what this test measures: how much of the
# 0.733 came from naming/measuring B on its own site, and how
# much came from A + X + Magpie alone.
# WHAT IS DROPPED: every column that names or measures B on its
# own site — the 52 B one-hot columns, the 12 B-site physics
# columns, AND all 5 derived columns (tolerance factor,
# octahedral factor, rA-rB, EN_A-EN_B, EN_B-EN_X) because every
# one of them is built from a B-site number. 69 columns dropped.
# WHAT IS KEPT: Magpie 132 + A-site physics 12 + X-site physics
# 12 + X one-hot 7 = 163 columns.
# The 8-family diagnostic said: no-B-info pooled 0.586 vs
# full site-aware 0.484 on Mg/Te/W/Cr/Ni/Al/B/Re — B-physics
# helped some families (W, Re) and hurt others (Cr). This cell
# settles it on ALL 52 families before the paper's mechanism
# story is written. No Ridge here (Ridge already failed on
# site-aware, -0.193) — one clean XGBoost comparison.
# NO random CV, no band gap, no conformal in this cell.
# Paste back: the GATE 3c line, the pooled table, and the
# comparison table (or the last rows printed, if it stopped
# part-way).
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs only leave-one-B-out for 3 B metals (Mg, Te, W),
# tuned XGBoost only (3 rows), on the FULL 364-row families — and
# writes results_siteaware_nob_v2_smoke.csv instead of
# results_siteaware_nob_v2.csv. Use it only to check the code runs.
SMOKE = False

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# --- 1. Reload the three parquet files (do NOT rebuild features). ---
feat_path = OUT_DIR / "features.parquet"
audit_path = OUT_DIR / "audited.parquet"
sa_path = OUT_DIR / "features_siteaware.parquet"
for p, name in [(feat_path, "features.parquet"),
                (audit_path, "audited.parquet"),
                (sa_path, "features_siteaware.parquet")]:
    if not p.exists():
        raise FileNotFoundError(
            f"{name} not found in OUT_DIR. Run Stages 1-2 (and 3b for "
            "features_siteaware.parquet) first, then this cell again."
        )
df = pd.read_parquet(feat_path)
audited = pd.read_parquet(audit_path)
sa = pd.read_parquet(sa_path)
print("features.parquet shape:", df.shape, "(expected (18928, 150))")
print("audited.parquet shape:", audited.shape, "(expected (18928, 18))")
print("features_siteaware.parquet shape:", sa.shape, "(expected (18928, 250))")
if df.shape != (18928, 150) or audited.shape != (18928, 18) or sa.shape != (18928, 250):
    raise AssertionError(
        "STOP: input shapes are not (18928, 150) / (18928, 18) / "
        "(18928, 250). Re-run Stages 1-2 and 3b before modelling."
    )
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
print("Magpie columns:", len(magpie_cols), "(expected 132)")
for needed in ["A_site", "B_site", "X_anion", "e_form_per_atom"]:
    if needed not in df.columns:
        raise AssertionError(f"STOP: needed column {needed} is missing from features.parquet.")
# Row-by-row alignment between all three files.
align = bool(
    len(df) == len(audited) == len(sa)
    and (df["formula_ordered"].values == audited["formula_ordered"].values).all()
    and (df["formula_ordered"].values == sa["formula_ordered"].values).all()
    and np.allclose(df["e_form"].values, audited["e_form"].values)
    and np.allclose(df["e_form"].values, sa["e_form"].values)
)
print("alignment features vs audited vs site-aware (formula_ordered, e_form):", align)
if not align:
    raise AssertionError(
        "ALIGNMENT FAILED: the three parquet files do not match row by "
        "row. STOP — do not build the no-B-info feature set."
    )

# --- 2. No-B-info feature set: Magpie + A physics + X physics + X one-hot. ---
# No DFT columns in X, ever: vbm/cbm define the band gap directly,
# fermi level / fermi width / mu_b are DFT outputs, e_form /
# band_gap / e_form_per_atom are targets. Including any is cheating.
FORBIDDEN = ["vbm", "cbm", "fermi level", "fermi width", "mu_b",
             "e_form", "band_gap", "e_form_per_atom"]
# The 12 per-site property keys, EXACTLY as Stage 3b defined them
# (same names, same order) — so the kept A/X columns are the same
# numbers Stage 3b used, taken from the saved site-aware frame.
PROP_KEYS = ["Z", "mass", "EN", "r_ion", "r_atom", "ionE", "EA",
             "val_s", "val_p", "val_d", "val_f", "mendeleev"]
B_CATS = sorted(df["B_site"].unique().tolist())
X_CATS = sorted(df["X_anion"].unique().tolist())
A_PHYS = [f"A_{k}" for k in PROP_KEYS]
X_PHYS = [f"X_{k}" for k in PROP_KEYS]
OH_X = [f"X_{c}" for c in X_CATS]
# Everything B-related that Stage 3b built — all of it is dropped.
DROPPED_B_ONEHOT = [f"B_{c}" for c in B_CATS]
DROPPED_B_PHYS = [f"B_{k}" for k in PROP_KEYS]
DROPPED_DERIVED = ["tolerance_factor", "octahedral_factor", "rA_minus_rB",
                   "EN_A_minus_EN_B", "EN_B_minus_EN_X"]
dropped_cols = DROPPED_B_ONEHOT + DROPPED_B_PHYS + DROPPED_DERIVED
kept_cols = magpie_cols + A_PHYS + X_PHYS + OH_X
print(f"KEPT columns: {len(kept_cols)} "
      f"(Magpie {len(magpie_cols)} + A physics {len(A_PHYS)} "
      f"+ X physics {len(X_PHYS)} + X one-hot {len(OH_X)})")
print(f"DROPPED B columns: {len(dropped_cols)} "
      f"(B one-hot {len(DROPPED_B_ONEHOT)} + B physics {len(DROPPED_B_PHYS)} "
      f"+ derived {len(DROPPED_DERIVED)})")
# The drop must be REAL, not a name mismatch: every dropped column
# must actually exist in the site-aware frame, and every kept column
# must exist there too. If a name were wrong, the drop would silently
# miss a B column — this check stops that.
missing_kept = [c for c in kept_cols if c not in sa.columns]
missing_dropped = [c for c in dropped_cols if c not in sa.columns]
print("kept columns missing from features_siteaware.parquet (must be []):", missing_kept)
print("dropped columns missing from features_siteaware.parquet (must be []):", missing_dropped)
if missing_kept or missing_dropped:
    raise AssertionError(
        "STOP: column-name mismatch with features_siteaware.parquet — "
        f"missing kept {missing_kept}, missing dropped {missing_dropped}. "
        "The Stage 3b column names changed; do not guess, find why."
    )
# The kept set must contain NO B-site marker at all. B-site markers
# are: any column starting with "B_" (B one-hot B_Ag..., B physics
# B_Z..., and the B_site ID itself) and any of the 5 derived columns
# (they are built from B-site numbers even though their names do not
# start with B_). This uses the same naming conventions as Stage 3b.
b_markers = [c for c in kept_cols if c.startswith("B_") or c in DROPPED_DERIVED]
print("B-site markers left in kept set (must be []):", b_markers)
if b_markers:
    raise AssertionError(f"STOP: B information leaked into the kept set: {b_markers}")
leak = [c for c in kept_cols if c in FORBIDDEN]
print("Leakage columns in no-B-info X (should be []):", leak)
if leak:
    raise AssertionError(f"STOP: forbidden columns in no-B-info X: {leak}")
X_nob = sa[kept_cols].values
print(f"no-B-info X shape: {X_nob.shape} (expected (18928, {len(kept_cols)}))")
print("no-B-info X NaN cells:", int(np.isnan(X_nob).sum()), "(expected 0)")
if X_nob.shape != (18928, len(kept_cols)) or int(np.isnan(X_nob).sum()) != 0:
    raise AssertionError("STOP: no-B-info X has the wrong shape or NaN cells.")
y = df["e_form_per_atom"].values
print("y = e_form_per_atom, n =", len(y))

# --- 3. Model — TUNED XGBoost only, seed 0. No Ridge (it already ---
# failed on site-aware, pooled -0.193). Leave-one-B-out is
# deterministic given the config, so there is no seed loop.
def make_model():
    return XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
        tree_method="hist", n_jobs=-1, random_state=0)

print("Model (tuned only, seed 0):")
print("  XGBoost: n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5")

# --- 4. Families: 52 B metals, 364 rows each. ---
B_FAMILIES = sorted(df["B_site"].unique().tolist())
b_counts = df["B_site"].value_counts()
print(f"B families: {len(B_FAMILIES)} (expected 52), "
      f"rows each: {sorted(b_counts.unique().tolist())} (expected [364])")
if len(B_FAMILIES) != 52 or set(b_counts.unique()) != {364}:
    raise AssertionError("STOP: B families are not 52 x 364 rows.")
if SMOKE:
    B_FAMILIES = ["Mg", "Te", "W"]
    print(f"SMOKE: B families {B_FAMILIES} only (Mg = good case, "
          "Te = v1 metal, W = hard failure case).")
print(f"Full run = 52 B x 1 model = 52 rows (SMOKE = 3 rows).")

# --- 5. Results file: ONE fixed column set, checkpoint after every row. ---
# Same pattern as Stage 3b: resume-by-skipping, separate smoke file
# so test rows can never mix into real results, and a family counts
# as DONE only if it is in BOTH the CSV and the predictions file.
CSV_COLS = ["feature_set", "family_type", "family", "model", "n_test",
            "r2", "rmse", "mae", "experiment"]
csv_name = ("results_siteaware_nob_v2_smoke.csv" if SMOKE
            else "results_siteaware_nob_v2.csv")
csv_path = OUT_DIR / csv_name
pred_name = ("predictions_siteaware_nob_v2_smoke.parquet" if SMOKE
             else "predictions_siteaware_nob_v2.parquet")
pred_path = OUT_DIR / pred_name
EXPERIMENT = "smoke" if SMOKE else "siteaware_nob_b_out"

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
        pred_done = {(str(r.feature_set), str(r.family_type), str(r.family), str(r.model))
                     for r in old_pred.itertuples()}
    done_csv = {(str(r.feature_set), str(r.family_type), str(r.family), str(r.model))
                for r in old.itertuples()}
    orphaned = done_csv - pred_done
    if orphaned:
        print(f"WARNING: {len(orphaned)} CSV rows have no saved predictions "
              "— dropping them so the family is re-run cleanly.")
        keep = old[~old.apply(
            lambda r: (str(r.feature_set), str(r.family_type), str(r.family), str(r.model))
            in orphaned, axis=1)]
        keep.to_csv(csv_path, index=False)
        old = keep
    done = done_csv - orphaned
    print(f"Resuming: {len(done)} rows already in {csv_name} "
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
        "feature_set": row["feature_set"],
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
    print(f"  SAVED {row['feature_set']} B {row['family']} {row['model']}: "
          f"n={row['n_test']} R2 {row['r2']:.3f}, "
          f"RMSE {row['rmse']:.3f}, MAE {row['mae']:.3f}")


def run_one(family):
    """One B family, tuned XGBoost, no-B-info features: train on every
    other row, test on this family's 364 rows."""
    t0 = time.time()
    te_mask = (df["B_site"] == family).values
    tr_mask = ~te_mask
    m = make_model()
    m.fit(X_nob[tr_mask], y[tr_mask])
    pred = m.predict(X_nob[te_mask])
    yt = y[te_mask]
    row = {
        "feature_set": "nob",
        "family_type": "B",
        "family": family,
        "model": "XGBoost",
        "n_test": int(te_mask.sum()),
        "r2": float(r2_score(yt, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yt, pred))),
        "mae": float(mean_absolute_error(yt, pred)),
        "experiment": EXPERIMENT,
    }
    print(f"[nob B {family} XGBoost] done in {time.time() - t0:.0f} s — "
          f"per-family R2: {row['r2']:.3f}")
    save_row(row, yt, pred)
    return row


# --- 6. Run everything, checkpointing after every family. ---
all_rows = []
for family in B_FAMILIES:
    key = ("nob", "B", family, "XGBoost")
    if key in done:
        print(f"SKIP nob B {family} XGBoost — already in {csv_name}.")
        continue
    all_rows.append(run_one(family))
print(f"Rows finished in this run: {len(all_rows)} "
      f"(total in {csv_name} after this run: {len(done) + len(all_rows)}).")

# --- 7. Pooled headline + per-family summary (from the saved files). ---
# POOLED is the headline: concatenate ALL test predictions and
# compute R2 / RMSE / MAE once on the pooled predictions. It is NOT
# the mean of the per-family R2s.
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


nob_pooled = pooled_metrics(preds)
print("POOLED headline — no-B-info XGBoost (all test predictions together):")
print(f"  n_test={nob_pooled['n']}, R2={nob_pooled['r2']:.3f}, "
      f"RMSE={nob_pooled['rmse']:.3f}, MAE={nob_pooled['mae']:.3f}")
summary = {
    "mean_r2": float(results["r2"].mean()),
    "median_r2": float(results["r2"].median()),
    "min_r2": float(results["r2"].min()),
    "max_r2": float(results["r2"].max()),
    "worst_family": str(results.loc[results["r2"].idxmin(), "family"]),
    "best_family": str(results.loc[results["r2"].idxmax(), "family"]),
}
print("Per-family R2 summary (SECONDARY — context only, never the headline):")
print(f"  mean={summary['mean_r2']:.3f}, median={summary['median_r2']:.3f}, "
      f"min={summary['min_r2']:.3f} ({summary['worst_family']}), "
      f"max={summary['max_r2']:.3f} ({summary['best_family']})")

# --- 8. COMPARISON table — every number read from disk, never typed. ---
# Magpie-only and site-aware pooled XGBoost come from the Stage 3b
# predictions file (it holds BOTH feature sets, same families, same
# model — apples to apples). Full run: predictions_siteaware_v2.parquet;
# smoke run: predictions_siteaware_v2_smoke.parquet. If that file is
# missing, fall back to the Stage 4 predictions file for Magpie-only
# (predictions_family_v2.parquet, family_type B, model XGBoost).
# The Stage 3b CSV (results_siteaware_v2.csv) is also read, to check
# the row counts behind the comparison.
def _pooled_from_pred_file(path, feature_set, model="XGBoost"):
    """Pooled R2/RMSE/MAE for one feature_set in a predictions file."""
    if not path.exists():
        return None, f"{path.name} not found"
    p = pd.read_parquet(path)
    sub = p[(p["feature_set"] == feature_set) & (p["model"] == model)]
    if len(sub) == 0:
        return None, f"no {feature_set}/{model} rows in {path.name}"
    return pooled_metrics(sub), f"{path.name}"


sa_pred_name = ("predictions_siteaware_v2_smoke.parquet" if SMOKE
                else "predictions_siteaware_v2.parquet")
sa_pred_path = OUT_DIR / sa_pred_name
sa_csv_name = ("results_siteaware_v2_smoke.csv" if SMOKE
               else "results_siteaware_v2.csv")
sa_csv_path = OUT_DIR / sa_csv_name
comp_rows = []
comp_sources = []
mag_pooled, mag_src = _pooled_from_pred_file(sa_pred_path, "magpie")
if mag_pooled is None:
    # Fall back to the Stage 4 predictions file for Magpie-only.
    f4_path = OUT_DIR / ("predictions_family_v2_smoke.parquet" if SMOKE
                         else "predictions_family_v2.parquet")
    if f4_path.exists():
        p4 = pd.read_parquet(f4_path)
        sub4 = p4[(p4["family_type"] == "B") & (p4["model"] == "XGBoost")]
        if len(sub4) > 0:
            mag_pooled = pooled_metrics(sub4)
            mag_src = f4_path.name + " (Stage 4, B-out XGBoost)"
if mag_pooled is not None:
    comp_rows.append({"feature_set": "magpie", "model": "XGBoost",
                      "n_test": mag_pooled["n"], "r2_pooled": mag_pooled["r2"],
                      "rmse_pooled": mag_pooled["rmse"]})
    comp_sources.append(f"magpie XGBoost pooled from {mag_src}")
else:
    comp_sources.append(f"magpie XGBoost pooled UNAVAILABLE ({mag_src})")
site_pooled, site_src = _pooled_from_pred_file(sa_pred_path, "siteaware")
if site_pooled is not None:
    comp_rows.append({"feature_set": "siteaware", "model": "XGBoost",
                      "n_test": site_pooled["n"], "r2_pooled": site_pooled["r2"],
                      "rmse_pooled": site_pooled["rmse"]})
    comp_sources.append(f"site-aware XGBoost pooled from {site_src}")
else:
    comp_sources.append(f"site-aware XGBoost pooled UNAVAILABLE ({site_src})")
comp_rows.append({"feature_set": "nob", "model": "XGBoost",
                  "n_test": nob_pooled["n"], "r2_pooled": nob_pooled["r2"],
                  "rmse_pooled": nob_pooled["rmse"]})
comp_sources.append(f"no-B-info XGBoost pooled from {pred_name} (this run)")
comparison = pd.DataFrame(comp_rows)
print("COMPARISON — pooled B-out XGBoost R2 (all numbers read from disk):")
print(comparison.to_string(index=False))
for src in comp_sources:
    print("  source:", src)
if sa_csv_path.exists():
    sa_csv = pd.read_csv(sa_csv_path)
    print(f"  check: {sa_csv_name} has {len(sa_csv)} rows "
          f"(full run: 208; smoke: 12).")
else:
    print(f"  check: {sa_csv_name} NOT FOUND — comparison pooled values "
          "come from the predictions parquet only.")
# Which is best, and by how much — computed, not typed.
best_row = comparison.loc[comparison["r2_pooled"].idxmax()]
print(f"Best pooled B-out XGBoost: {best_row['feature_set']} "
      f"R2={best_row['r2_pooled']:.3f} (RMSE={best_row['rmse_pooled']:.3f})")
for _, r in comparison.iterrows():
    if r["feature_set"] != best_row["feature_set"]:
        print(f"  {r['feature_set']}: {best_row['r2_pooled'] - r['r2_pooled']:.3f} "
              f"below best ({r['feature_set']} R2={r['r2_pooled']:.3f})")

# Interpretation rule (pre-registered, before the run):
# - no-B-info pooled >= site-aware pooled: B-physics does not add
#   beyond A/X + Magpie on B-out.
# - site-aware higher by > 0.05: B-physics adds on B-out.
# - otherwise: B-physics adds little on B-out.
interpretation = None
if site_pooled is not None:
    if nob_pooled["r2"] >= site_pooled["r2"]:
        interpretation = ("B-physics does not add beyond A/X + Magpie on B-out")
    elif site_pooled["r2"] - nob_pooled["r2"] > 0.05:
        interpretation = ("B-physics adds on B-out")
    else:
        interpretation = ("B-physics adds little on B-out")
    print("INTERPRETATION:", interpretation)
else:
    print("INTERPRETATION: unavailable — site-aware pooled value could "
          "not be read from disk, so no comparison can be judged.")

# --- 9. GATE 3c — files complete + comparison table present. ---
# This gate does NOT judge pass/fail on the score itself — the score
# decides the mechanism story (see INTERPRETATION above), so the
# gate only checks the run is complete and the comparison is real.
gate_lines = []
gate_failures = []
if SMOKE:
    print("GATE 3c: SKIPPED in SMOKE mode — smoke runs only 3 B families, "
          "so pooled numbers cannot be compared with the full run.")
    gate_lines.append("GATE 3c: SKIPPED (SMOKE mode — no completeness check).")
else:
    expected_rows = 52
    expected_preds = 52 * 364
    if len(results) == expected_rows:
        gate_lines.append(f"  [PASS] {csv_name}: {len(results)} rows (expected {expected_rows})")
    else:
        gate_failures.append(f"{csv_name} has {len(results)} rows, expected {expected_rows}")
        gate_lines.append(f"  [FAIL] {csv_name}: {len(results)} rows (expected {expected_rows})")
    if len(preds) == expected_preds:
        gate_lines.append(f"  [PASS] {pred_name}: {len(preds)} predictions (expected {expected_preds})")
    else:
        gate_failures.append(f"{pred_name} has {len(preds)} predictions, expected {expected_preds}")
        gate_lines.append(f"  [FAIL] {pred_name}: {len(preds)} predictions (expected {expected_preds})")
    if len(comparison) == 3 and mag_pooled is not None and site_pooled is not None:
        gate_lines.append("  [PASS] comparison table present: magpie, site-aware and "
                          "no-B-info pooled values all read from disk")
    else:
        gate_failures.append("comparison table incomplete — a pooled value could not be read from disk")
        gate_lines.append("  [FAIL] comparison table incomplete (see sources above)")
    if interpretation is not None:
        gate_lines.append(f"  [PASS] interpretation printed: {interpretation}")
    else:
        gate_failures.append("no interpretation — site-aware pooled value missing")
        gate_lines.append("  [FAIL] no interpretation printed")

# --- 10. Save the Stage 3c report. ---
report_name = ("siteaware_nob_report_v2_smoke.md" if SMOKE
               else "siteaware_nob_report_v2.md")
report_lines = [
    "# Perovskite v2 — Stage 3c no-B-information ablation report",
    "",
    "No-B-info features = Magpie (132) + A-site physics (12) + "
    "X-site physics (12) + X one-hot (7) = 163 columns. Dropped: "
    "B one-hot (52) + B-site physics (12) + all 5 derived columns "
    "(tolerance_factor, octahedral_factor, rA_minus_rB, "
    "EN_A_minus_EN_B, EN_B_minus_EN_X) = 69 columns. No Ridge, "
    "tuned XGBoost only, leave-one-B-out only, seed 0.",
    "",
    "Why it can still work: Magpie is composition-only, so it still "
    "carries B's contribution in aggregate, and with A and X given "
    "explicitly the model can work out B's part by subtraction.",
    "",
    "## Feature groups",
    "",
    f"- Kept: Magpie {len(magpie_cols)}, A physics {len(A_PHYS)}, "
    f"X physics {len(X_PHYS)}, X one-hot {len(OH_X)} — total {len(kept_cols)}",
    f"- Dropped: B one-hot {len(DROPPED_B_ONEHOT)}, B physics "
    f"{len(DROPPED_B_PHYS)}, derived {len(DROPPED_DERIVED)} — total {len(dropped_cols)}",
    "- B-site markers left in kept set: [] (asserted in code)",
    "- Leakage columns in no-B-info X: [] (asserted in code)",
    "",
    "## Pooled headline — no-B-info XGBoost",
    "",
    f"- n_test={nob_pooled['n']}, R2={nob_pooled['r2']:.3f}, "
    f"RMSE={nob_pooled['rmse']:.3f}, MAE={nob_pooled['mae']:.3f}",
    f"- Per-family R2: mean={summary['mean_r2']:.3f}, "
    f"median={summary['median_r2']:.3f}, min={summary['min_r2']:.3f} "
    f"({summary['worst_family']}), max={summary['max_r2']:.3f} "
    f"({summary['best_family']})",
    "",
    "## Comparison — pooled B-out XGBoost R2 (all numbers read from disk)",
    "",
    comparison.to_string(index=False),
    "",
]
for src in comp_sources:
    report_lines.append(f"- source: {src}")
report_lines += [
    "",
    "## Interpretation",
    "",
    f"- {interpretation if interpretation is not None else 'unavailable — site-aware pooled value missing'}",
    "",
    "## GATE 3c",
    "",
]
report_lines += gate_lines
report_lines += [
    "",
    "## Notes",
    "",
    "- experiment column: 'siteaware_nob_b_out' for the full run, 'smoke' for test runs.",
    "- Pooled R2 is computed once on all pooled test predictions; "
    "it is not the mean of per-family R2s.",
    "- Stage 3b reference numbers (full Colab run): site-aware pooled "
    "B-out XGBoost 0.733, Ridge -0.193; Magpie-only 0.206 / 0.086. "
    "These are quoted here for context only — the comparison table "
    "above recomputes them from the files on disk.",
    "- The 8-family diagnostic (Mg/Te/W/Cr/Ni/Al/B/Re) gave no-B-info "
    "pooled 0.586 vs site-aware 0.484 on those 8 families only. "
    "Do NOT quote the diagnostic as the Stage 3c result.",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")
print(f"saved {csv_name}: {len(results)} rows")
print(f"saved {pred_name}: {len(preds)} predictions")

# --- GATE 3c verdict. ---
if SMOKE:
    print("STAGE 3c SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 3c: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 3c FAILED: " + "; ".join(gate_failures))
else:
    print(f"GATE 3c: PASS — 52 rows, pooled R2={nob_pooled['r2']:.3f}, "
          f"comparison complete. {interpretation}. Next: Stage 8 paper draft.")
    print("STAGE 3c done — paste back the GATE 3c line and the comparison table.")
