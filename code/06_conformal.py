# ============================================================
# v2 — STAGE 6: Conformal uncertainty (ONE cell, paste into Colab)
# Run AFTER Stage 5 (GATE 5 PASS) in the same session, or after
# Setup B then Setup C in a fresh session. It reloads
# features.parquet from Drive — it does NOT redo Stages 1-5.
# Full run takes about 5-10 minutes in Colab. It is ONE write:
# all 4 rows are saved to results_conformal_v2.csv at the end,
# and a backup is saved first if the file already exists.
# Model: tuned XGBoost regressor ONLY, Magpie only (132 columns).
# Exact split conformal: residuals are sorted, rank
# k = ceil((n_cal + 1) * 0.90), q = r[k-1]. NO interpolated
# np.quantile — that was the v1 review issue, fixed here.
# Two tests: (1) random 60/20/20 split, (2) family-out shift
# for THREE B metals — Te, W, Mg — with the Te leak FIXED:
# train + calibrate use ONLY rows with B != the held-out metal,
# so metal-in-train and metal-in-cal are both 0. The old v2/v4
# leak (training on ~218 Te rows, then testing on all 364 Te)
# must never return — the cell asserts it in code.
# Framing, read this before quoting coverage: random coverage is
# an empirical check of a 90% promise under exchangeability.
# Family-out coverage is an empirical OBSERVATION under domain
# shift, where the guarantee does not apply. Coverage is NOT
# accuracy — a wide interval can cover while the point
# predictions are worse than the mean.
# No band gap, no family-out CV, no formation-energy CV here.
# Paste back: the GATE 6 line and the 4-row table.
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs the random test on the first 4,000 rows only
# and the family-out test for Te only (its 364 rows), seed 0,
# and writes results_conformal_v2_smoke.csv instead of
# results_conformal_v2.csv. Use it only to check the code runs.
SMOKE = False

import math
import shutil
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

# --- 1. Reload features.parquet from Stage 2 (do NOT redo Stages 1-5) ---
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
for needed in ["B_site", "e_form_per_atom"]:
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
X_all = df[magpie_cols].values
y_all = df["e_form_per_atom"].values
B_all = df["B_site"].values
print("X shape:", X_all.shape, "| y shape:", y_all.shape)
print("X is Magpie only. Forbidden columns are NOT in X by construction.")

# --- 3. Model — tuned XGBoost regressor only, seed 0. ---
# Default XGBoost scored below the mean on formation energy in v1.
# Never ship defaults again.
def make_reg():
    return XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
        tree_method="hist", n_jobs=-1, random_state=0)

print("Model (tuned only): XGBoost n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5, seed 0")

# --- 4. Exact split conformal — the quantile, done by hand. ---
# Residuals on calibration: r = |y - yhat|. Sort ascending.
# Rank k = ceil((n_cal + 1) * 0.90). q = r[k-1] if k <= n_cal,
# else r[n_cal-1] (the largest residual). Interval = yhat +- q.
# Mean width = 2q. This is the EXACT order-statistic quantile,
# not np.quantile (which interpolates between residuals and was
# the review's Issue 5). k and q are printed for every test so
# the quantile can be checked by hand.
def exact_quantile(resid, level=0.90):
    """Exact conformal quantile. Returns (q, k, n_cal)."""
    r = np.sort(np.abs(np.asarray(resid, dtype=float)))
    n = len(r)
    k = int(math.ceil((n + 1) * level))
    q = float(r[k - 1]) if k <= n else float(r[n - 1])
    return q, k, n


def conformal_metrics(y_true, y_pred, q):
    """Coverage, width, R2, RMSE, MAE for one test set, given q."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    covered = np.abs(yt - yp) <= q
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(2 * q),
        "r2_test": float(r2_score(yt, yp)),
        "rmse_test": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae_test": float(mean_absolute_error(yt, yp)),
    }


def run_conformal(X_tr, y_tr, X_cal, y_cal, X_te, y_te, label):
    """Fit on train, calibrate q on calibration, score on test."""
    m = make_reg()
    m.fit(X_tr, y_tr)
    q, k, n_cal = exact_quantile(y_cal - m.predict(X_cal), level=0.90)
    metrics = conformal_metrics(y_te, m.predict(X_te), q)
    print(f"[{label}] n_cal={n_cal}, k={k}, q={q:.4f} eV/atom, "
          f"coverage={metrics['coverage']:.3f}, "
          f"mean width={metrics['mean_width']:.3f} eV/atom, "
          f"R2 on test={metrics['r2_test']:.3f}, "
          f"RMSE={metrics['rmse_test']:.3f}, MAE={metrics['mae_test']:.3f}")
    return q, k, n_cal, metrics


# --- 5. The tests. ---
# Random split: shuffle all rows with seed 0, then take
# train = first int(0.6 * N), cal = next int(0.2 * N),
# test = the rest. This is the v1 Cell 15 split, so the random
# row is a REPRODUCTION: v1 had N_train 11,356, N_cal 3,785,
# N_test 3,787. Same int() truncation, same seed.
# Family-out: for each held-out metal, train + calibrate use
# ONLY rows with B != that metal (same 60/20 int split on the
# shuffled non-metal rows; the leftover non-metal rows are
# unused and counted in the log), test = all 364 rows of that
# metal. metal-in-train and metal-in-cal are asserted to be 0.
if SMOKE:
    # Random test on the first 4,000 rows only; family-out Te
    # still tests on ALL 364 Te rows (train/cal from the full
    # non-Te rows) — that is what the task specifies, and the
    # fits are still seconds.
    X_rand, y_rand = X_all[:4000], y_all[:4000]
    SHIFT_METALS = ["Te"]
    print("SMOKE: random test on the first 4,000 rows only; "
          "family-out Te only (all 364 Te rows); seed 0.")
else:
    X_rand, y_rand = X_all, y_all
    SHIFT_METALS = ["Te", "W", "Mg"]
    print("SHIFT metals: Te (the v1 metal), W (a hard failure case "
          "from Stage 4), Mg (a good case from Stage 4).")

CSV_COLS = ["test", "model", "n_train", "n_cal", "n_test", "q",
            "coverage", "mean_width", "r2_test", "rmse_test", "mae_test",
            "metal_in_train", "metal_in_cal", "experiment"]
rows = []

# --- 5a. Random test (reproduction of v1 Cell 15). ---
rng = np.random.RandomState(0)
perm = rng.permutation(len(y_rand))
n_tr = int(0.6 * len(y_rand))
n_cal = int(0.2 * len(y_rand))
tr_idx, cal_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr + n_cal], perm[n_tr + n_cal:]
print(f"Random split: N={len(y_rand)}, train={len(tr_idx)}, "
      f"cal={len(cal_idx)}, test={len(te_idx)} "
      "(v1 full run: 11356 / 3785 / 3787)")
q, k, n_cal_out, metrics = run_conformal(
    X_rand[tr_idx], y_rand[tr_idx], X_rand[cal_idx], y_rand[cal_idx],
    X_rand[te_idx], y_rand[te_idx], "random")
rows.append({
    "test": "random", "model": "XGBoost",
    "n_train": int(len(tr_idx)), "n_cal": int(n_cal_out),
    "n_test": int(len(te_idx)), "q": q,
    "coverage": metrics["coverage"], "mean_width": metrics["mean_width"],
    "r2_test": metrics["r2_test"], "rmse_test": metrics["rmse_test"],
    "mae_test": metrics["mae_test"],
    # No metal is held out in the random test: -1 = not applicable.
    "metal_in_train": -1, "metal_in_cal": -1,
    "experiment": "reproduction",
})

# --- 5b. Family-out shift tests (corrected — the Te leak is fixed). ---
for metal in SHIFT_METALS:
    te_mask = (B_all == metal)
    rest_idx = np.where(~te_mask)[0]
    rng_m = np.random.RandomState(0)
    rest_perm = rng_m.permutation(rest_idx)
    n_tr_m = int(0.6 * len(rest_perm))
    n_cal_m = int(0.2 * len(rest_perm))
    tr_m = rest_perm[:n_tr_m]
    cal_m = rest_perm[n_tr_m:n_tr_m + n_cal_m]
    unused_m = len(rest_perm) - n_tr_m - n_cal_m
    te_m = np.where(te_mask)[0]
    metal_in_train = int((B_all[tr_m] == metal).sum())
    metal_in_cal = int((B_all[cal_m] == metal).sum())
    print(f"Shift {metal}: test rows={len(te_m)} (expected 364), "
          f"train={len(tr_m)}, cal={len(cal_m)}, "
          f"unused non-{metal} rows={unused_m}, "
          f"{metal}-in-train={metal_in_train}, {metal}-in-cal={metal_in_cal}")
    if metal_in_train != 0 or metal_in_cal != 0:
        raise AssertionError(
            f"LEAK: {metal} appears in train ({metal_in_train}) or "
            f"calibration ({metal_in_cal}). STOP — the old v2/v4 Te leak "
            "has returned. Do not test on a metal that was trained on."
        )
    q_m, k_m, n_cal_m_out, metrics_m = run_conformal(
        X_all[tr_m], y_all[tr_m], X_all[cal_m], y_all[cal_m],
        X_all[te_m], y_all[te_m], f"family-out {metal}")
    rows.append({
        "test": f"family-out {metal}", "model": "XGBoost",
        "n_train": int(len(tr_m)), "n_cal": int(n_cal_m_out),
        "n_test": int(len(te_m)), "q": q_m,
        "coverage": metrics_m["coverage"],
        "mean_width": metrics_m["mean_width"],
        "r2_test": metrics_m["r2_test"],
        "rmse_test": metrics_m["rmse_test"],
        "mae_test": metrics_m["mae_test"],
        "metal_in_train": metal_in_train,
        "metal_in_cal": metal_in_cal,
        # v1's Te number was LEAKED (trained on ~218 Te rows), so
        # the family-out rows are CORRECTED, not reproductions:
        # do not expect an exact match with v1 Te.
        "experiment": "corrected",
    })

results = pd.DataFrame(rows, columns=CSV_COLS)
print("Conformal results (4 rows in the full run, 2 in SMOKE):")
print(results.to_string(index=False))

# --- 6. Framing — print it, and save it in the report. ---
# Random coverage is an empirical check of a 90% promise under
# exchangeability: if the test rows are like the training rows,
# about 90% of true values should fall inside the intervals.
# Family-out coverage is an empirical OBSERVATION under domain
# shift, where the guarantee does not apply: the held-out metal
# is systematically different, so coverage can be anything.
# Coverage is NOT accuracy: a wide interval can cover while the
# point predictions are worse than the mean (v1 Te: coverage
# 0.948, but R2 on test -0.086). Always report coverage AND
# width AND point R2 together.
print("FRAMING: random coverage checks a 90% promise (exchangeable "
      "test rows). Family-out coverage is only an observation under "
      "shift — no guarantee applies. Coverage is NOT accuracy: wide "
      "intervals can cover while point R2 is negative.")


def gate6_check(results_df):
    """GATE 6 — random row is gated; family-out rows are reported,
    NOT gated (observations under shift, not promises).
    Returns (gate_lines, gate_failures)."""
    lines, failures = [], []

    def _check(ok, text):
        status = "PASS" if ok else "FAIL"
        lines.append(f"  [{status}] {text}")
        if not ok:
            failures.append(text)
        return ok

    rand = results_df[results_df["test"] == "random"]
    if len(rand) != 1:
        _check(False, "random row: exactly 1 row expected")
    else:
        r = rand.iloc[0]
        _check(0.88 <= r["coverage"] <= 0.92,
               f"random coverage {r['coverage']:.3f} in [0.88, 0.92] "
               f"(v1: 0.900)")
        _check(0.25 <= r["mean_width"] <= 0.50,
               f"random mean width {r['mean_width']:.3f} in [0.25, 0.50] "
               f"(v1: 0.368)")
    shift = results_df[results_df["test"].str.startswith("family-out")]
    for _, s in shift.iterrows():
        _check(int(s["metal_in_train"]) == 0 and int(s["metal_in_cal"]) == 0,
               f"{s['test']}: metal-in-train={int(s['metal_in_train'])}, "
               f"metal-in-cal={int(s['metal_in_cal'])} (both must be 0)")
    _check(list(results_df.columns) == CSV_COLS,
           "columns unified (one fixed column set)")
    return lines, failures


# --- 7. GATE 6 — random row is gated, family-out is reported. ---
gate_lines = []
gate_failures = []
if SMOKE:
    print("GATE 6: SKIPPED in SMOKE mode — smoke runs the random test "
          "on 4,000 rows and Te only, so coverage cannot be compared "
          "with v1. SMOKE-INFO checks instead:")
    smoke_lines, _ = gate6_check(results)
    for line in smoke_lines:
        print("  SMOKE-INFO " + line)
    gate_lines.append("GATE 6: SKIPPED (SMOKE mode — no v1 comparison).")
else:
    print("GATE 6 — random coverage in [0.88, 0.92], leak asserts 0, "
          "4 rows, columns unified (family-out coverages NOT gated):")
    gate_lines, gate_failures = gate6_check(results)
    for line in gate_lines:
        print(line)

# --- 8. Save: ONE DataFrame, ONE write, unified columns. ---
csv_name = ("results_conformal_v2_smoke.csv" if SMOKE
            else "results_conformal_v2.csv")
csv_path = OUT_DIR / csv_name
if csv_path.exists():
    shutil.copy2(csv_path, OUT_DIR / (csv_name + ".backup.csv"))
    print(f"backup saved: {csv_name}.backup.csv")
results.to_csv(csv_path, index=False)
print(f"saved {csv_name}: {len(results)} rows")

# --- 9. Save the Stage 6 report. ---
report_name = "conformal_report_v2_smoke.md" if SMOKE else "conformal_report_v2.md"
report_lines = [
    "# Perovskite v2 — Stage 6 conformal uncertainty report",
    "",
    "Exact split conformal, 90% nominal, tuned XGBoost, Magpie only.",
    "Quantile is the exact order statistic: sort calibration residuals,",
    "k = ceil((n_cal + 1) * 0.90), q = r[k-1]. No np.quantile.",
    "",
    "## Results",
    "",
    results.to_string(index=False),
    "",
    "## Framing",
    "",
    "Random coverage is an empirical check of a 90% promise under",
    "exchangeability. Family-out coverage is an empirical observation",
    "under domain shift, where the guarantee does not apply. Coverage",
    "is NOT accuracy: a wide interval can cover while point",
    "predictions are worse than the mean (v1 Te: coverage 0.948,",
    "R2 on test -0.086). Always report coverage AND width AND",
    "point R2 together.",
    "",
    "## v1 comparison",
    "",
    "- Random (reproduction): v1 coverage 0.900, mean width 0.368,",
    "  R2 on test 0.438, N_test 3,787.",
    "- Te (corrected, NOT a reproduction): v1 coverage 0.948, mean",
    "  width 0.381, R2 on test -0.086 — but v1 Te was LEAKED",
    "  (trained on ~218 Te rows, tested on all 364 Te). v2 trains",
    "  and calibrates on B != Te only, so do not expect an exact",
    "  match. W (hard failure) and Mg (good case) are new in v2.",
    "",
    "## GATE 6",
    "",
]
report_lines += gate_lines
report_lines += [
    "",
    "## Notes",
    "",
    "- experiment column: 'reproduction' for the random row,",
    "  'corrected' for the family-out rows.",
    "- metal_in_train / metal_in_cal are -1 for the random row",
    "  (not applicable — no metal is held out); 0 for every",
    "  family-out row (asserted in code).",
    "- If GATE 6 failed on random coverage, the quantile code is",
    "  wrong — stop. Family-out coverages are reported, not gated.",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")

# --- GATE 6 verdict. ---
if SMOKE:
    print("STAGE 6 SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 6: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 6 FAILED: " + "; ".join(gate_failures))
else:
    print("GATE 6: PASS — random coverage is ~0.90, leak asserts are 0. "
          "Next: Stage 7 figures.")
    print("STAGE 6 done — paste back the GATE 6 line and the 4-row table.")
