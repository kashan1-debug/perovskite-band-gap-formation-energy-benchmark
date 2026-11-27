# ============================================================
# v2 — STAGE 9a: SHAP feature importance (site-aware vs Magpie-only)
# (ONE cell, paste into Colab)
# Run AFTER Stages 3b and 3c (GATE 3b and GATE 3c PASS). It
# reloads features_siteaware.parquet + audited.parquet from
# Drive — it does NOT redo Stages 1-7, 3b or 3c.
# Run in the same session, or after Setup B then Setup C in a
# fresh session.
# Full run takes about 2-5 minutes in Colab: two XGBoost fits
# on ~15,000 rows, then SHAP on 2,000 test rows. Every output
# is written to Drive at once, so a disconnect loses nothing.
# If the run stops, just paste this cell again: it overwrites
# its own outputs (backup first), it never duplicates rows.
# WHAT QUESTION THIS ANSWERS: Stages 3b/3c showed site-aware
# features recover leave-one-B-out (pooled R2 0.733 vs Magpie
# 0.086), and the no-B-info ablation (0.683) showed most of the
# gain comes from giving A and X explicitly — site resolution —
# not from B-site physics (+0.050 only). But that was proved by
# REMOVING feature groups. This stage asks the same question
# from the other side: when the full site-aware model makes a
# prediction, WHICH features does it actually lean on? SHAP
# (SHapley Additive exPlanations) gives every feature a fair
# share of the credit for each prediction: mean |SHAP| over the
# test rows is how much that feature moves predictions, on
# average, in eV/atom. If B-physics were the key, B-physics
# features would top the list. If site resolution is the key,
# A-physics, X-physics, X one-hot and the derived columns
# should carry the weight — and the Magpie share should fall
# compared with the Magpie-only model.
# HONEST LIMIT, read this before quoting the numbers: SHAP
# here is computed on a RANDOM 80/20 split (interpolation),
# not on leave-one-B-out. It shows which features the model
# uses when it CAN interpolate. That is supporting evidence
# for the mechanism, not a direct test of transfer — the
# direct test stays Stages 3b/3c. Do not write "SHAP proves
# transfer"; write "SHAP agrees with the ablation".
# Models: tuned XGBoost ONLY, seed 0, same config as Stages
# 3/4/5/6/3b/3c. No defaults anywhere. Two models: (a)
# site-aware X (232 cols), (b) Magpie-only control (132 cols).
# SHAP: shap.TreeExplainer on each trained model, computed on
# a subsample of the test set (2,000 rows max, random_state=0)
# to keep runtime sane. The SHAP identity — base value plus
# the SHAP values of a row equals the model prediction — is
# checked on 5 test rows and asserted in code.
# Paste back: the GATE 9a line, the two test R2 values, and
# the top-3 features + top group from the report.
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True uses only the first 4,000 rows and 200 test rows
# for SHAP, and writes *_smoke.* files instead, so test outputs
# can never mix into real results. Use it only to check the
# code runs.
SMOKE = False

import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

# --- 1. Reload features_siteaware.parquet + audited.parquet. ---
# features_siteaware.parquet is audited (18) + Magpie (132) +
# one-hot B (52) + one-hot X (7) + physics (36) + derived (5) = 250.
sa_path = OUT_DIR / "features_siteaware.parquet"
audit_path = OUT_DIR / "audited.parquet"
for p, name in [(sa_path, "features_siteaware.parquet"), (audit_path, "audited.parquet")]:
    if not p.exists():
        raise FileNotFoundError(
            f"{name} not found in OUT_DIR. Run Stages 1-2 and 3b first, "
            "then this cell again."
        )
sa = pd.read_parquet(sa_path)
audited = pd.read_parquet(audit_path)
print("features_siteaware.parquet shape:", sa.shape, "(expected (18928, 250))")
print("audited.parquet shape:", audited.shape, "(expected (18928, 18))")
if not SMOKE:
    if sa.shape != (18928, 250) or audited.shape != (18928, 18):
        raise AssertionError(
            "STOP: shapes are not (18928, 250) and (18928, 18). "
            "Re-run Stages 1-2 and 3b before this stage."
        )
else:
    if sa.shape[1] != 250 or audited.shape[1] != 18 or sa.shape[0] < 4000:
        raise AssertionError(
            "STOP (SMOKE): need at least 4,000 rows x 250 cols in "
            "features_siteaware.parquet and 18 cols in audited.parquet."
        )
    sa = sa.iloc[:4000].reset_index(drop=True)
    audited = audited.iloc[:4000].reset_index(drop=True)
    print("SMOKE: using first 4,000 rows only.")
# Row-by-row alignment between the two files — the site-aware
# features must be attached to the right rows.
align = bool(
    len(sa) == len(audited)
    and (sa["formula_ordered"].values == audited["formula_ordered"].values).all()
    and np.allclose(sa["e_form"].values, audited["e_form"].values)
)
print("alignment features_siteaware vs audited (formula_ordered, e_form):", align)
if not align:
    raise AssertionError(
        "ALIGNMENT FAILED: features_siteaware.parquet does not match "
        "audited.parquet row by row. STOP — do not run SHAP on "
        "misaligned rows."
    )
for needed in ["A_site", "B_site", "X_anion", "e_form_per_atom"]:
    if needed not in sa.columns:
        raise AssertionError(f"STOP: needed column {needed} is missing from features_siteaware.parquet.")

# --- 2. Feature groups — the whole point of this stage. ---
# Group definitions (same layout as Stage 3b built it):
#   magpie     MagpieData*                132 cols
#   A_physics  A_Z, A_mass, ...           12 cols
#   B_physics  B_Z, B_mass, ...           12 cols
#   X_physics  X_Z, X_mass, ...           12 cols
#   derived    tolerance_factor, octahedral_factor, rA_minus_rB,
#              EN_A_minus_EN_B, EN_B_minus_EN_X   5 cols
#   B_onehot   B_<element symbol>         52 cols
#   X_onehot   X_<anion label>             7 cols
# Total X = 232. A one-hot does not exist (Stage 3b never built one).
PHYS_KEYS = ["Z", "mass", "EN", "r_ion", "r_atom", "ionE", "EA",
             "val_s", "val_p", "val_d", "val_f", "mendeleev"]
DERIVED = ["tolerance_factor", "octahedral_factor", "rA_minus_rB",
           "EN_A_minus_EN_B", "EN_B_minus_EN_X"]
AUDITED_COLS = list(audited.columns)


def group_of(col):
    """Which group one feature column belongs to. Physics keys are
    checked BEFORE the one-hot fallback, so B_EN is B-physics
    while B_Mg is a B one-hot."""
    if col.startswith("MagpieData"):
        return "magpie"
    if col in DERIVED:
        return "derived"
    for prefix, gname in [("A_", "A_physics"), ("B_", "B_physics"), ("X_", "X_physics")]:
        if col.startswith(prefix) and col[len(prefix):] in PHYS_KEYS:
            return gname
    if col.startswith("B_"):
        return "B_onehot"
    if col.startswith("X_"):
        return "X_onehot"
    return "other"


X_COLS = [c for c in sa.columns if c not in AUDITED_COLS]
groups = pd.Series({c: group_of(c) for c in X_COLS})
group_counts = groups.value_counts().to_dict()
print("Feature columns in X:", len(X_COLS), "(expected 232)")
print("Group counts:", group_counts)
EXPECTED_COUNTS = {"magpie": 132, "A_physics": 12, "B_physics": 12,
                   "X_physics": 12, "derived": 5, "B_onehot": 52, "X_onehot": 7}
for gname, expected in EXPECTED_COUNTS.items():
    got = group_counts.get(gname, 0)
    if got != expected:
        raise AssertionError(
            f"STOP: group {gname} has {got} columns, expected {expected}. "
            "The site-aware frame is not the Stage 3b layout."
        )
if "other" in group_counts:
    raise AssertionError(
        f"STOP: {group_counts['other']} feature columns fall in no group: "
        f"{[c for c in X_COLS if group_of(c) == 'other'][:10]}")
MAGPIE_COLS = [c for c in X_COLS if group_of(c) == "magpie"]
print("Magpie-only X:", len(MAGPIE_COLS), "cols (expected 132)")
y = sa["e_form_per_atom"].values
print("y = e_form_per_atom, n =", len(y))
# Leakage assert, before any model: no DFT column in X, ever.
FORBIDDEN = ["vbm", "cbm", "fermi level", "fermi width", "mu_b",
             "e_form", "band_gap", "e_form_per_atom"]
leak = [c for c in X_COLS if c in FORBIDDEN]
print("Leakage columns in X (should be []):", leak)
if leak:
    raise AssertionError(f"STOP: forbidden columns in X: {leak}")

# --- 3. Models — TUNED XGBoost only, never defaults. Seed 0. ---
# Same config as Stages 3/4/5/6/3b/3c. Random 80/20 split, seed 0.
# This is a RANDOM split (interpolation), not leave-one-B-out —
# see the header: SHAP here is supporting evidence for the
# mechanism, the direct transfer test stays Stages 3b/3c.
XGB_CONFIG = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                  tree_method="hist", n_jobs=-1, random_state=0)
print("Models (tuned only, seed 0):")
print("  XGBoost: n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5")
print("Split: random 80/20, seed 0 (interpolation split — see header).")
X_sa = sa[X_COLS].values
X_mag = sa[MAGPIE_COLS].values
idx = np.arange(len(y))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=0)
print(f"Train n = {len(idx_train)}, test n = {len(idx_test)}")

t0 = time.time()
model_sa = XGBRegressor(**XGB_CONFIG)
model_sa.fit(X_sa[idx_train], y[idx_train])
pred_sa = model_sa.predict(X_sa[idx_test])
r2_sa = float(r2_score(y[idx_test], pred_sa))
rmse_sa = float(np.sqrt(mean_squared_error(y[idx_test], pred_sa)))
mae_sa = float(mean_absolute_error(y[idx_test], pred_sa))
print(f"site-aware model: test R2 {r2_sa:.3f}, RMSE {rmse_sa:.3f}, "
      f"MAE {mae_sa:.3f} (fit in {time.time() - t0:.0f} s)")

t0 = time.time()
model_mag = XGBRegressor(**XGB_CONFIG)
model_mag.fit(X_mag[idx_train], y[idx_train])
pred_mag = model_mag.predict(X_mag[idx_test])
r2_mag = float(r2_score(y[idx_test], pred_mag))
rmse_mag = float(np.sqrt(mean_squared_error(y[idx_test], pred_mag)))
mae_mag = float(mean_absolute_error(y[idx_test], pred_mag))
print(f"Magpie-only model: test R2 {r2_mag:.3f}, RMSE {rmse_mag:.3f}, "
      f"MAE {mae_mag:.3f} (fit in {time.time() - t0:.0f} s)")
print("Sanity only: these R2 values are on a random split, so they "
      "are interpolation numbers, not family-out numbers. They are "
      "printed, not gated.")

# --- 4. SHAP — TreeExplainer on a subsample of the test set. ---
# 2,000 test rows max (200 in SMOKE), random_state=0, so the run
# stays fast in Colab and the subsample is reproducible.
N_SHAP = 200 if SMOKE else 2000
rng = np.random.default_rng(0)
n_shap = min(N_SHAP, len(idx_test))
shap_idx = np.sort(rng.choice(idx_test, size=n_shap, replace=False))
print(f"SHAP subsample: {n_shap} test rows (max {N_SHAP}, random_state=0)")

IDENTITY_TOL = 1e-6
identity_results = {}


def run_shap(model, X, cols, label):
    """Mean |SHAP| per feature for one trained model, plus the
    identity check: base_value + sum(SHAP) == model prediction."""
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    X_sub = X[shap_idx]
    sv = explainer.shap_values(X_sub)
    if hasattr(sv, "values"):  # Explanation object in newer shap
        sv = sv.values
    sv = np.asarray(sv, dtype=float)
    if sv.shape != (len(shap_idx), len(cols)):
        raise AssertionError(
            f"STOP: SHAP values for {label} have shape {sv.shape}, "
            f"expected ({len(shap_idx)}, {len(cols)}).")
    base = explainer.expected_value
    base = float(np.asarray(base).ravel()[0])
    # Identity check on the first 5 subsample rows.
    preds = model.predict(X_sub[:5])
    recon = base + sv[:5].sum(axis=1)
    max_err = float(np.max(np.abs(recon - preds)))
    identity_results[label] = (base, max_err)
    print(f"SHAP {label}: base_value {base:.6f}, identity max error "
          f"over 5 rows {max_err:.2e} (tolerance {IDENTITY_TOL:.0e}), "
          f"computed in {time.time() - t0:.0f} s")
    if max_err >= IDENTITY_TOL:
        raise AssertionError(
            f"STOP: SHAP identity check FAILED for {label}: max error "
            f"{max_err:.2e} >= {IDENTITY_TOL:.0e}. The SHAP values do "
            "not add up to the model predictions — do not trust them.")
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.DataFrame({
        "model": label,
        "feature": cols,
        "group": [group_of(c) for c in cols],
        "mean_abs_shap": mean_abs,
    })


shap_sa = run_shap(model_sa, X_sa, X_COLS, "siteaware")
shap_mag = run_shap(model_mag, X_mag, MAGPIE_COLS, "magpie")
results = pd.concat([shap_sa, shap_mag], ignore_index=True)
# Rank within each model: 1 = largest mean |SHAP|.
results["rank"] = (results.groupby("model")["mean_abs_shap"]
                   .rank(method="first", ascending=False).astype(int))
RESULT_COLS = ["model", "feature", "group", "mean_abs_shap", "rank"]
results = results[RESULT_COLS].sort_values(["model", "rank"]).reset_index(drop=True)
print(f"SHAP table: {len(results)} rows (expected 232 + 132 = 364)")

# --- 5. Per-group totals — the mechanism answer. ---
# For each model: total = sum of mean |SHAP| over the group's
# features (how much credit the whole group takes), mean =
# total / n_features (credit per feature, so big groups are not
# flattered just for being big), share = total / model total.
group_rows = []
for (model, group), sub in results.groupby(["model", "group"]):
    total = float(sub["mean_abs_shap"].sum())
    group_rows.append({
        "model": model,
        "group": group,
        "n_features": int(len(sub)),
        "total_mean_abs_shap": total,
        "mean_mean_abs_shap": float(sub["mean_abs_shap"].mean()),
        "share": None,  # filled below, once the model total is known
    })
groups_df = pd.DataFrame(group_rows)
for model in ["siteaware", "magpie"]:
    tot = groups_df.loc[groups_df["model"] == model, "total_mean_abs_shap"].sum()
    groups_df.loc[groups_df["model"] == model, "share"] = (
        groups_df.loc[groups_df["model"] == model, "total_mean_abs_shap"] / tot)
GROUP_COLS = ["model", "group", "n_features", "total_mean_abs_shap",
              "mean_mean_abs_shap", "share"]
groups_df = groups_df[GROUP_COLS].sort_values(
    ["model", "total_mean_abs_shap"], ascending=[True, False]).reset_index(drop=True)
print("Per-group SHAP totals (site-aware model):")
print(groups_df[groups_df["model"] == "siteaware"].to_string(index=False))

# The plain-language mechanism sentence. Which group carries the
# site-aware gain? Compare the site-aware model's group totals:
# "site-resolution" groups are A_physics + X_physics + X_onehot +
# derived; B groups are B_physics + B_onehot; Magpie is the control.
sa_groups = groups_df[groups_df["model"] == "siteaware"].set_index("group")
site_res_share = float(sa_groups.loc[["A_physics", "X_physics", "X_onehot", "derived"],
                                     "share"].sum())
b_share = float(sa_groups.loc[["B_physics", "B_onehot"], "share"].sum())
mag_share_sa = float(sa_groups.loc["magpie", "share"])
top_group_sa = str(sa_groups["total_mean_abs_shap"].idxmax())
top_feature_sa = str(results[(results["model"] == "siteaware")
                             & (results["rank"] == 1)]["feature"].iloc[0])
if site_res_share > b_share and site_res_share > mag_share_sa:
    mechanism = (
        f"Site resolution carries the gain: A/X physics, X one-hot and "
        f"derived columns together take {site_res_share:.0%} of the "
        f"site-aware model's SHAP credit, vs {b_share:.0%} for all "
        f"B-site columns and {mag_share_sa:.0%} for Magpie — this AGREES "
        f"with the Stage 3c ablation (no-B-info 0.683 vs site-aware 0.733)."
    )
elif b_share > site_res_share and b_share > mag_share_sa:
    mechanism = (
        f"B-site columns carry the gain: B physics + B one-hot take "
        f"{b_share:.0%} of the site-aware model's SHAP credit, vs "
        f"{site_res_share:.0%} for site-resolution columns and "
        f"{mag_share_sa:.0%} for Magpie — this DISAGREES with the "
        f"Stage 3c ablation (no-B-info 0.683 vs site-aware 0.733), "
        f"so the two tests must be reconciled before the paper "
        f"claims a mechanism."
    )
else:
    mechanism = (
        f"Magpie still carries the gain: Magpie takes "
        f"{mag_share_sa:.0%} of the site-aware model's SHAP credit, vs "
        f"{site_res_share:.0%} for site-resolution columns and "
        f"{b_share:.0%} for B-site columns — the site-aware features "
        f"add little on this random split, and the Stage 3b/3c "
        f"family-out story is not explained by SHAP here."
    )
print("MECHANISM:", mechanism)

# --- 6. Figure — top 20 site-aware features + group totals. ---
fig_name = "shap_importance_smoke.png" if SMOKE else "shap_importance.png"
fig_path = OUT_DIR / "figures" / fig_name
fig_path.parent.mkdir(parents=True, exist_ok=True)
GROUP_COLORS = {"magpie": "#4C78A8", "A_physics": "#54A24B",
                "B_physics": "#E45756", "X_physics": "#F58518",
                "derived": "#72B7B2", "B_onehot": "#EECA3B",
                "X_onehot": "#B279A2"}
top20 = results[results["model"] == "siteaware"].nsmallest(20, "rank").copy()
top20 = top20.sort_values("mean_abs_shap")  # largest on top in a horiz bar
sa_grp = groups_df[groups_df["model"] == "siteaware"].sort_values("total_mean_abs_shap")
fig, axes = plt.subplots(1, 2, figsize=(13, 7))
ax = axes[0]
ax.barh(top20["feature"],
        top20["mean_abs_shap"],
        color=[GROUP_COLORS.get(g, "#888888") for g in top20["group"]])
ax.set_xlabel("Mean |SHAP| (eV/atom)")
ax.set_title("Top 20 features — site-aware model\n(random 80/20 split, "
             f"{n_shap} test rows)")
for label in ax.get_yticklabels():
    label.set_fontsize(8)
ax = axes[1]
ax.barh(sa_grp["group"], sa_grp["total_mean_abs_shap"],
        color=[GROUP_COLORS.get(g, "#888888") for g in sa_grp["group"]])
ax.set_xlabel("Total mean |SHAP| (eV/atom, summed over group)")
ax.set_title("SHAP credit by feature group — site-aware model")
for label in ax.get_yticklabels():
    label.set_fontsize(9)
fig.tight_layout()
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"saved figures/{fig_name} at 300 dpi")

# --- 7. Save outputs (backup first, safe to run twice). ---
suffix = "_smoke" if SMOKE else ""
csv_name = f"results_shap_v2{suffix}.csv"
groups_name = f"results_shap_groups_v2{suffix}.csv"
report_name = f"shap_report_v2{suffix}.md"
for name, df in [(csv_name, results), (groups_name, groups_df)]:
    p = OUT_DIR / name
    if p.exists():
        shutil.copy2(p, OUT_DIR / (name + ".backup.csv"))
        print(f"backup saved: {name}.backup.csv")
    df.to_csv(p, index=False)
    print(f"saved {name}: {len(df)} rows")
top10_sa = results[(results["model"] == "siteaware")].nsmallest(10, "rank")
top10_mag = results[(results["model"] == "magpie")].nsmallest(10, "rank")
report_lines = [
    "# Perovskite v2 — Stage 9a SHAP feature-importance report",
    "",
    "SHAP (SHapley Additive exPlanations) gives every feature a fair "
    "share of the credit for each prediction. Mean |SHAP| is how much "
    "a feature moves predictions on average, in eV/atom. Two tuned "
    "XGBoost models (n_estimators=100, max_depth=3, lr=0.05, "
    "subsample=0.8, colsample_bytree=0.8, reg_lambda=5, seed 0), "
    "random 80/20 split seed 0: site-aware X (232 cols) and "
    "Magpie-only control (132 cols). SHAP computed with "
    "shap.TreeExplainer on a subsample of the test set "
    f"({n_shap} rows, random_state=0).",
    "",
    "## Honest limit",
    "",
    "SHAP here is computed on a RANDOM split (interpolation), not on "
    "leave-one-B-out. It shows which features the model uses when it "
    "can interpolate. It is supporting evidence for the mechanism, "
    "not a direct test of transfer — the direct test stays Stages "
    "3b/3c (site-aware pooled B-out R2 0.733, no-B-info 0.683, "
    "Magpie-only 0.086). Do not write 'SHAP proves transfer'.",
    "",
    "## Test scores (random split — sanity only, not gated)",
    "",
    f"- site-aware: test R2 {r2_sa:.3f}, RMSE {rmse_sa:.3f}, MAE {mae_sa:.3f}",
    f"- Magpie-only: test R2 {r2_mag:.3f}, RMSE {rmse_mag:.3f}, MAE {mae_mag:.3f}",
    "",
    "## SHAP identity check (base_value + sum(SHAP) == prediction)",
    "",
]
for label, (base, max_err) in identity_results.items():
    report_lines.append(
        f"- {label}: base_value {base:.6f}, max error over 5 rows "
        f"{max_err:.2e} (tolerance {IDENTITY_TOL:.0e}) — PASS")
report_lines += [
    "",
    "## Top 10 features — site-aware model",
    "",
    top10_sa[["rank", "feature", "group", "mean_abs_shap"]].to_string(index=False),
    "",
    "## Top 10 features — Magpie-only model",
    "",
    top10_mag[["rank", "feature", "group", "mean_abs_shap"]].to_string(index=False),
    "",
    "## Per-group SHAP totals",
    "",
    groups_df.to_string(index=False),
    "",
    "## Mechanism — which group carries the site-aware gain?",
    "",
    mechanism,
    "",
    f"Top group (site-aware model): {top_group_sa}. Top feature: {top_feature_sa}.",
    "",
    "## Notes",
    "",
    "- experiment: random 80/20 split, seed 0, tuned XGBoost only.",
    "- 'share' is the group's total mean |SHAP| divided by the model's "
    "total mean |SHAP|; 'mean' divides the total by the number of "
    "features in the group, so large groups are not flattered for size.",
    "- Figure: figures/shap_importance.png (top 20 features + group totals).",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")

# --- 8. GATE 9a. ---
# Files exist with the expected columns, identity check passed,
# both models trained, test R2 printed. Test R2 values are sanity
# only — they are NOT pass/fail, because this is a random-split
# diagnostic and the family-out numbers already live in Stages 3b/3c.
gate_lines = []
gate_failures = []
for name, expected_cols, expected_rows in [
        (csv_name, RESULT_COLS, 364),
        (groups_name, GROUP_COLS, None)]:
    p = OUT_DIR / name
    if not p.exists():
        gate_failures.append(f"{name} missing")
        gate_lines.append(f"  [FAIL] {name} missing")
        continue
    got = pd.read_csv(p)
    if list(got.columns) != expected_cols:
        gate_failures.append(f"{name} has wrong columns: {list(got.columns)}")
        gate_lines.append(f"  [FAIL] {name} has wrong columns: {list(got.columns)}")
    elif expected_rows is not None and len(got) != expected_rows:
        gate_failures.append(f"{name} has {len(got)} rows, expected {expected_rows}")
        gate_lines.append(f"  [FAIL] {name} has {len(got)} rows, expected {expected_rows}")
    else:
        gate_lines.append(f"  [PASS] {name}: {len(got)} rows, columns as expected")
if not fig_path.exists():
    gate_failures.append(f"figures/{fig_name} missing")
    gate_lines.append(f"  [FAIL] figures/{fig_name} missing")
else:
    gate_lines.append(f"  [PASS] figures/{fig_name} exists at 300 dpi")
for label, (base, max_err) in identity_results.items():
    if max_err < IDENTITY_TOL:
        gate_lines.append(f"  [PASS] SHAP identity {label}: max error {max_err:.2e}")
    else:
        gate_failures.append(f"SHAP identity {label} failed: {max_err:.2e}")
        gate_lines.append(f"  [FAIL] SHAP identity {label}: max error {max_err:.2e}")
for label, r2v in [("siteaware", r2_sa), ("magpie", r2_mag)]:
    gate_lines.append(f"  [INFO] {label} test R2 (random split, sanity only): {r2v:.3f}")
print("GATE 9a — files, columns, SHAP identity, both models trained:")
for line in gate_lines:
    print(line)

# --- GATE 9a verdict. ---
if SMOKE:
    print("STAGE 9a SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 9a: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 9a FAILED: " + "; ".join(gate_failures))
else:
    print("GATE 9a: PASS — SHAP table, group totals, figure and report "
          "saved; identity check passed for both models. Next: read the "
          "mechanism sentence in shap_report_v2.md, then the Stage 9b "
          "graph-model comparison.")
    print("STAGE 9a done — paste back the GATE 9a line, the two test "
          "R2 values, and the top-3 features + top group.")
