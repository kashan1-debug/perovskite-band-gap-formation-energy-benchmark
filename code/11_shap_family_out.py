# ============================================================
# v2 — STAGE 9a2: SHAP on held-out B metals (family-out SHAP)
# (ONE cell, paste into Colab)
# Run AFTER Stage 9a (GATE 9a PASS). It reloads
# features_siteaware.parquet + audited.parquet from Drive — it
# does NOT redo Stages 1-7, 3b, 3c or 9a.
# Run in the same session, or after Setup B then Setup C in a
# fresh session.
# Full run takes about 2-4 minutes in Colab: three tuned XGBoost
# fits on ~18,500 rows each (a fit took ~2 s in Stage 9a) plus
# three SHAP runs on 364 rows each (SHAP on 2,000 rows took
# under 1 s in Stage 9a). Every finished family is saved to
# Drive at once, so a disconnect loses nothing already done. If
# the run stops, just paste this cell again: families already
# in results_shap_familyout_v2.csv are skipped, not redone.
# WHAT QUESTION THIS ANSWERS: Stage 9a SHAP was on a RANDOM
# split (interpolation), where Magpie still took 40% of the
# credit. That does NOT prove the mechanism under shift — on a
# random split the model can interpolate, so Magpie is allowed
# to help. This stage asks the same question on the HARD test:
# train the site-aware model with NEVER having seen a B metal,
# then ask SHAP which feature groups drive its predictions ON
# that held-out metal. If site resolution (A/X physics, X
# one-hot, derived) takes over from Magpie under shift, the
# Stage 3b/3c story is confirmed from the other side. If
# Magpie still dominates even on a metal the model never saw,
# the 3b/3c gain is NOT explained by which features the model
# leans on — and the paper must say so.
# WHY B one-hot and B-physics are NOT leakage here: they
# describe the TEST crystals, whose B identity is known at
# prediction time — exactly as in Stage 3b. The held-out metal
# never appears in the TRAINING rows (asserted in code). Its
# B one-hot column is all zero in training, so the model cannot
# have memorised that metal; on the held-out rows it must rely
# on the physics features, not the identity. That is the point
# of the test, not a leak.
# Families: Te (the v1 conformal metal), W (the hard failure
# case: Stage 4 B-out R2 -5.10 Magpie / site-aware 0.226 in 3b),
# Mg (the good case: Stage 4 B-out R2 0.31-0.43). 364 rows each.
# Models: tuned XGBoost ONLY, seed 0, same config as Stages
# 3/4/5/6/3b/3c/9a. No defaults anywhere.
# SHAP: shap.TreeExplainer on ALL 364 held-out rows of each
# family (100 in SMOKE). The SHAP identity — base value plus
# the SHAP values of a row equals the model prediction — is
# checked on 5 rows per family and asserted in code.
# Held-out R2 / RMSE / MAE are printed per family so the SHAP
# story is tied to an actual transfer score. They are sanity
# only, NOT gated — W will be low, that is expected.
# Paste back: the GATE 9a2 line, the three held-out R2 values,
# and the top group per family + the overall mechanism sentence.
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs only Te (1 family) with SHAP on 100 held-out
# rows, and writes *_smoke.* files instead, so test outputs can
# never mix into real results. Use it only to check the code runs.
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
    print("SMOKE: using the synthetic frame as-is (no row truncation).")
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

# --- 2. Feature groups — same definitions as Stage 9a. ---
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
y = sa["e_form_per_atom"].values
print("y = e_form_per_atom, n =", len(y))
# Leakage assert, before any model: no DFT column in X, ever.
FORBIDDEN = ["vbm", "cbm", "fermi level", "fermi width", "mu_b",
             "e_form", "band_gap", "e_form_per_atom"]
leak = [c for c in X_COLS if c in FORBIDDEN]
print("Leakage columns in X (should be []):", leak)
if leak:
    raise AssertionError(f"STOP: forbidden columns in X: {leak}")

# --- 3. Families: Te, W, Mg — 364 rows each in the full run. ---
FAMILIES = ["Te", "W", "Mg"] if not SMOKE else ["Te"]
X_sa = sa[X_COLS].values
B_site = sa["B_site"].values
for fam in FAMILIES:
    n_fam = int((B_site == fam).sum())
    print(f"Family {fam}: {n_fam} rows (expected 364 in the full run)")
    if not SMOKE and n_fam != 364:
        raise AssertionError(
            f"STOP: family {fam} has {n_fam} rows, expected 364.")
    if SMOKE and n_fam < 100:
        raise AssertionError(
            f"STOP (SMOKE): family {fam} has {n_fam} rows, need >= 100.")
if SMOKE:
    print("SMOKE: family Te only, SHAP on 100 held-out rows.")

# --- 4. Results files: ONE fixed column set per file, resume by skipping. ---
# A family counts as DONE only if it is in BOTH files with the full
# row count (232 feature rows in the per-feature CSV, 7 group rows
# in the groups CSV). Partial rows are dropped and the family is
# re-run, so the two files can never disagree. In SMOKE mode
# separate files are used so test rows can never mix into real results.
suffix = "_smoke" if SMOKE else ""
csv_name = f"results_shap_familyout_v2{suffix}.csv"
groups_name = f"results_shap_familyout_groups_v2{suffix}.csv"
report_name = f"shap_familyout_report_v2{suffix}.md"
fig_name = "shap_familyout_smoke.png" if SMOKE else "shap_familyout.png"
RESULT_COLS = ["family_B", "feature", "group", "mean_abs_shap", "rank"]
GROUP_COLS = ["family_B", "group", "n_features", "total_mean_abs_shap",
              "mean_mean_abs_shap", "share", "r2", "rmse", "mae", "n_test"]
csv_path = OUT_DIR / csv_name
groups_path = OUT_DIR / groups_name


def _load_done(path, expected_cols, rows_per_family):
    """Families fully present in a saved file (exact row count)."""
    if not path.exists():
        return pd.DataFrame(columns=expected_cols), set()
    df = pd.read_csv(path)
    if list(df.columns) != expected_cols:
        backup = OUT_DIR / (path.name + ".broken_backup.csv")
        df.to_csv(backup, index=False)
        print(f"WARNING: old {path.name} had wrong columns — backed up to "
              f"{backup.name}, starting fresh.")
        path.unlink()
        return pd.DataFrame(columns=expected_cols), set()
    counts = df["family_B"].value_counts().to_dict()
    done = {f for f, n in counts.items() if n == rows_per_family}
    partial = {f: n for f, n in counts.items() if n != rows_per_family}
    if partial:
        print(f"WARNING: partial families in {path.name} {partial} — "
              "dropping them so they are re-run cleanly.")
        df = df[~df["family_B"].isin(partial)].reset_index(drop=True)
    return df, done


results_all, done_csv = _load_done(csv_path, RESULT_COLS, 232)
groups_all, done_groups = _load_done(groups_path, GROUP_COLS, 7)
done = done_csv & done_groups
if done_csv != done_groups:
    print(f"WARNING: families in only one file (csv {sorted(done_csv)}, "
          f"groups {sorted(done_groups)}) — only families in BOTH count "
          f"as done: {sorted(done)}.")
    # Keep rows only for families done in both files; the rest re-run.
    results_all = results_all[results_all["family_B"].isin(done)].reset_index(drop=True)
    groups_all = groups_all[groups_all["family_B"].isin(done)].reset_index(drop=True)
if done:
    print(f"Resuming: families already done in both files: {sorted(done)} — skipped.")
else:
    print(f"No completed families in {csv_name} — starting fresh.")
# One backup of the pre-existing files at the start of this run,
# so a re-run can never silently overwrite finished work.
for p in [csv_path, groups_path]:
    if p.exists():
        shutil.copy2(p, OUT_DIR / (p.name + ".backup.csv"))
        print(f"backup saved: {p.name}.backup.csv")


def save_family(family_rows, family_group_rows):
    """Append one family's rows to the in-memory frames and rewrite
    both CSVs at once, so a disconnect cannot lose either."""
    global results_all, groups_all
    # Direct assignment when the frame is still empty avoids a
    # pandas FutureWarning about concatenating empty frames.
    results_all = (family_rows.copy() if results_all.empty else
                   pd.concat([results_all, family_rows], ignore_index=True))
    groups_all = (family_group_rows.copy() if groups_all.empty else
                  pd.concat([groups_all, family_group_rows], ignore_index=True))
    results_all.to_csv(csv_path, index=False)
    groups_all.to_csv(groups_path, index=False)
    print(f"  SAVED {csv_name} + {groups_name}: family {family_rows['family_B'].iloc[0]} "
          f"({len(family_rows)} feature rows, {len(family_group_rows)} group rows)")


# --- 5. Model — TUNED XGBoost only, never defaults. Seed 0. ---
XGB_CONFIG = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                  tree_method="hist", n_jobs=-1, random_state=0)
print("Model (tuned only, seed 0):")
print("  XGBoost: n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5")
N_SHAP = 100 if SMOKE else 364
IDENTITY_TOL = 1e-6
identity_results = {}
family_metrics = {}  # family -> dict(r2, rmse, mae, n_test)


def run_family(family):
    """Train on B != family, test + SHAP on the held-out family."""
    te_mask = (B_site == family)
    tr_mask = ~te_mask
    # The held-out metal must not appear in training — asserted,
    # not assumed. Its B one-hot column is all zero in training.
    assert family not in set(B_site[tr_mask]), (
        f"STOP: held-out metal {family} found in the training rows.")
    assert int(tr_mask.sum()) + int(te_mask.sum()) == len(y)
    print(f"[{family}] train n = {int(tr_mask.sum())} (B != {family}), "
          f"test n = {int(te_mask.sum())} (B == {family})")
    t0 = time.time()
    model = XGBRegressor(**XGB_CONFIG)
    model.fit(X_sa[tr_mask], y[tr_mask])
    pred = model.predict(X_sa[te_mask])
    yt = y[te_mask]
    r2v = float(r2_score(yt, pred))
    rmsev = float(np.sqrt(mean_squared_error(yt, pred)))
    maev = float(mean_absolute_error(yt, pred))
    family_metrics[family] = {"r2": r2v, "rmse": rmsev, "mae": maev,
                              "n_test": int(te_mask.sum())}
    print(f"[{family}] held-out R2 {r2v:.3f}, RMSE {rmsev:.3f}, MAE {maev:.3f} "
          f"(fit in {time.time() - t0:.0f} s) — sanity only, NOT gated.")
    # SHAP on the held-out rows (all 364; 100 in SMOKE).
    te_idx = np.where(te_mask)[0]
    if len(te_idx) > N_SHAP:
        rng = np.random.default_rng(0)
        shap_idx = np.sort(rng.choice(te_idx, size=N_SHAP, replace=False))
    else:
        shap_idx = te_idx
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sa[shap_idx])
    if hasattr(sv, "values"):  # Explanation object in newer shap
        sv = sv.values
    sv = np.asarray(sv, dtype=float)
    if sv.shape != (len(shap_idx), len(X_COLS)):
        raise AssertionError(
            f"STOP: SHAP values for {family} have shape {sv.shape}, "
            f"expected ({len(shap_idx)}, {len(X_COLS)}).")
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    preds5 = model.predict(X_sa[shap_idx[:5]])
    recon = base + sv[:5].sum(axis=1)
    max_err = float(np.max(np.abs(recon - preds5)))
    identity_results[family] = (base, max_err)
    print(f"SHAP {family}: base_value {base:.6f}, identity max error "
          f"over 5 rows {max_err:.2e} (tolerance {IDENTITY_TOL:.0e}), "
          f"computed in {time.time() - t0:.0f} s on {len(shap_idx)} rows")
    if max_err >= IDENTITY_TOL:
        raise AssertionError(
            f"STOP: SHAP identity check FAILED for {family}: max error "
            f"{max_err:.2e} >= {IDENTITY_TOL:.0e}. The SHAP values do "
            "not add up to the model predictions — do not trust them.")
    mean_abs = np.abs(sv).mean(axis=0)
    fam_rows = pd.DataFrame({
        "family_B": family,
        "feature": X_COLS,
        "group": [group_of(c) for c in X_COLS],
        "mean_abs_shap": mean_abs,
    })
    fam_rows["rank"] = (fam_rows["mean_abs_shap"]
                        .rank(method="first", ascending=False).astype(int))
    fam_rows = fam_rows[RESULT_COLS].sort_values("rank").reset_index(drop=True)
    # Per-group totals for this family.
    grp_rows = []
    for group, sub in fam_rows.groupby("group"):
        grp_rows.append({
            "family_B": family,
            "group": group,
            "n_features": int(len(sub)),
            "total_mean_abs_shap": float(sub["mean_abs_shap"].sum()),
            "mean_mean_abs_shap": float(sub["mean_abs_shap"].mean()),
            "share": None,  # filled below
            "r2": r2v, "rmse": rmsev, "mae": maev,
            "n_test": int(te_mask.sum()),
        })
    grp_df = pd.DataFrame(grp_rows)
    tot = grp_df["total_mean_abs_shap"].sum()
    grp_df["share"] = grp_df["total_mean_abs_shap"] / tot
    grp_df = grp_df[GROUP_COLS].sort_values(
        "total_mean_abs_shap", ascending=False).reset_index(drop=True)
    print(f"Per-group SHAP totals ({family}, held-out R2 {r2v:.3f}):")
    print(grp_df[["group", "n_features", "total_mean_abs_shap", "share"]]
          .to_string(index=False))
    save_family(fam_rows, grp_df)
    return fam_rows, grp_df


# --- 6. Run the families not already done. ---
for fam in FAMILIES:
    if fam in done:
        print(f"SKIP {fam} — already in {csv_name} and {groups_name}.")
        # Recover its metrics + identity from the saved files for the
        # report (re-run does not retrain a finished family).
        g = groups_all[groups_all["family_B"] == fam].iloc[0]
        family_metrics[fam] = {"r2": float(g["r2"]), "rmse": float(g["rmse"]),
                               "mae": float(g["mae"]), "n_test": int(g["n_test"])}
        identity_results[fam] = (float("nan"), 0.0)
        continue
    run_family(fam)
print(f"Families finished in this run: "
      f"{[f for f in FAMILIES if f not in done]}; "
      f"total in {csv_name}: {sorted(set(results_all['family_B']))}.")

# Reload from disk so the report/figure use exactly what is saved.
results_all = pd.read_csv(csv_path)
groups_all = pd.read_csv(groups_path)

# --- 7. Mechanism sentences — per family, then overall vs Stage 9a. ---
# "Site resolution" groups are A_physics + X_physics + X_onehot +
# derived; B groups are B_physics + B_onehot; Magpie is the control.
# Stage 9a Colab-confirmed shares (random 80/20 split, site-aware
# model): Magpie 0.398, B_physics 0.319, derived 0.161, A_physics
# 0.116, B_onehot 0.005, X_onehot 0.001, X_physics 0.000 — so
# site-resolution 0.278, B-site 0.324, Magpie 0.398.
SHARE_9A = {"site_res": 0.278, "b_site": 0.324, "magpie": 0.398}


def shares_of(fam):
    sub = groups_all[groups_all["family_B"] == fam].set_index("group")
    site_res = float(sub.loc[["A_physics", "X_physics", "X_onehot", "derived"],
                             "share"].sum())
    b_site = float(sub.loc[["B_physics", "B_onehot"], "share"].sum())
    mag = float(sub.loc["magpie", "share"])
    top_group = str(sub["total_mean_abs_shap"].idxmax())
    return site_res, b_site, mag, top_group


def mechanism_sentence(site_res, b_site, mag, fam, r2v):
    if site_res > b_site and site_res > mag:
        return (
            f"Site resolution carries the transfer prediction on {fam} "
            f"(held-out R2 {r2v:.3f}): A/X physics, X one-hot and derived "
            f"columns together take {site_res:.0%} of the SHAP credit, vs "
            f"{b_site:.0%} for all B-site columns and {mag:.0%} for Magpie "
            f"— this AGREES with the Stage 3b/3c ablation story."
        )
    if b_site > site_res and b_site > mag:
        return (
            f"B-site columns carry the transfer prediction on {fam} "
            f"(held-out R2 {r2v:.3f}): B physics + B one-hot take "
            f"{b_site:.0%} of the SHAP credit, vs {site_res:.0%} for "
            f"site-resolution columns and {mag:.0%} for Magpie — this "
            f"DISAGREES with the Stage 3c ablation (no-B-info 0.683 vs "
            f"site-aware 0.733), so the two tests must be reconciled "
            f"before the paper claims a mechanism."
        )
    return (
        f"Magpie still carries the transfer prediction on {fam} "
        f"(held-out R2 {r2v:.3f}): Magpie takes {mag:.0%} of the SHAP "
        f"credit, vs {site_res:.0%} for site-resolution columns and "
        f"{b_site:.0%} for B-site columns — the site-aware features add "
        f"little even on a metal the model never saw, and the Stage "
        f"3b/3c family-out gain is NOT explained by which features the "
        f"model leans on."
    )


per_family_mechanism = {}
for fam in FAMILIES:
    site_res, b_site, mag, top_group = shares_of(fam)
    per_family_mechanism[fam] = mechanism_sentence(
        site_res, b_site, mag, fam, family_metrics[fam]["r2"])
    print(f"MECHANISM ({fam}): {per_family_mechanism[fam]}")

# Overall: average the three families' shares and compare with 9a.
avg_site_res = float(np.mean([shares_of(f)[0] for f in FAMILIES]))
avg_b_site = float(np.mean([shares_of(f)[1] for f in FAMILIES]))
avg_mag = float(np.mean([shares_of(f)[2] for f in FAMILIES]))
if avg_site_res > SHARE_9A["site_res"] and avg_mag < SHARE_9A["magpie"]:
    overall = (
        f"OVERALL: under shift, site resolution takes over from Magpie — "
        f"averaged over {', '.join(FAMILIES)}, site-resolution columns take "
        f"{avg_site_res:.0%} of the SHAP credit (vs {SHARE_9A['site_res']:.0%} "
        f"on the Stage 9a random split) and Magpie falls to {avg_mag:.0%} "
        f"(vs {SHARE_9A['magpie']:.0%} on 9a). The Stage 3b/3c mechanism "
        f"story is CONFIRMED from the other side: when the model cannot "
        f"interpolate, it leans on A/X site resolution, not on Magpie."
    )
elif avg_b_site > SHARE_9A["b_site"]:
    overall = (
        f"OVERALL: under shift, B-site columns take over — averaged over "
        f"{', '.join(FAMILIES)}, B physics + B one-hot take {avg_b_site:.0%} "
        f"of the SHAP credit (vs {SHARE_9A['b_site']:.0%} on the Stage 9a "
        f"random split), while site resolution takes {avg_site_res:.0%} "
        f"and Magpie {avg_mag:.0%}. This DISAGREES with the Stage 3c "
        f"ablation and must be reconciled before the paper claims a mechanism."
    )
else:
    overall = (
        f"OVERALL: Magpie still dominates even under shift — averaged over "
        f"{', '.join(FAMILIES)}, Magpie takes {avg_mag:.0%} of the SHAP "
        f"credit (vs {SHARE_9A['magpie']:.0%} on the Stage 9a random split) "
        f"and site resolution only {avg_site_res:.0%} (vs "
        f"{SHARE_9A['site_res']:.0%} on 9a). The Stage 3b/3c family-out "
        f"gain is NOT explained by which features the model leans on — "
        f"the paper must say so, and must not claim SHAP proves the mechanism."
    )
print(overall)

# --- 8. Figure — group shares per held-out family, 300 dpi. ---
fig_path = OUT_DIR / "figures" / fig_name
fig_path.parent.mkdir(parents=True, exist_ok=True)
GROUP_COLORS = {"magpie": "#4C78A8", "A_physics": "#54A24B",
                "B_physics": "#E45756", "X_physics": "#F58518",
                "derived": "#72B7B2", "B_onehot": "#EECA3B",
                "X_onehot": "#B279A2"}
GROUP_ORDER = ["magpie", "B_physics", "derived", "A_physics",
               "B_onehot", "X_onehot", "X_physics"]
piv = groups_all.pivot(index="group", columns="family_B", values="share")
piv = piv.reindex([g for g in GROUP_ORDER if g in piv.index])
n_fam = len(FAMILIES)
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(piv))
bar_h = 0.8 / n_fam
for i, fam in enumerate(FAMILIES):
    vals = piv[fam].fillna(0.0).values if fam in piv.columns else np.zeros(len(piv))
    ax.barh(y_pos + (i - (n_fam - 1) / 2) * bar_h, vals,
            height=bar_h, label=f"{fam} (R2 {family_metrics[fam]['r2']:.2f})",
            color="#4C78A8" if i == 0 else ("#E45756" if i == 1 else "#54A24B"))
ax.set_yticks(y_pos)
ax.set_yticklabels(piv.index.tolist())
ax.set_xlabel("SHAP share of total credit (per family)")
ax.set_title("SHAP credit share by feature group — held-out B family\n"
             "(site-aware model, trained without that metal)")
ax.legend(title="Held-out B metal", fontsize=9)
ax.set_xlim(0, 1)
for label in ax.get_yticklabels():
    label.set_fontsize(9)
fig.tight_layout()
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"saved figures/{fig_name} at 300 dpi")

# --- 9. Save the report. ---
report_lines = [
    "# Perovskite v2 — Stage 9a2 SHAP on held-out B metals report",
    "",
    "SHAP (SHapley Additive exPlanations) gives every feature a fair "
    "share of the credit for each prediction. Mean |SHAP| is how much "
    "a feature moves predictions on average, in eV/atom. For each of "
    "Te, W and Mg, one tuned XGBoost model (n_estimators=100, "
    "max_depth=3, lr=0.05, subsample=0.8, colsample_bytree=0.8, "
    "reg_lambda=5, seed 0) was trained ONLY on rows with B != that "
    "metal, and SHAP was computed with shap.TreeExplainer on all 364 "
    "held-out rows of that metal (100 in SMOKE).",
    "",
    "## Why B one-hot and B-physics are not leakage here",
    "",
    "They describe the TEST crystals, whose B identity is known at "
    "prediction time — exactly as in Stage 3b. The held-out metal "
    "never appears in the training rows (asserted in code); its B "
    "one-hot column is all zero in training, so the model cannot "
    "have memorised that metal.",
    "",
    "## Held-out transfer scores (sanity only, NOT gated)",
    "",
]
for fam in FAMILIES:
    m = family_metrics[fam]
    report_lines.append(
        f"- {fam}: held-out R2 {m['r2']:.3f}, RMSE {m['rmse']:.3f}, "
        f"MAE {m['mae']:.3f}, n_test {m['n_test']}")
report_lines += [
    "",
    "## SHAP identity check (base_value + sum(SHAP) == prediction)",
    "",
]
for fam in FAMILIES:
    base, max_err = identity_results[fam]
    if np.isnan(base):
        report_lines.append(
            f"- {fam}: identity NOT re-checked in this run (family was "
            f"resumed from saved files; it passed when first computed).")
    else:
        report_lines.append(
            f"- {fam}: base_value {base:.6f}, max error over 5 rows "
            f"{max_err:.2e} (tolerance {IDENTITY_TOL:.0e}) — PASS")
report_lines += [
    "",
    "## Per-group SHAP totals (share = group total / family total)",
    "",
    groups_all.to_string(index=False),
    "",
]
for fam in FAMILIES:
    top10 = results_all[results_all["family_B"] == fam].nsmallest(10, "rank")
    report_lines += [
        f"## Top 10 features — held-out {fam}",
        "",
        top10[["rank", "feature", "group", "mean_abs_shap"]].to_string(index=False),
        "",
    ]
report_lines += [
    "## Mechanism — which group carries the transfer prediction?",
    "",
]
for fam in FAMILIES:
    report_lines.append(per_family_mechanism[fam])
    report_lines.append("")
report_lines += [
    overall,
    "",
    "## Comparison with Stage 9a (random split, site-aware model)",
    "",
    f"- Stage 9a shares: site-resolution {SHARE_9A['site_res']:.1%}, "
    f"B-site {SHARE_9A['b_site']:.1%}, Magpie {SHARE_9A['magpie']:.1%}.",
    f"- This stage, averaged over {', '.join(FAMILIES)}: "
    f"site-resolution {avg_site_res:.1%}, B-site {avg_b_site:.1%}, "
    f"Magpie {avg_mag:.1%}.",
    "- Stage 3b/3c reference: site-aware pooled B-out R2 0.733, "
    "no-B-info 0.683, Magpie-only 0.086.",
    "",
    "## Notes",
    "",
    "- experiment: leave-one-B-out SHAP, tuned XGBoost only, seed 0.",
    "- 'share' is the group's total mean |SHAP| divided by the "
    "family's total mean |SHAP|; 'mean' divides the total by the "
    "number of features in the group, so large groups are not "
    "flattered for size.",
    "- Held-out R2 values are printed for context only and are not "
    "pass/fail: W is expected to be low (Stage 4 Magpie B-out W "
    "R2 -5.10; site-aware 0.226 in Stage 3b).",
    "- Figure: figures/shap_familyout.png (group shares per family).",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")

# --- 10. GATE 9a2. ---
# Files exist with the expected columns/rows, identity passed for
# every family, held-out R2 printed. R2 values are sanity only —
# they are NOT pass/fail, because W is expected to be low and the
# transfer numbers already live in Stages 3b/3c.
expected_rows = 232 if SMOKE else 696
expected_group_rows = 7 if SMOKE else 21
gate_lines = []
gate_failures = []
for name, expected_cols, exp_rows in [
        (csv_name, RESULT_COLS, expected_rows),
        (groups_name, GROUP_COLS, expected_group_rows)]:
    p = OUT_DIR / name
    if not p.exists():
        gate_failures.append(f"{name} missing")
        gate_lines.append(f"  [FAIL] {name} missing")
        continue
    got = pd.read_csv(p)
    if list(got.columns) != expected_cols:
        gate_failures.append(f"{name} has wrong columns: {list(got.columns)}")
        gate_lines.append(f"  [FAIL] {name} has wrong columns: {list(got.columns)}")
    elif len(got) != exp_rows:
        gate_failures.append(f"{name} has {len(got)} rows, expected {exp_rows}")
        gate_lines.append(f"  [FAIL] {name} has {len(got)} rows, expected {exp_rows}")
    else:
        gate_lines.append(f"  [PASS] {name}: {len(got)} rows, columns as expected")
if not fig_path.exists():
    gate_failures.append(f"figures/{fig_name} missing")
    gate_lines.append(f"  [FAIL] figures/{fig_name} missing")
else:
    gate_lines.append(f"  [PASS] figures/{fig_name} exists at 300 dpi")
for fam in FAMILIES:
    base, max_err = identity_results[fam]
    if np.isnan(base):
        gate_lines.append(f"  [INFO] SHAP identity {fam}: resumed, passed on first run")
    elif max_err < IDENTITY_TOL:
        gate_lines.append(f"  [PASS] SHAP identity {fam}: max error {max_err:.2e}")
    else:
        gate_failures.append(f"SHAP identity {fam} failed: {max_err:.2e}")
        gate_lines.append(f"  [FAIL] SHAP identity {fam}: max error {max_err:.2e}")
for fam in FAMILIES:
    m = family_metrics[fam]
    gate_lines.append(
        f"  [INFO] {fam} held-out R2 (sanity only): {m['r2']:.3f} "
        f"(RMSE {m['rmse']:.3f}, n {m['n_test']})")
print("GATE 9a2 — files, columns, SHAP identity, all families done:")
for line in gate_lines:
    print(line)

# --- GATE 9a2 verdict. ---
if SMOKE:
    print("STAGE 9a2 SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 9a2: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 9a2 FAILED: " + "; ".join(gate_failures))
else:
    print("GATE 9a2: PASS — family-out SHAP table, group totals, figure "
          "and report saved; identity check passed for all families. "
          "Next: read the overall mechanism sentence in "
          "shap_familyout_report_v2.md — it says whether site resolution "
          "takes over from Magpie under shift, then the Stage 9b "
          "graph-model comparison.")
    print("STAGE 9a2 done — paste back the GATE 9a2 line, the three "
          "held-out R2 values, and the top group per family + the "
          "overall mechanism sentence.")
