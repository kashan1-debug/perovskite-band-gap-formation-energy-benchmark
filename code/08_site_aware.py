# ============================================================
# v2 — STAGE 3b: Site-aware features, leave-one-B-out ONLY
# (ONE cell, paste into Colab)
# Run AFTER Stage 3 (GATE 3 PASS) — it does NOT need Stages 4-7,
# only features.parquet + audited.parquet from Stages 1-2.
# Run in the same session, or after Setup B then Setup C in a
# fresh session. It reloads the two parquet files from Drive —
# it does NOT redo Stages 1-7.
# Full run takes about 10-15 minutes in Colab. Every finished
# family is saved to Drive at once, so a disconnect loses
# nothing already done. If the run stops, just paste this cell
# again: families already in results_siteaware_v2.csv are
# skipped, not redone.
# THIS IS THE NEW STEP. Magpie-only models cannot tell A from B:
# A/B-swapped crystals share one Magpie input but have different
# energies, which caps Magpie-only R2 at 0.543 and makes
# leave-one-B-out fail (pooled Ridge 0.206 / XGBoost 0.086 in
# Stage 4). Site-aware features give the model, for each site
# separately, the element's own properties — so it can finally
# tell which element sits on the A site, which on B, which on X.
# NO random CV in this cell. Random CV only measures
# interpolation inside the closed grid (audit: Magpie + one-hot
# Ridge 0.82 / tuned XGBoost 0.76, default XGBoost 0.90 — never
# quote 0.90, it is the untuned config). The ONLY test here is
# leave-one-B-out, the hard test. If site-aware features cannot
# transfer to a new B metal, we say so — that is also a result.
# One-hot: B_site (52) + X_anion (7) ONLY, no A_site one-hot.
# A is never held out, so an A one-hot would just be an identity
# crutch the model could lean on for interpolation. A new B metal
# has a one-hot column that is all zero in training, so on the
# held-out metal the model MUST rely on the physics features,
# not the identity — that is the point of the test.
# Physics features: from pymatgen Element data only, no DFT.
# Ionic radius uses the audit's charge convention (A = smallest
# positive charge, B = largest positive charge, X = most negative
# charge; majority element for mixed anions), which gives 0 NaN
# on this dataset. NOTE: the task brief said "most common ionic
# radius (coordination-averaged)" — pymatgen's ionic_radii dict
# is keyed by CHARGE, not coordination, so the audit's charge
# convention is the method used here, and it is the one proven
# to give 0 NaN. NOTE 2: pymatgen has NO covalent_radius
# attribute in this version, so the empirical atomic_radius is
# used instead — documented here, not silently swapped.
# Models: Ridge and tuned XGBoost ONLY, seed 0. No Random Forest
# (208 retrains already; RF would make the run far too long).
# No band gap, no conformal, no CV in this cell.
# Paste back: the GATE 3b line and the pooled table (or the last
# rows printed, if it stopped part-way).
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs only leave-one-B-out for 3 B metals (Mg, Te, W),
# both models, both feature sets (12 rows), on the FULL 364-row
# families — and writes results_siteaware_v2_smoke.csv instead of
# results_siteaware_v2.csv. Use it only to check the code runs.
SMOKE = False

import csv
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element
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

# --- 1. Reload features.parquet + audited.parquet (do NOT redo Stages 1-7) ---
feat_path = OUT_DIR / "features.parquet"
audit_path = OUT_DIR / "audited.parquet"
for p, name in [(feat_path, "features.parquet"), (audit_path, "audited.parquet")]:
    if not p.exists():
        raise FileNotFoundError(
            f"{name} not found in OUT_DIR. Run Stages 1-2 first, "
            "then this cell again."
        )
df = pd.read_parquet(feat_path)
audited = pd.read_parquet(audit_path)
print("features.parquet shape:", df.shape, "(expected (18928, 150))")
print("audited.parquet shape:", audited.shape, "(expected (18928, 18))")
magpie_cols = [c for c in df.columns if c.startswith("MagpieData")]
print("Magpie columns:", len(magpie_cols), "(expected 132)")
for needed in ["A_site", "B_site", "X_anion", "e_form_per_atom"]:
    if needed not in df.columns:
        raise AssertionError(f"STOP: needed column {needed} is missing from features.parquet.")
if df.shape != (18928, 150) or len(magpie_cols) != 132:
    raise AssertionError(
        "STOP: features.parquet is not (18928, 150) with 132 Magpie columns. "
        "Re-run Stage 2 before modelling."
    )
# Row-by-row alignment between the two files — the site-aware
# features must be attached to the right rows.
align = bool(
    len(df) == len(audited)
    and (df["formula_ordered"].values == audited["formula_ordered"].values).all()
    and np.allclose(df["e_form"].values, audited["e_form"].values)
)
print("alignment features vs audited (formula_ordered, e_form):", align)
if not align:
    raise AssertionError(
        "ALIGNMENT FAILED: features.parquet does not match audited.parquet "
        "row by row. STOP — do not build site-aware features."
    )

# --- 2. Site-aware features, from elemental properties ONLY. ---
# No DFT columns in X, ever: vbm/cbm define the band gap directly,
# fermi level / fermi width / mu_b are DFT outputs, e_form /
# band_gap / e_form_per_atom are targets. Including any is cheating.
FORBIDDEN = ["vbm", "cbm", "fermi level", "fermi width", "mu_b",
             "e_form", "band_gap", "e_form_per_atom"]


def _ionic_radius(symbol, kind):
    """Shannon-type ionic radius (Angstrom) from Element.ionic_radii,
    audit charge convention: A = smallest positive charge,
    B = largest positive charge, X = most negative charge.
    Fallback (counted, expected 0): average_ionic_radius, then
    atomic_radius. Returns NaN only if all three are missing."""
    radii = Element(symbol).ionic_radii
    if radii:
        pos = [c for c in radii if c > 0]
        neg = [c for c in radii if c < 0]
        if kind == "A":
            charge = min(pos) if pos else None
        elif kind == "B":
            charge = max(pos) if pos else None
        else:
            charge = min(neg) if neg else None
        if charge is not None:
            return float(radii[charge]), False
    el = Element(symbol)
    for fallback in (el.average_ionic_radius, el.atomic_radius):
        if fallback is not None:
            v = float(fallback)
            if not math.isnan(v):
                return v, True
    return float("nan"), True


def _valence_spdf(symbol):
    """Valence s/p/d/f electron counts from full_electronic_structure.
    n_max = highest principal quantum number: s/p from n_max,
    d from n_max-1, f from n_max-2. E.g. Fe (3d6 4s2) -> s=2, p=0,
    d=6, f=0; W (4f14 5d4 6s2) -> s=2, p=0, d=4, f=14."""
    struct = {(n, sub): cnt for n, sub, cnt in Element(symbol).full_electronic_structure}
    n_max = max(n for n, _ in struct)
    s = struct.get((n_max, "s"), 0)
    p = struct.get((n_max, "p"), 0)
    d = struct.get((n_max - 1, "d"), struct.get((n_max, "d"), 0))
    f = struct.get((n_max - 2, "f"), 0)
    return s, p, d, f


def _majority_x_element(x_label):
    """Majority element of a mixed-anion label like 'O2F' -> 'O'.
    A documented approximation, same as the audit's tolerance factor."""
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", x_label)
    return max(parts, key=lambda p: int(p[1]) if p[1] else 1)[0]


def _site_props(symbol, kind):
    """12 elemental properties for one site element. Returns a dict
    and whether the ionic radius needed the fallback."""
    el = Element(symbol)
    r_ion, used_fallback = _ionic_radius(symbol, kind)
    s, p, d, f = _valence_spdf(symbol)
    return {
        "Z": float(el.Z),
        "mass": float(el.atomic_mass),
        "EN": float(el.X) if el.X is not None and not (isinstance(el.X, float) and math.isnan(el.X)) else float("nan"),
        "r_ion": r_ion,
        # atomic_radius, NOT covalent_radius: pymatgen has no
        # covalent_radius attribute in this version.
        "r_atom": float(el.atomic_radius) if el.atomic_radius is not None else float("nan"),
        "ionE": float(el.ionization_energy),
        "EA": float(el.electron_affinity) if el.electron_affinity is not None else float("nan"),
        "val_s": float(s),
        "val_p": float(p),
        "val_d": float(d),
        "val_f": float(f),
        "mendeleev": float(el.mendeleev_no),
    }, used_fallback


# One property dict per distinct element per site kind (52 A + 52 B
# + 4 X elements — computed once, then mapped onto rows).
site_kinds = {"A": sorted(df["A_site"].unique().tolist()),
              "B": sorted(df["B_site"].unique().tolist()),
              "X": sorted({_majority_x_element(l) for l in df["X_anion"].unique().tolist()})}
props = {}
fallback_count = 0
for kind, symbols in site_kinds.items():
    for sym in symbols:
        props[(kind, sym)], fb = _site_props(sym, kind)
        fallback_count += int(fb)
print(f"Site elements: A {len(site_kinds['A'])}, B {len(site_kinds['B'])}, "
      f"X-majority {len(site_kinds['X'])}; ionic-radius fallbacks used: "
      f"{fallback_count} (expected 0)")

phys = {}
for kind, col in [("A", "A_site"), ("B", "B_site")]:
    for key in ["Z", "mass", "EN", "r_ion", "r_atom", "ionE", "EA",
                "val_s", "val_p", "val_d", "val_f", "mendeleev"]:
        phys[f"{kind}_{key}"] = df[col].map(lambda s, k=kind, kk=key: props[(k, s)][kk])
x_el = df["X_anion"].map(_majority_x_element)
for key in ["Z", "mass", "EN", "r_ion", "r_atom", "ionE", "EA",
            "val_s", "val_p", "val_d", "val_f", "mendeleev"]:
    phys[f"X_{key}"] = x_el.map(lambda s, kk=key: props[("X", s)][kk])
phys_df = pd.DataFrame(phys)
# Derived physics: Goldschmidt tolerance factor t, octahedral factor
# mu, and the site differences that set the band edges / stability.
phys_df["tolerance_factor"] = ((phys_df["A_r_ion"] + phys_df["X_r_ion"])
                               / (math.sqrt(2.0) * (phys_df["B_r_ion"] + phys_df["X_r_ion"])))
phys_df["octahedral_factor"] = phys_df["B_r_ion"] / phys_df["X_r_ion"]
phys_df["rA_minus_rB"] = phys_df["A_r_ion"] - phys_df["B_r_ion"]
phys_df["EN_A_minus_EN_B"] = phys_df["A_EN"] - phys_df["B_EN"]
phys_df["EN_B_minus_EN_X"] = phys_df["B_EN"] - phys_df["X_EN"]
phys_cols = list(phys_df.columns)
print(f"Physics columns: {len(phys_cols)} (12 per site x 3 = 36 + 5 derived)")
print("Physics NaN cells:", int(phys_df.isna().sum().sum()), "(expected 0)")
if int(phys_df.isna().sum().sum()) != 0:
    raise AssertionError(
        "STOP: physics features contain NaN cells. The ionic-radius "
        "fallback did not cover every element — find which one first."
    )

# One-hot: B_site (52) + X_anion (7) ONLY, no A_site one-hot (see
# header). Fixed category lists, so the columns are identical in
# every train/test split and in smoke/full runs.
B_CATS = sorted(df["B_site"].unique().tolist())
X_CATS = sorted(df["X_anion"].unique().tolist())
oh_b = pd.get_dummies(df["B_site"], prefix="B").reindex(
    columns=[f"B_{c}" for c in B_CATS], fill_value=0).astype(float)
oh_x = pd.get_dummies(df["X_anion"], prefix="X").reindex(
    columns=[f"X_{c}" for c in X_CATS], fill_value=0).astype(float)
oh_cols = list(oh_b.columns) + list(oh_x.columns)
print(f"One-hot columns: {len(oh_cols)} (B {len(oh_b.columns)} + X {len(oh_x.columns)})")

# --- 3. Build the two feature sets. ---
# (a) Magpie only (132) — the control, recomputed here, not copied
#     from Stage 4. (b) Site-aware = Magpie + one-hot + physics (232).
X_mag = df[magpie_cols].values
X_sa = np.hstack([X_mag, oh_b.values, oh_x.values, phys_df.values])
sa_cols = magpie_cols + oh_cols + phys_cols
print(f"Magpie-only X: {X_mag.shape} | site-aware X: {X_sa.shape} "
      f"(expected (18928, 132) and (18928, {132 + len(oh_cols) + len(phys_cols)}))")
for name, cols in [("Magpie-only", magpie_cols), ("site-aware", sa_cols)]:
    leak = [c for c in cols if c in FORBIDDEN]
    print(f"Leakage columns in {name} X (should be []):", leak)
    if leak:
        raise AssertionError(f"STOP: forbidden columns in {name} X: {leak}")
y = df["e_form_per_atom"].values
print("y = e_form_per_atom, n =", len(y))

# Save the site-aware feature frame (audited cols + Magpie + one-hot
# + physics), backup first, safe to run twice. String ID columns are
# fine for parquet; only Structure/Composition objects were the v1
# bug, and neither is present here.
sa_frame = pd.concat([df.reset_index(drop=True),
                      oh_b.reset_index(drop=True), oh_x.reset_index(drop=True),
                      phys_df.reset_index(drop=True)], axis=1)
sa_path = OUT_DIR / "features_siteaware.parquet"
if sa_path.exists():
    shutil.copy2(sa_path, OUT_DIR / "features_siteaware_backup.parquet")
    print("backup saved: features_siteaware_backup.parquet")
sa_frame.to_parquet(sa_path, index=False)
print("saved features_siteaware.parquet:", sa_frame.shape)
reloaded_sa = pd.read_parquet(sa_path)
print("reloaded shape:", reloaded_sa.shape, "(expected", str(sa_frame.shape) + ")")
if reloaded_sa.shape != sa_frame.shape:
    raise AssertionError("STOP: saved site-aware frame does not re-open with the same shape.")

# --- 4. Models — TUNED configs only, never defaults. Seed 0 only. ---
# Leave-one-B-out is deterministic given the config: the split is
# fixed (one B metal held out), so there is no seed loop. Seed 0 is
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
FEATURE_SETS = {"magpie": X_mag, "siteaware": X_sa}
print("Models (tuned only, seed 0):")
print("  Ridge: StandardScaler + Ridge(alpha=1.0)")
print("  XGBoost: n_estimators=100, max_depth=3, lr=0.05, "
      "subsample=0.8, colsample_bytree=0.8, reg_lambda=5")
print("Feature sets: magpie (132 cols, control) and siteaware "
      f"({X_sa.shape[1]} cols)")

# --- 5. Families: 52 B metals, 364 rows each. ---
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
print(f"Full run = 52 B x 2 models x 2 feature sets = 208 rows "
      f"(SMOKE = 3 B x 2 x 2 = 12 rows).")

# --- 6. Results file: ONE fixed column set, checkpoint after every row. ---
# Resume-by-skipping: if results_siteaware_v2.csv already exists (a
# re-run after a disconnect), rows already in it are skipped, not
# redone. The file is never wiped on a re-run. In SMOKE mode a
# separate file is used so test rows can never mix into real results.
# Pooled metrics need the actual predictions, not just the per-family
# R2s, so every family's predictions are also saved, one row per
# test crystal, to predictions_siteaware_v2.parquet. A family counts
# as DONE only if it is in BOTH files: if a re-run finds a row in
# the CSV but its predictions are missing, that row is dropped from
# the CSV and the family is re-run, so the two files can never disagree.
CSV_COLS = ["feature_set", "family_type", "family", "model", "n_test",
            "r2", "rmse", "mae", "experiment"]
csv_name = "results_siteaware_v2_smoke.csv" if SMOKE else "results_siteaware_v2.csv"
csv_path = OUT_DIR / csv_name
pred_name = ("predictions_siteaware_v2_smoke.parquet" if SMOKE
             else "predictions_siteaware_v2.parquet")
pred_path = OUT_DIR / pred_name
EXPERIMENT = "smoke" if SMOKE else "siteaware_b_out"

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
    # Drop CSV rows whose predictions are missing — they will be re-run.
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


def run_one(feature_set, family, model_name):
    """One B family, one model, one feature set: train on every other
    row, test on this family's 364 rows."""
    t0 = time.time()
    X = FEATURE_SETS[feature_set]
    te_mask = (df["B_site"] == family).values
    tr_mask = ~te_mask
    m = make_model(model_name)
    m.fit(X[tr_mask], y[tr_mask])
    pred = m.predict(X[te_mask])
    yt = y[te_mask]
    row = {
        "feature_set": feature_set,
        "family_type": "B",
        "family": family,
        "model": model_name,
        "n_test": int(te_mask.sum()),
        "r2": float(r2_score(yt, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yt, pred))),
        "mae": float(mean_absolute_error(yt, pred)),
        "experiment": EXPERIMENT,
    }
    print(f"[{feature_set} B {family} {model_name}] done in {time.time() - t0:.0f} s — "
          f"per-family R2: {row['r2']:.3f}")
    save_row(row, yt, pred)
    return row


# --- 7. Run everything, checkpointing after every family. ---
all_rows = []
for feature_set in ["magpie", "siteaware"]:
    for family in B_FAMILIES:
        for model_name in MODELS:
            key = (feature_set, "B", family, model_name)
            if key in done:
                print(f"SKIP {feature_set} B {family} {model_name} — already in {csv_name}.")
                continue
            all_rows.append(run_one(feature_set, family, model_name))
print(f"Rows finished in this run: {len(all_rows)} "
      f"(total in {csv_name} after this run: {len(done) + len(all_rows)}).")

# --- 8. Pooled headline + per-family summary (from the saved files). ---
# POOLED is the headline: concatenate ALL test predictions for one
# (feature_set, model) and compute R2 / RMSE / MAE once on the pooled
# predictions. It is NOT the mean of the per-family R2s.
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
for (feature_set, model_name), sub in preds.groupby(["feature_set", "model"]):
    pm = pooled_metrics(sub)
    pooled_rows.append({
        "feature_set": feature_set,
        "model": model_name,
        "n_test": pm["n"],
        "r2_pooled": pm["r2"],
        "rmse_pooled": pm["rmse"],
        "mae_pooled": pm["mae"],
    })
pooled = pd.DataFrame(pooled_rows).sort_values(["feature_set", "model"])
print("POOLED headline (all test predictions together, per feature_set x model):")
print(pooled.to_string(index=False))

summary_rows = []
for (feature_set, model_name), sub in results.groupby(["feature_set", "model"]):
    summary_rows.append({
        "feature_set": feature_set,
        "model": model_name,
        "mean_r2": float(sub["r2"].mean()),
        "median_r2": float(sub["r2"].median()),
        "min_r2": float(sub["r2"].min()),
        "max_r2": float(sub["r2"].max()),
        "worst_family": str(sub.loc[sub["r2"].idxmin(), "family"]),
        "best_family": str(sub.loc[sub["r2"].idxmax(), "family"]),
    })
summary = pd.DataFrame(summary_rows).sort_values(["feature_set", "model"])
print("Per-family R2 summary (SECONDARY — context only, never the headline):")
print(summary.to_string(index=False))

# Control check: the Magpie-only B-out pooled numbers recomputed
# here must match Stage 4 (Ridge 0.206, XGBoost 0.086) within
# +/-0.03. If they do not, something changed in the features or
# the models — stop and find why before trusting the site-aware
# numbers.
STAGE4_REF = {"Ridge": 0.206, "XGBoost": 0.086}
control_lines = []
control_failures = []
if SMOKE:
    # Smoke pooled numbers cover only 3 B families, so they cannot
    # match the full 52-family Stage 4 pooled numbers — do NOT
    # compare them here, or the smoke run would print a false FAIL.
    control_lines.append(
        "  [INFO] Control check SKIPPED in SMOKE mode — smoke pooled "
        "numbers cover only 3 B families, not all 52.")
else:
    for model_name, ref in STAGE4_REF.items():
        sub = pooled[(pooled["feature_set"] == "magpie") & (pooled["model"] == model_name)]
        if len(sub) == 0:
            control_lines.append(f"  [FAIL] magpie control {model_name}: no pooled row")
            control_failures.append(f"magpie control {model_name}: no pooled row")
            continue
        v = float(sub["r2_pooled"].iloc[0])
        ok = abs(v - ref) <= 0.03 + 1e-9
        status = "PASS" if ok else "FAIL"
        control_lines.append(
            f"  [{status}] magpie control {model_name}: pooled {v:.3f} "
            f"vs Stage 4 {ref:.3f} (diff {v - ref:+.3f})")
        if not ok:
            control_failures.append(f"magpie control {model_name}: {v:.3f} vs Stage 4 {ref:.3f}")
print("Control check — Magpie-only B-out recomputed here vs Stage 4:")
for line in control_lines:
    print(line)


def gate3b_check(pooled_df):
    """GATE 3b — pre-registered success criterion: site-aware pooled
    B-out R2 > 0.25 for at least one model counts as 'breaks the
    Magpie-only B-out failure' (Magpie-only pooled: Ridge 0.206 /
    XGBoost 0.086). FAIL is also a result — even site-aware features
    do not transfer to a new B metal — not a reason to hide the
    numbers. Leakage alarm: site-aware pooled B-out R2 > 0.60 is
    suspicious (a B identity column may have leaked) and fails."""
    lines, failures = [], []
    best_sa = None
    leak = False
    for model_name in MODELS:
        sub = pooled_df[(pooled_df["feature_set"] == "siteaware")
                        & (pooled_df["model"] == model_name)]
        if len(sub) == 0:
            failures.append(f"siteaware {model_name}: no pooled row")
            lines.append(f"  [FAIL] siteaware {model_name}: no pooled row")
            continue
        v = float(sub["r2_pooled"].iloc[0])
        best_sa = v if best_sa is None else max(best_sa, v)
        if v > 0.60:
            leak = True
            failures.append(
                f"LEAKAGE ALARM: siteaware {model_name} pooled R2 {v:.3f} > 0.60")
            lines.append(
                f"  [FAIL] siteaware {model_name}: pooled R2 {v:.3f} "
                "— above 0.60, suspicious: a B identity column may have leaked.")
        else:
            lines.append(
                f"  [INFO] siteaware {model_name}: pooled B-out R2 {v:.3f} "
                f"(criterion: > 0.25 to break the Magpie-only failure)")
    if best_sa is not None and best_sa > 0.25 and not leak:
        lines.append(
            f"  [PASS] Best site-aware pooled B-out R2 = {best_sa:.3f} > 0.25 "
            "— site-aware features BREAK the Magpie-only B-out failure.")
    elif leak:
        lines.append(
            "  [FAIL] The 0.25 criterion cannot be judged — a leakage "
            "alarm fired above. Fix the leak first, do not claim a pass.")
    else:
        failures.append(
            "site-aware pooled B-out R2 did not beat 0.25 for either model")
        lines.append(
            "  [FAIL] Best site-aware pooled B-out R2 did not beat 0.25 — "
            "even site-aware features do not transfer to a new B metal. "
            "That is a RESULT, not a reason to hide the numbers.")
    return lines, failures


# --- 9. GATE 3b — pre-registered success criterion (full run only). ---
gate_lines = []
gate_failures = []
if SMOKE:
    print("GATE 3b: SKIPPED in SMOKE mode — smoke runs only 3 B families, "
          "so pooled numbers cannot be compared with the 0.25 criterion.")
    gate_lines.append("GATE 3b: SKIPPED (SMOKE mode — no criterion check).")
    gate_lines += control_lines
else:
    print("GATE 3b — site-aware pooled B-out R2 > 0.25 for at least one model:")
    gate_lines, gate_failures = gate3b_check(pooled)
    for line in gate_lines:
        print(line)
    gate_failures += control_failures
    if control_failures:
        print("Control check FAILED — see above. Do not trust the "
              "site-aware numbers until the control matches Stage 4.")

# --- 10. Save the Stage 3b report. ---
report_name = "siteaware_report_v2_smoke.md" if SMOKE else "siteaware_report_v2.md"
report_lines = [
    "# Perovskite v2 — Stage 3b site-aware leave-one-B-out report",
    "",
    "Site-aware features = Magpie (132) + one-hot B (52) + one-hot X (7) "
    "+ per-site physics (36) + derived t/mu/differences (5) = 232 columns. "
    "No A_site one-hot. No DFT columns in X. No random CV in this stage — "
    "leave-one-B-out only, the hard test.",
    "",
    "## Feature groups",
    "",
    f"- Magpie: {len(magpie_cols)} columns",
    f"- One-hot B: {len(oh_b.columns)}, one-hot X: {len(oh_x.columns)}",
    f"- Physics: {len(phys_cols)} columns (12 per site x 3 + 5 derived: "
    "tolerance_factor, octahedral_factor, rA_minus_rB, "
    "EN_A_minus_EN_B, EN_B_minus_EN_X)",
    f"- Physics NaN cells: {int(phys_df.isna().sum().sum())}; "
    f"ionic-radius fallbacks used: {fallback_count}",
    "- Ionic radius: audit charge convention (A smallest positive, "
    "B largest positive, X most negative, majority X element). "
    "atomic_radius used in place of covalent_radius (pymatgen has no "
    "covalent_radius attribute in this version).",
    "",
    "## Pooled headline (lead with this, not mean per-family R2)",
    "",
    pooled.to_string(index=False),
    "",
    "## Per-family R2 summary (secondary — context only)",
    "",
    summary.to_string(index=False),
    "",
    "## Control check — Magpie-only B-out vs Stage 4",
    "",
]
report_lines += control_lines
report_lines += [
    "",
    "## GATE 3b — site-aware pooled B-out R2 > 0.25",
    "",
]
report_lines += gate_lines
report_lines += [
    "",
    "## Notes",
    "",
    "- experiment column: 'siteaware_b_out' for the full run, 'smoke' for test runs.",
    "- Pooled R2 is computed once on all pooled test predictions per "
    "(feature_set, model); it is not the mean of per-family R2s.",
    "- Random-CV site-aware numbers from the audit (Ridge 0.82 / tuned "
    "XGBoost 0.76, default XGBoost 0.90) are INTERPOLATION only and "
    "must never be quoted as this stage's result.",
    "- If GATE 3b failed, that is a result: even site-aware features "
    "do not transfer to a new B metal. Report it, do not hide it.",
    "",
]
(OUT_DIR / report_name).write_text("\n".join(report_lines) + "\n")
print(f"saved {report_name}")
print(f"saved {csv_name}: {len(results)} rows")
print(f"saved {pred_name}: {len(preds)} predictions")
print(f"saved features_siteaware.parquet: {sa_frame.shape}")

# --- GATE 3b verdict. ---
if SMOKE:
    print("STAGE 3b SMOKE done — code runs end to end. "
          "Now set SMOKE = False and run the full cell in Colab.")
elif gate_failures:
    print("GATE 3b: FAIL — " + "; ".join(gate_failures))
    raise AssertionError("GATE 3b FAILED: " + "; ".join(gate_failures))
else:
    print("GATE 3b: PASS — site-aware features break the Magpie-only "
          "B-out failure (pooled B-out R2 > 0.25). Next: Stage 8 paper draft.")
    print("STAGE 3b done — paste back the GATE 3b line and the pooled table.")
