# ============================================================
# v2 — STAGE 9b: Graph-model comparison, PRETRAINED MEGNet zero-shot
# (ONE cell, paste into Colab)
# Run AFTER Stages 1-7 + 3b/3c (needs audited.parquet for the
# alignment check; needs Castelli structures, reloaded here).
# Run in the same session, or after Setup B then Setup C in a
# fresh session. It reloads audited.parquet from Drive and loads
# Castelli from matminer — it does NOT redo Stages 1-7.
# SETUP NOTE: this cell needs TWO extra installs beyond Setup A:
#   !pip install -q megnet
# (TensorFlow is already in Colab. Do NOT pip-install tensorflow
# in Colab unless the cell tells you it is missing.)
# Full run takes about 6-10 minutes in Colab. Every finished
# family is saved to Drive at once, so a disconnect loses
# nothing already done. If the run stops, just paste this cell
# again: families already in results_megnet_v2.csv are skipped.
#
# WHAT THIS CELL DOES, in plain words:
# A graph model (MEGNet) reads the actual crystal structure —
# which atom sits where — instead of only the recipe. It is the
# comparison a journal reviewer will demand: does structure beat
# our site-aware XGBoost (pooled leave-one-B-out R2 0.733)?
# Here MEGNet is used PRETRAINED (trained by its authors on
# Materials Project data, 133,420 structures, 2019 snapshot)
# and evaluated ZERO-SHOT: it is never trained on our data at
# all. That is both its strength (no training time) and its
# weakness (see the two warnings below).
#
# WARNING 1 — READ THIS BEFORE YOU PANIC AT THE RAW R2:
# MEGNet predicts formation energies on the MATERIALS PROJECT
# scale (about -3.4 eV/atom for SrTiO3). Our Castelli targets
# are on a DIFFERENT scale (about -0.05 eV/atom for SrTiO3,
# because Castelli references oxygen to water vapour, not O2).
# The two scales differ by roughly -1 eV/atom on average, so
# the RAW R2 will be hugely negative (about -100). That is
# EXPECTED — it is a units/reference difference, not a bug and
# not proof that MEGNet is useless. The honest numbers are:
# (a) Spearman rank correlation — does MEGNet rank crystals
#     in the right order inside a held-out family? (This does
#     not care about the scale at all.)
# (b) CALIBRATED R2 — one straight-line rescaling
#     (y_castelli = a * y_megnet + b) fitted on 2,000 random
#     rows that are NEVER in the held-out family, then applied
#     to that family. This is what a real user would do:
#     recalibrate the pretrained model on data they have.
# Report (a) and (b). Never quote the raw R2 alone.
#
# WARNING 2 — CONTAMINATION (measured, not guessed):
# MEGNet was trained on Materials Project data. We measured how
# much of our grid already appears there, by composition, in
# the Oct-2018 MP snapshot (the closest public snapshot to its
# 2019 training set): 1,175 of 18,928 Castelli rows (6.2%)
# share a reduced formula with MP; the overlap is concentrated
# in the oxides (39.8% of O3 rows, 31-44% of the oxide rows
# inside the Mg/Te/W families). Same formula is NOT the same
# structure (MP entries are relaxed, ours are ideal cubic), but
# the model has seen those compositions before. So the zero-shot
# numbers are slightly FLATTERED for oxides, and the true
# training set (MP 2019, 133,420 structures) is bigger than the
# 2018 snapshot, so the real overlap is somewhat above 6.2%.
# The comparison is therefore an UPPER BOUND on what a clean
# pretrained model would do. A from-scratch graph model (no
# pretraining) is the clean comparison — see the Stage 9b
# report; it is NOT in this cell, because the megnet package
# cannot currently be trained from scratch on the modern
# Colab stack (Keras 3 incompatibilities — see report).
#
# HOW THE TWO MEGNET BUGS ARE WORKED AROUND (proven in the VM):
# (1) megnet 1.3.2 imports BrunnerNN_real etc., renamed in
#     pymatgen 2026 — fixed here with a 3-line alias shim
#     BEFORE importing megnet.
# (2) megnet's from_file() loads the saved model WITH its old
#     Keras-2 optimizer, which Keras 3 (TF 2.21 in Colab)
#     refuses — fixed here by loading with compile=False
#     (prediction needs no optimizer) and rebuilding the
#     GraphModel by hand from the .json sidecar.
# Do NOT "fix" these by downgrading pymatgen or tensorflow —
# that would break the rest of the v2 workflow in this session.
# Models: pretrained MEGNet Eform_MP_2019 ONLY. No training,
# no tuning, no defaults to set (there is nothing to tune in
# a zero-shot evaluation — the model is used exactly as its
# authors shipped it). Seed 0 only for the calibration sample.
# Paste back: the GATE 9b line and the pooled table.
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True runs only leave-one-B-out for 3 B metals (Mg, Te, W),
# both raw and calibrated, and writes results_megnet_v2_smoke.csv
# instead of results_megnet_v2.csv. Use it only to check the
# code runs.
SMOKE = False

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
print("SMOKE =", SMOKE, "(True = 3 families only, False = all 52)")
T0 = time.time()

# --- 0. Imports with a loud failure if megnet is missing ---
try:
    import pymatgen.analysis.local_env as _le
except ImportError as e:
    raise ImportError("pymatgen is missing. Run Setup B first.") from e
# SHIM (proven in the VM): megnet 1.3.2 imports BrunnerNN_real /
# BrunnerNN_reciprocal / BrunnerNN_relative, renamed in pymatgen 2026.
# Alias the old names to the new classes BEFORE importing megnet.
for _old, _new in [
    ("BrunnerNN_real", "BrunnerNNReal"),
    ("BrunnerNN_reciprocal", "BrunnerNNReciprocal"),
    ("BrunnerNN_relative", "BrunnerNNRelative"),
]:
    if not hasattr(_le, _old) and hasattr(_le, _new):
        setattr(_le, _old, getattr(_le, _new))
        print(f"shim: aliased {_old} -> {_new}")
try:
    import megnet
    from megnet.layers import _CUSTOM_OBJECTS
    from megnet.models.base import GraphModel
    from megnet.utils import models as _mm
    from monty.serialization import loadfn
    from tensorflow.keras.models import load_model
except ImportError as e:
    raise ImportError(
        "megnet (or tensorflow) is missing. Run: !pip install -q megnet "
        "in a setup cell, then restart the runtime and run Setup B, "
        "Setup C, then this cell again. Original error: " + str(e)
    ) from e
print("megnet", megnet.__version__)
import tensorflow as tf

print("tensorflow", tf.__version__)

# --- 1. Reload Castelli structures + audited.parquet (alignment check) ---
from matminer.datasets import load_dataset

audit_path = OUT_DIR / "audited.parquet"
if not audit_path.exists():
    raise FileNotFoundError(
        "audited.parquet not found in OUT_DIR. Run Stages 1-2 first."
    )
audited = pd.read_parquet(audit_path)
print("audited.parquet shape:", audited.shape, "(expected (18928, 18))")
df = load_dataset("castelli_perovskites")
print("Castelli shape:", df.shape, "(expected (18928, 10))")
df["e_form_per_atom"] = df["e_form"] / 5.0
# A/B/X from site order ONLY (D5, validated in Stage 1): site 1 = B.
df["B_site"] = [s.sites[1].specie.symbol for s in df["structure"]]
df["formula_ordered"] = df["formula"]
assert (df["structure"].apply(lambda s: s.num_sites) == 5).all(), \
    "STOP: not every structure has 5 sites."
align = bool(
    len(df) == len(audited)
    and (df["formula_ordered"].values == audited["formula_ordered"].values).all()
    and np.allclose(df["e_form"].values, audited["e_form"].values)
)
print("alignment Castelli vs audited (formula_ordered, e_form):", align)
assert align, "STOP: Castelli reload does not match audited.parquet."
print("B families:", sorted(df["B_site"].unique()), "(expected 52)")

# --- 2. Load pretrained MEGNet Eform_MP_2019 (compile=False workaround) ---
# megnet's own load_model()/from_file() loads the saved model WITH its
# Keras-2 Adam optimizer, which Keras 3 refuses ("Argument(s) not
# recognized: {'lr': ...}"). Prediction needs no optimizer, so load
# with compile=False and rebuild the GraphModel from the .json sidecar.
_model_dir = Path(_mm.__file__).parent / "mvl_models" / "mp-2019.4.1"
_h5 = _model_dir / "formation_energy.hdf5"
if not _h5.exists():
    print("MEGNet model files not found — downloading (~5 MB zip)...")
    _mm._download_models()
    assert _h5.exists(), "STOP: MEGNet download did not produce the model file."
t1 = time.time()
_configs = loadfn(str(_h5) + ".json")
_keras_model = load_model(str(_h5), custom_objects=_CUSTOM_OBJECTS, compile=False)
model = GraphModel(**{**_configs, "model": _keras_model})
print(f"MEGNet Eform_MP_2019 loaded in {time.time()-t1:.1f}s")
print("model metadata:", model.metadata)
print("model inputs:", [(i.name, tuple(i.shape)) for i in _keras_model.inputs])

# --- 3. Warm-up + reference-scale check on SrTiO3 ---
from pymatgen.core import Lattice, Structure

_s = Structure(
    Lattice.cubic(3.905),
    ["Sr", "Ti", "O", "O", "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
)
t1 = time.time()
_warm = float(model.predict_structure(_s)[0])
print(f"warm-up SrTiO3 MEGNet = {_warm:.3f} eV/atom ({time.time()-t1:.1f}s)")
_row = df[df["formula"] == "SrTiO3"]
assert len(_row) == 1, "STOP: SrTiO3 not found exactly once in Castelli."
print(f"SrTiO3 Castelli e_form_per_atom = {_row['e_form_per_atom'].iloc[0]:.3f} "
      f"-> offset (MEGNet - Castelli) = {_warm - _row['e_form_per_atom'].iloc[0]:+.3f} "
      f"eV/atom. THIS offset is why raw R2 will be hugely negative — read "
      f"the CALIBRATED R2 and Spearman instead.")

# --- 4. Calibration pool: 2,000 random rows, seed 0 (predicted ONCE) ---
# For each held-out family F, the affine map is fitted ONLY on pool
# rows with B != F, so no test row ever enters its own calibration.
CALIB_N = 2000
calib_path = OUT_DIR / ("predictions_megnet_calib_smoke.parquet" if SMOKE
                        else "predictions_megnet_calib.parquet")
rng = np.random.default_rng(0)
_cal_idx = rng.choice(len(df), size=CALIB_N, replace=False)
_cal = df.iloc[_cal_idx].reset_index(drop=True)
if calib_path.exists():
    calib_df = pd.read_parquet(calib_path)
    assert len(calib_df) == CALIB_N, "STOP: calibration file has wrong size — delete it and re-run."
    print(f"calibration pool reloaded: {len(calib_df)} rows (not re-predicted)")
else:
    t1 = time.time()
    _cal_pred = np.array(
        model.predict_structures(list(_cal["structure"]), batch_size=64)).ravel()
    calib_df = pd.DataFrame({
        "formula_ordered": _cal["formula_ordered"].values,
        "B_site": _cal["B_site"].values,
        "y_true": _cal["e_form_per_atom"].to_numpy(),
        "y_pred_raw": _cal_pred,
    })
    calib_df.to_parquet(calib_path, index=False)
    print(f"calibration pool predicted: {len(calib_df)} rows in {time.time()-t1:.1f}s")
_cal_r2 = r2_score(calib_df["y_true"], calib_df["y_pred_raw"])
print(f"calibration pool raw R2 = {_cal_r2:.3f} (will be very negative — scale offset, expected)")

# --- 5. Leave-one-B-out loop (pretrained MEGNet, zero-shot) ---
RESULT_COLS = ["family_B", "n", "r2_raw", "rmse_raw", "mae_raw", "spearman",
               "bias", "r2_cal", "rmse_cal", "mae_cal", "calib_a", "calib_b",
               "calib_n", "time_s"]
PRED_COLS = ["family_B", "formula_ordered", "B_site", "y_true",
             "y_pred_raw", "y_pred_cal"]
results_path = OUT_DIR / ("results_megnet_v2_smoke.csv" if SMOKE
                          else "results_megnet_v2.csv")
pred_path = OUT_DIR / ("predictions_megnet_v2_smoke.parquet" if SMOKE
                       else "predictions_megnet_v2.parquet")
# Rows are kept in plain lists and written out after every family.
# (Building an empty DataFrame first and concatenating rows into it
# triggers a pandas FutureWarning and mixes dtypes — lists avoid that.)
if results_path.exists():
    results_df = pd.read_csv(results_path)
    assert list(results_df.columns) == RESULT_COLS, \
        "STOP: existing results file has different columns — do not mix schemas."
    results_rows = results_df.to_dict("records")
    done = set(results_df["family_B"])
    print(f"resume: {len(done)} families already done, skipping them")
else:
    results_rows = []
    done = set()
if pred_path.exists():
    pred_all = pd.read_parquet(pred_path)
    assert list(pred_all.columns) == PRED_COLS, \
        "STOP: existing predictions file has different columns."
    pred_frames = [pred_all]
else:
    pred_frames = []

families = ["Mg", "Te", "W"] if SMOKE else sorted(df["B_site"].unique())
print(f"families to run: {len(families)} "
      f"(total rows per family: 364)")
for fam in families:
    if fam in done:
        print(f"[{fam}] skipped (already done)")
        continue
    t1 = time.time()
    sub = df[df["B_site"] == fam].reset_index(drop=True)
    y = sub["e_form_per_atom"].to_numpy()
    pred = np.array(
        model.predict_structures(list(sub["structure"]), batch_size=64)).ravel()
    # Calibration for THIS family: affine fit on pool rows with B != fam.
    pool = calib_df[calib_df["B_site"] != fam]
    assert len(pool) >= 500, f"STOP: calibration pool too small for {fam}."
    A = np.vstack([pool["y_pred_raw"].to_numpy(),
                   np.ones(len(pool))]).T
    (a, b), *_ = np.linalg.lstsq(A, pool["y_true"].to_numpy(), rcond=None)
    pred_cal = a * pred + b
    r2_raw = float(r2_score(y, pred))
    rmse_raw = float(np.sqrt(mean_squared_error(y, pred)))
    mae_raw = float(mean_absolute_error(y, pred))
    spear = float(pd.Series(y).corr(pd.Series(pred), method="spearman"))
    bias = float(np.mean(pred - y))
    r2_cal = float(r2_score(y, pred_cal))
    rmse_cal = float(np.sqrt(mean_squared_error(y, pred_cal)))
    mae_cal = float(mean_absolute_error(y, pred_cal))
    dt = time.time() - t1
    print(f"[{fam}] n={len(y)} {dt:.1f}s | RAW R2={r2_raw:.3f} RMSE={rmse_raw:.3f} "
          f"MAE={mae_raw:.3f} Spearman={spear:.3f} bias={bias:+.3f} | "
          f"CAL R2={r2_cal:.3f} RMSE={rmse_cal:.3f} MAE={mae_cal:.3f} "
          f"(a={a:.3f}, b={b:+.3f}, n_cal={len(pool)})", flush=True)
    row = {"family_B": fam, "n": len(y), "r2_raw": r2_raw, "rmse_raw": rmse_raw,
           "mae_raw": mae_raw, "spearman": spear, "bias": bias,
           "r2_cal": r2_cal, "rmse_cal": rmse_cal, "mae_cal": mae_cal,
           "calib_a": float(a), "calib_b": float(b), "calib_n": len(pool),
           "time_s": round(dt, 1)}
    results_rows.append(row)
    pd.DataFrame(results_rows, columns=RESULT_COLS).to_csv(results_path,
                                                          index=False)
    fam_pred = pd.DataFrame({
        "family_B": fam, "formula_ordered": sub["formula_ordered"].values,
        "B_site": fam, "y_true": y, "y_pred_raw": pred, "y_pred_cal": pred_cal,
    })
    # Drop any stale rows for this family (re-run safety), then append.
    pred_frames = [f[f["family_B"] != fam] for f in pred_frames]
    pred_frames.append(fam_pred)
    pd.concat(pred_frames, ignore_index=True).to_parquet(pred_path,
                                                        index=False)
    done.add(fam)

# --- 6. Pooled metrics recomputed from saved predictions (never from means) ---
results_df = pd.read_csv(results_path)
pred_all = pd.read_parquet(pred_path)
y_all = pred_all["y_true"].to_numpy()
p_raw = pred_all["y_pred_raw"].to_numpy()
p_cal = pred_all["y_pred_cal"].to_numpy()
pooled = {
    "n_families": len(results_df),
    "n_rows": len(pred_all),
    "pooled_r2_raw": float(r2_score(y_all, p_raw)),
    "pooled_rmse_raw": float(np.sqrt(mean_squared_error(y_all, p_raw))),
    "pooled_mae_raw": float(mean_absolute_error(y_all, p_raw)),
    "pooled_spearman": float(pd.Series(y_all).corr(pd.Series(p_raw),
                                                   method="spearman")),
    "pooled_bias": float(np.mean(p_raw - y_all)),
    "pooled_r2_cal": float(r2_score(y_all, p_cal)),
    "pooled_rmse_cal": float(np.sqrt(mean_squared_error(y_all, p_cal))),
    "pooled_mae_cal": float(mean_absolute_error(y_all, p_cal)),
}
print("\n===== POOLED LEAVE-ONE-B-OUT, PRETRAINED MEGNet ZERO-SHOT =====")
print(f"families: {pooled['n_families']}, rows: {pooled['n_rows']}")
print(f"RAW (MP scale):        R2={pooled['pooled_r2_raw']:.3f} "
      f"RMSE={pooled['pooled_rmse_raw']:.3f} MAE={pooled['pooled_mae_raw']:.3f} "
      f"bias={pooled['pooled_bias']:+.3f}")
print(f"Spearman (rank):       {pooled['pooled_spearman']:.3f}")
print(f"CALIBRATED (Castelli scale): R2={pooled['pooled_r2_cal']:.3f} "
      f"RMSE={pooled['pooled_rmse_cal']:.3f} MAE={pooled['pooled_mae_cal']:.3f}")
print("Comparison (pooled leave-one-B-out XGBoost): Magpie-only 0.086, "
      "site-aware 0.733, no-B-info 0.683 (Stages 4 / 3b / 3c).")
print("CONTAMINATION: ~6.2% of Castelli rows share a reduced formula with "
      "MP-2018 (39.8% of oxides), so these zero-shot numbers are an UPPER "
      "BOUND for a clean pretrained model. See the Stage 9b report.")
print(f"total cell time: {(time.time()-T0)/60:.1f} min")

# --- 7. GATE 9b ---
expected_fams = 3 if SMOKE else 52
expected_rows = 364 * expected_fams
gate_ok = (
    len(results_df) == expected_fams
    and len(pred_all) == expected_rows
    and len(calib_df) == CALIB_N
    and pred_all[["y_true", "y_pred_raw", "y_pred_cal"]].notna().all().all()
    and set(results_df["family_B"]) == set(families)
)
print(f"\nGATE 9b: {'PASS' if gate_ok else 'FAIL'} "
      f"({len(results_df)}/{expected_fams} families, "
      f"{len(pred_all)}/{expected_rows} predictions, "
      f"calib pool {len(calib_df)}/{CALIB_N})")
print("Files saved:")
print(" ", results_path)
print(" ", pred_path)
print(" ", calib_path)
if not gate_ok:
    raise AssertionError("GATE 9b FAILED — see counts above. Do not use these numbers.")
