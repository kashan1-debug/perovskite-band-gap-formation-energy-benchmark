#!/usr/bin/env python3
"""
Perovskite band gap + stability under family shift — AUDITED pipeline v2.

WHAT THIS IS, IN PLAIN LANGUAGE
--------------------------------
A perovskite is a crystal ABX3: a big A cation, a smaller B metal in an
octahedron of six X anions. This project asks: can a model that only
sees the COMPOSITION (which elements, in which sites) predict
  (1) formation energy per atom — how stable the crystal is, and
  (2) whether it has a band gap at all (metal vs non-metal), and if so
      how big the gap is?
and, more importantly, does it still work when a whole family of
materials (a B metal, or an X anion type) was never in the training
data? That second question is the point of the paper: random splits
flatter the model, family-out splits tell the truth.

WHAT THE AUDIT (2026-11-22, run in this VM — see AUDIT_REPORT.md)
CHANGED vs v1
--------------
1. PRIMARY DATASET IS CASTELLI ALONE (18,928 rows). MatBench has only
   ['structure','e_form'] and NO formula column. Its reduced formula
   (e.g. 'AgAsN2O') collapses A/B-swapped structures ('AgAsON2' vs
   'AsAgON2'), so merging on reduced_formula is WRONG: 9,282 of 9,646
   reduced formulas are shared by >1 ordered formula, and the merge
   silently assigns the wrong band gap. Castelli already has e_form,
   structure, the ORDERED formula and 'gap gllbsc' — no merge needed.
   (MatBench e_form values are identical, same row order, verified.)
2. A/B/X COME FROM STRUCTURE SITE ORDER, not from an atomic-radius
   rule on the reduced formula. Site 0 = A, site 1 = B, sites 2-4 = X.
   Verified on all 18,928 rows (only "mismatches" are the 364 A==B
   diagonal rows, where the formula itself cannot split Bi2 into 1+1).
   The old radius rule agreed with site order on only 14.0% of A-sites.
   Coordinates are relaxed/distorted, so species ORDER — not exact
   (0,0,0)/(0.5,0.5,0.5) positions — is the invariant.
3. THE DATASET IS A FULL FACTORIAL: 52 A elements x 52 B elements x
   7 X anion types = 18,928, every combination exactly once. So
   leave-one-B-out is 52 perfectly balanced families of 364 rows, and
   leave-one-X-out is 7 families of 2,704. A_family 'other' (14,560
   rows) is too big/coarse to leave out as one family — use B and X.
4. MAGPIE IS SITE-BLIND — AND THERE IS A HARD CEILING ON WHAT ANY
   COMPOSITION-ONLY MODEL CAN DO. Magpie statistics are computed over
   the whole composition, so A/B-swapped rows have IDENTICAL Magpie
   features but different targets: 18,564 of 18,928 Magpie rows are
   exact duplicates, and 9,282 of 9,646 reduced-formula groups have
   more than one row. The pooled within-group variance of
   e_form_per_atom is 0.0101 vs 0.0221 total, so a model that knew
   each composition's TRUE group mean could reach at most
   R2 = 1 - 0.0101/0.0221 = 0.54. In practice the ceiling is lower
   (~0.46 by an independent single-fold diagnostic in Colab, using
   mean within-group std 0.109 eV/atom vs 0.149 overall), because
   group means must be estimated from training rows. Verified here,
   full data, 5-fold, seed 0, Magpie-only: Ridge R2 = 0.385 (it
   shrinks toward the group mean); DEFAULT XGBoost (depth 6,
   300 trees) R2 = -0.272 and DEFAULT RandomForest R2 = -0.084 —
   both worse than predicting the mean, because untuned trees
   memorise training y for a composition and then miss the swapped
   test row by the full within-group difference. TUNED trees fix
   it: XGBoost (depth 3, 100 trees, lr 0.05, subsample/colsample
   0.8, reg_lambda 5) R2 = 0.440, RMSE 0.111 — essentially AT the
   practical ceiling; RandomForest (depth 8, min_samples_leaf 5)
   R2 = 0.372. This script uses ONLY the tuned configs.
   The structural fix is SITE-AWARE features: one-hot A_site /
   B_site / X_anion (+ tolerance/octahedral factors). On full
   site-aware X, 5-fold, seed 0: Ridge R2 = 0.817, XGBoost
   (DEFAULT, 100 trees, depth 6) R2 = 0.899, XGBoost (TUNED, the
   config shipped in this script) R2 = 0.763. The tuned config
   underfits the one-hot signal — it is kept anyway because it is
   the only config that does not catastrophically fail on
   Magpie-only features (default: -0.272, tuned: 0.440). Do NOT
   quote 0.90 as this script's result: v2 as shipped uses tuned
   models, so its site-aware random-CV headline is Ridge 0.82 /
   tuned XGBoost 0.76. One-hot alone (default XGB): R2 = 0.83.
   WARNING: all of these are interpolation inside a closed
   factorial grid — every A/B/X level appears in training.
   Family-out is the real test and will be much lower; report both,
   never only random CV.
5. BAND GAP IS 96.1% ZERO (18,193 of 18,928). Only 735 rows have a
   non-zero gap (mean 1.82, median 1.40, max 7.0 eV; only 77 in the
   1.3-1.6 eV solar range). Plain band-gap regression on all rows is
   zero-inflated and misleading (XGBoost R2 = -0.20, worse than
   predicting 0 everywhere, R2 = -0.02). So v2 treats band gap as:
   (a) metal vs non-metal CLASSIFICATION (gap>0; XGBoost ROC-AUC
   0.925, accuracy 0.944 vs 96.1% majority baseline — report ROC-AUC,
   not accuracy), and (b) regression ONLY on the 735 non-zero rows
   (XGBoost R2 = 0.41, RMSE 1.15 eV — provisional, n=735).
6. e_form_per_atom IS MOSTLY POSITIVE (mean 0.294, median 0.272,
   min -0.128, max 1.032 eV/atom; only 74 of 18,928 rows are negative).
   Do NOT assume a negative formation energy, and do NOT call
   e_form_per_atom < 0 "stable" without an energy-above-hull check
   (this dataset has no hull energies — that is a stated limitation).
7. PARQUET CANNOT HOLD STRUCTURE OBJECTS. featurize_dataframe returns
   the original columns (Structure/Composition objects) plus Magpie
   features, and to_parquet on that raises ArrowInvalid. Every save
   in v2 drops ['structure','composition'] first. X is Magpie features
   ONLY — columns in the featurised frame that are NOT in the original
   frame (~132 columns). NEVER include Castelli DFT columns as
   features: vbm/cbm leak the gap directly (gap = cbm - vbm), and
   fermi level/width, mu_b and e_form are DFT outputs too.
8. RANDOM FOREST IS THE BOTTLENECK (Colab Cell 10: RF seed 0 still
   running after ~8 min). v2 uses RF with 100 trees (was 200) and the
   full CV is estimated at 1+ hour in Colab. Run stages separately.

USAGE
-----
  python perovskite_benchmark_v2.py --stage data --out-dir results
  python perovskite_benchmark_v2.py --stage features --out-dir results
  python perovskite_benchmark_v2.py --stage smoke --out-dir results
  python perovskite_benchmark_v2.py --stage cv --out-dir results --resume
  python perovskite_benchmark_v2.py --stage family --out-dir results --resume
  python perovskite_benchmark_v2.py --stage conformal --out-dir results
  python perovskite_benchmark_v2.py --stage figures --out-dir results
  python perovskite_benchmark_v2.py --stage all --out-dir results --resume
  # Colab: put --out-dir on Drive, e.g. /content/drive/MyDrive/perovskite/results
  # Every stage appends + flushes per fold/family, and --resume skips
  # finished work, so a dead runtime loses at most one fold.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

log = logging.getLogger("perovskite_v2")

SEEDS = [0, 1, 2]
FOLDS = 5
N_TREES_RF = 100  # was 200 in v1; RF is the bottleneck (see header, point 8)

# DFT columns that must NEVER be features (leakage). 'gap is direct'
# is also a DFT output. e_form / band_gap are targets.
DFT_COLUMNS = ["fermi level", "fermi width", "e_form", "mu_b", "vbm",
               "cbm", "gap is direct", "band_gap", "e_form_per_atom"]

A_FAMILIES = {
    "alkali": {"Li", "Na", "K", "Rb", "Cs"},
    "alkaline-earth": {"Mg", "Ca", "Sr", "Ba"},
    "lanthanide": {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
                   "Ho", "Er", "Tm", "Yb", "Lu", "Sc", "Y"},
}

REGRESSORS = {
    "Mean": lambda s: DummyRegressor(strategy="mean"),
    "Ridge": lambda s: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    # TUNED RandomForest — verified here, full data, 5-fold, seed 0:
    # depth 8 + min_samples_leaf 5 gives R2 0.372. The DEFAULT
    # (unbounded depth) scored R2 -0.084 in Colab. Still slow:
    # ~2 min per 5-fold here, ~8+ min per seed/split in Colab.
    "Random Forest": lambda s: RandomForestRegressor(
        n_estimators=N_TREES_RF, max_depth=8, min_samples_leaf=5,
        n_jobs=-1, random_state=s),
    # TUNED XGBoost — the audited config (do NOT use the defaults).
    # Default (depth 6, 300 trees) scores R2 -0.272 on this data,
    # worse than predicting the mean: it memorises training y for a
    # composition and then misses the A/B-swapped test row by the
    # full within-group difference. Tuned (depth 3, lr 0.05,
    # subsample/colsample 0.8, lambda 5) scores R2 0.440 — essentially
    # at the composition-only ceiling (~0.46-0.54, see header).
    "XGBoost": lambda s: XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
        tree_method="hist", n_jobs=-1, random_state=s),
}
CLASSIFIERS = {
    "Majority": lambda s: DummyClassifier(strategy="most_frequent"),
    "Logistic": lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)),
    "XGBoost": lambda s: XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
        tree_method="hist", n_jobs=-1, random_state=s),
}


# ------------------------------------------------------------------ utils
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def reg_metrics(y, p):
    return {"R2": float(r2_score(y, p)), "RMSE": rmse(y, p),
            "MAE": float(mean_absolute_error(y, p))}


def cls_metrics(y, proba):
    pred = (proba > 0.5).astype(int)
    return {"ROC_AUC": float(roc_auc_score(y, proba)),
            "accuracy": float(accuracy_score(y, pred)),
            "majority_acc": float(max(y.mean(), 1 - y.mean()))}


def append_row(csv_path: Path, row: dict):
    """Append ONE result row and flush immediately (Colab-proof)."""
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def save_scalar_parquet(df: pd.DataFrame, path: Path):
    """Save a dataframe to parquet, dropping any pymatgen object
    columns first — pyarrow cannot serialise Structure/Composition
    objects (ArrowInvalid). Raises if an object column remains."""
    drop = [c for c in ("structure", "composition") if c in df.columns]
    out = df.drop(columns=drop)
    bad = [c for c in out.columns if out[c].dtype == object
           and c not in ("formula", "reduced_formula", "A_site", "B_site",
                         "X_anion", "A_family")]
    if bad:
        raise ValueError(f"refusing to parquet object columns: {bad}")
    out.to_parquet(path, index=False)
    log.info("wrote %s (%d rows x %d cols)", path, len(out), out.shape[1])


def grouped_folds(groups, seed, k):
    """GroupKFold-like folds, balanced by group size. Groups are the
    REDUCED formula here: rows sharing a composition (A/B-swapped
    polymorphs) must not be split across train/test, because they
    share Magpie features exactly."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    sizes, assign = np.zeros(k), {}
    for j in rng.permutation(len(uniq)):
        f = int(np.argmin(sizes))
        assign[uniq[j]] = f
        sizes[f] += counts[j]
    fold_of = np.array([assign[g] for g in groups])
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0])
            for f in range(k)]


# ------------------------------------------------------------------ data
def load_castelli(out_dir: Path) -> pd.DataFrame:
    """Load Castelli perovskites — the ONLY dataset v2 uses.
    Returns the frame with formula (ordered), band_gap, e_form,
    e_form_per_atom, nsites, composition, reduced_formula.
    Asserts the facts the audit verified, so a changed dataset fails
    loudly instead of silently producing wrong numbers."""
    from matminer.datasets import load_dataset

    df = load_dataset("castelli_perovskites")
    log.info("castelli shape=%s columns=%s", df.shape, list(df.columns))
    assert df.shape == (18928, 10), f"unexpected shape {df.shape}"
    assert "gap gllbsc" in df.columns, "band-gap column 'gap gllbsc' missing"
    df = df.rename(columns={"gap gllbsc": "band_gap"})
    assert int(df["band_gap"].isna().sum()) == 0, "band gaps are missing"
    assert df["formula"].nunique() == 18928, "ordered formulas not unique"
    df["nsites"] = [s.num_sites for s in df["structure"]]
    assert (df["nsites"] == 5).all(), "not all structures have 5 sites"
    df["composition"] = [s.composition for s in df["structure"]]
    df["reduced_formula"] = [s.composition.reduced_formula
                             for s in df["structure"]]
    df["e_form_per_atom"] = df["e_form"] / df["nsites"]
    df["is_metal"] = (df["band_gap"] == 0).astype(int)
    n_zero = int((df["band_gap"] == 0).sum())
    log.info("band_gap == 0 (metals): %d / %d (%.1f%%); non-zero: %d",
             n_zero, len(df), 100 * n_zero / len(df), len(df) - n_zero)
    log.info("e_form_per_atom: mean %.3f median %.3f min %.3f max %.3f; "
             "negative (vs elements): %d rows",
             df["e_form_per_atom"].mean(), df["e_form_per_atom"].median(),
             df["e_form_per_atom"].min(), df["e_form_per_atom"].max(),
             int((df["e_form_per_atom"] < 0).sum()))
    save_scalar_parquet(df, out_dir / "castelli_scalar.parquet")
    return df


def parse_site_order(structure):
    """A/B/X from STRUCTURE SITE ORDER: site 0 = A, site 1 = B,
    sites 2-4 = X. Returns (A, B, X_label) where X_label keeps
    stoichiometry, e.g. 'O3', 'O2F', 'OFN' (elements sorted
    alphabetically, count shown only when >1 — so 'O2F', not 'FO2').
    The old atomic-radius rule on the reduced formula agreed with
    this on only 14% of rows and must never be used."""
    sp = [str(site.specie) for site in structure.sites]
    if len(sp) != 5:
        return None, None, None
    x_amt: dict[str, int] = {}
    for e in sp[2:]:
        x_amt[e] = x_amt.get(e, 0) + 1
    x_label = "".join(f"{e}{n if n > 1 else ''}"
                      for e, n in sorted(x_amt.items()))
    return sp[0], sp[1], x_label


def a_family(a):
    if a is None:
        return None
    for fam, els in A_FAMILIES.items():
        if a in els:
            return fam
    return "other"


def build_families(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Add A_site / B_site / X_anion / A_family by site order and
    verify the factorial: 52 A x 52 B x 7 X = 18,928, every B exactly
    364 rows, every X exactly 2,704."""
    parsed = [parse_site_order(s) for s in df["structure"]]
    df["A_site"], df["B_site"], df["X_anion"] = zip(*parsed)
    assert df["A_site"].isna().sum() == 0, "unparseable rows"
    df["A_family"] = df["A_site"].apply(a_family)
    log.info("B_site: %d unique, counts %s",
             df["B_site"].nunique(),
             sorted(df["B_site"].value_counts().unique().tolist()))
    log.info("X_anion counts:\n%s", df["X_anion"].value_counts().to_string())
    log.info("A_family counts:\n%s", df["A_family"].value_counts().to_string())
    assert df["B_site"].nunique() == 52
    assert set(df["B_site"].value_counts().unique()) == {364}
    assert df["X_anion"].nunique() == 7
    assert set(df["X_anion"].value_counts().unique()) == {2704}
    save_scalar_parquet(df, out_dir / "merged_with_families.parquet")
    return df


# ------------------------------------------------------------------ features
def ionic_radius(element: str, kind: str) -> float:
    """Shannon-type ionic radius (Angstrom) from pymatgen's
    Element.ionic_radii, with a DOCUMENTED charge convention — this
    dataset does not assign oxidation states, so no choice here is
    exact; these are approximate diagnostics, not precise criteria.
      A-site: smallest positive charge (alkali +1, alkaline earth +2,
              transition metals their lowest positive state).
      B-site: largest positive charge (e.g. Ti +4, Pb +4).
      X-site: most negative charge (O/S -2, N -3, F/Cl/Br/I -1).
    Returns NaN when pymatgen has no radius for that charge."""
    from pymatgen.core import Element

    radii = Element(element).ionic_radii
    if not radii:
        return float("nan")
    pos = [c for c in radii if c > 0]
    neg = [c for c in radii if c < 0]
    if kind == "A":
        charge = min(pos) if pos else None
    elif kind == "B":
        charge = max(pos) if pos else None
    else:
        charge = min(neg) if neg else None
    return float(radii[charge]) if charge is not None else float("nan")


def add_tolerance(df: pd.DataFrame) -> pd.DataFrame:
    """Goldschmidt tolerance factor t = (rA + rX) / (sqrt(2) (rB + rX))
    and octahedral factor mu = rB / rX, from the ionic radii above.
    Coverage is reported; rows without a radius get NaN (filled with
    the median later — the one-hot site features carry most of the
    signal anyway, see the audit)."""
    # X_anion is a label like 'O2F'; for mixed anions the radius uses
    # the MAJORITY element (e.g. O in O2F) — a documented approximation.
    import re

    t_vals, mu_vals = [], []
    for a, b, x_label in zip(df["A_site"], df["B_site"], df["X_anion"]):
        # majority X element: the one with the largest count in the label
        parts = re.findall(r"([A-Z][a-z]?)(\d*)", x_label)
        x_el = max(parts, key=lambda p: int(p[1]) if p[1] else 1)[0]
        ra, rb, rx = (ionic_radius(a, "A"), ionic_radius(b, "B"),
                      ionic_radius(x_el, "X"))
        if any(math.isnan(v) for v in (ra, rb, rx)):
            t_vals.append(float("nan"))
            mu_vals.append(float("nan"))
        else:
            t_vals.append((ra + rx) / (math.sqrt(2.0) * (rb + rx)))
            mu_vals.append(rb / rx)
    df["tolerance_factor"] = t_vals
    df["octahedral_factor"] = mu_vals
    log.info("tolerance_factor NaN: %d / %d; octahedral_factor NaN: %d",
             int(df["tolerance_factor"].isna().sum()), len(df),
             int(df["octahedral_factor"].isna().sum()))
    return df


def featurise(df: pd.DataFrame, out_dir: Path):
    """Magpie (132 cols) + site-aware one-hots (52 A + 52 B + 7 X)
    + tolerance/octahedral factors. X is Magpie features ONLY —
    columns in the featurised frame that are NOT in the original
    frame — plus the one-hots and t/mu. DFT columns are never
    features (asserted). Magpie runs in 2,000-row chunks and the
    features parquet is saved with Structure/Composition dropped."""
    from matminer.featurizers.composition import ElementProperty

    feat_path = out_dir / "features_magpie.parquet"
    cached = pd.read_parquet(feat_path) if feat_path.exists() else None
    if cached is not None and len(cached) == len(df):
        log.info("features_magpie.parquet exists — loading, not re-featurising")
        Xdf = cached
    else:
        if cached is not None:
            log.info("cached features have %d rows, need %d — re-featurising",
                     len(cached), len(df))
        feat = ElementProperty.from_preset("magpie")
        chunks, magpie_cols = [], None
        t0 = time.time()
        for i in range(0, len(df), 2000):
            part = feat.featurize_dataframe(
                df.iloc[i:i + 2000].copy(), col_id="composition",
                ignore_errors=True)
            # Magpie features ONLY: drop every original column.
            magpie_cols = [c for c in part.columns if c not in df.columns]
            chunks.append(part[magpie_cols])
            log.info("Magpie chunk %d-%d (%.0fs)",
                     i, i + len(part) - 1, time.time() - t0)
        Xdf = pd.concat(chunks, ignore_index=True)
        Xdf.to_parquet(feat_path, index=False)
        log.info("saved %s (%d rows x %d Magpie cols)",
                 feat_path, len(Xdf), Xdf.shape[1])
    assert len(Xdf) == len(df), "Magpie rows do not match df rows"
    log.info("Magpie NaN cells: %d", int(Xdf.isna().sum().sum()))
    # one-hot site identities — the audit's fix for Magpie site-blindness
    oh = pd.get_dummies(df[["A_site", "B_site", "X_anion"]],
                        columns=["A_site", "B_site", "X_anion"],
                        dtype=np.float32)
    feat_names = (list(Xdf.columns) + list(oh.columns)
                  + ["tolerance_factor", "octahedral_factor"])
    X = np.hstack([Xdf.fillna(0).values.astype(np.float32),
                   oh.values.astype(np.float32),
                   df[["tolerance_factor", "octahedral_factor"]]
                   .fillna(df[["tolerance_factor", "octahedral_factor"]]
                           .median()).values.astype(np.float32)])
    forbidden = [c for c in feat_names if c in DFT_COLUMNS
                 or c in ("formula", "structure", "composition")]
    assert not forbidden, f"LEAKAGE: forbidden columns in X: {forbidden}"
    log.info("X shape: %s (%d Magpie + %d one-hot + 2 physics)",
             X.shape, Xdf.shape[1], oh.shape[1])
    np.save(out_dir / "X_all.npy", X)
    pd.Series(feat_names).to_csv(out_dir / "feature_names.csv",
                                 index=False, header=["feature"])
    return X, feat_names

# ------------------------------------------------------------------ modelling
def load_inputs(out_dir: Path):
    """Load the cached data + features built by the data/features
    stages. Returns (df, X)."""
    df = pd.read_parquet(out_dir / "merged_with_families.parquet")
    X = np.load(out_dir / "X_all.npy")
    assert len(df) == len(X)
    return df, X


def run_cv(df, X, out_dir: Path, seeds=SEEDS, folds=FOLDS, resume=False):
    """Random + grouped-by-REDUCED-formula 5-fold CV, seeds 0,1,2.
    Targets:
      * e_form_per_atom — regression, all 4 models.
      * is_metal (gap==0) — classification, ROC-AUC (NOT accuracy:
        the 96.1% majority baseline makes accuracy meaningless).
      * band_gap on non-zero rows only (n=735) — XGBoost + Ridge,
        1 seed, flagged provisional in the CSV.
    Grouped split uses the REDUCED formula: rows sharing a composition
    share Magpie features exactly, so they must stay together.
    (Grouping by ORDERED formula would be trivial — every group has
    n=1 — and would not test anything.)"""
    y_e = df["e_form_per_atom"].values
    y_m = df["is_metal"].values
    nz = np.where(df["band_gap"].values > 0)[0]
    groups = df["reduced_formula"].values
    log.info("CV: n=%d, groups(reduced formula)=%d, non-zero gap n=%d",
             len(df), len(set(groups)), len(nz))

    # --- e_form_per_atom regression ---
    csv_e = out_dir / "results_cv_e_form_per_atom.csv"
    done_e = set()
    if resume and csv_e.exists():
        done_e = set(map(tuple, pd.read_csv(csv_e)
                         [["Split", "Model", "Seed"]].values.tolist()))
    for seed in seeds:
        for split, fl in [
                ("random", list(KFold(folds, shuffle=True,
                                      random_state=seed).split(X))),
                ("grouped_reduced_formula",
                 grouped_folds(groups, seed, folds))]:
            for name, factory in REGRESSORS.items():
                if (split, name, seed) in done_e:
                    continue
                yp = np.full(len(y_e), np.nan)
                for tr, te in fl:
                    m = factory(seed)
                    m.fit(X[tr], y_e[tr])
                    yp[te] = m.predict(X[te])
                append_row(csv_e, {"Split": split, "Model": name,
                                   "Seed": seed, "Target": "e_form_per_atom",
                                   "N": len(y_e), **reg_metrics(y_e, yp)})
                log.info("cv e_form %s %s seed=%d R2=%.3f",
                         split, name, seed, r2_score(y_e, yp))

    # --- metal vs non-metal classification ---
    csv_m = out_dir / "results_cv_is_metal.csv"
    done_m = set()
    if resume and csv_m.exists():
        done_m = set(map(tuple, pd.read_csv(csv_m)
                         [["Split", "Model", "Seed"]].values.tolist()))
    for seed in seeds:
        for split, fl in [
                ("random", list(StratifiedKFold(folds, shuffle=True,
                                                random_state=seed)
                                .split(X, y_m)))]:
            for name, factory in CLASSIFIERS.items():
                if (split, name, seed) in done_m:
                    continue
                proba = np.full(len(y_m), np.nan)
                for tr, te in fl:
                    m = factory(seed)
                    m.fit(X[tr], y_m[tr])
                    proba[te] = m.predict_proba(X[te])[:, 1]
                append_row(csv_m, {"Split": split, "Model": name,
                                   "Seed": seed, "Target": "is_metal",
                                   "N": len(y_m), **cls_metrics(y_m, proba)})
                log.info("cv is_metal %s %s seed=%d ROC-AUC=%.3f",
                         split, name, seed, roc_auc_score(y_m, proba))

    # --- non-zero gap regression (n=735, provisional) ---
    csv_g = out_dir / "results_cv_band_gap_nonzero.csv"
    if not (resume and csv_g.exists()):
        X_nz, y_nz = X[nz], df["band_gap"].values[nz]
        for name, factory in [("Ridge", REGRESSORS["Ridge"]),
                              ("XGBoost", REGRESSORS["XGBoost"])]:
            yp = np.full(len(y_nz), np.nan)
            for tr, te in KFold(folds, shuffle=True,
                                random_state=0).split(X_nz):
                m = factory(0)
                m.fit(X_nz[tr], y_nz[tr])
                yp[te] = m.predict(X_nz[te])
            append_row(csv_g, {"Split": "random", "Model": name,
                               "Seed": 0, "Target": "band_gap_nonzero",
                               "N": len(y_nz), "NOTE": "n=735 provisional",
                               **reg_metrics(y_nz, yp)})
            log.info("cv band_gap_nonzero %s R2=%.3f", name,
                     r2_score(y_nz, yp))


def run_family(df, X, out_dir: Path, resume=False):
    """Leave-one-family-out for e_form_per_atom (Ridge + XGBoost only
    — RF is too slow for 52+7 families) and for is_metal (XGBoost).
    Families: B_site (52 x 364 rows) and X_anion (7 x 2,704 rows).
    A_family: only the three NAMED families (alkali 1,820,
    alkaline-earth 1,456, lanthanide 1,092) — 'other' (14,560 rows,
    77% of the data) is too big and too mixed to leave out as one
    family, and leaving it out would leave almost nothing to train on.
    WARNING: one-hot features for a held-out level are all zero in
    the test rows, so family-out tests whether Magpie + physics
    features transfer — that is the point."""
    y_e = df["e_form_per_atom"].values
    y_m = df["is_metal"].values
    specs = [("B_site", df["B_site"].values),
             ("X_anion", df["X_anion"].values),
             ("A_family_named", df["A_family"].values)]
    for col, fam in specs:
        csv_path = out_dir / f"results_family_{col}_e_form.csv"
        done = set()
        if resume and csv_path.exists():
            done = set(pd.read_csv(csv_path)["Family"].unique().tolist())
        counts = pd.Series(fam).value_counts()
        fams = [f for f in counts[counts >= 200].index
                if not (col == "A_family_named" and f == "other")]
        for f in sorted(fams):
            if f in done:
                continue
            te, tr = fam == f, fam != f
            for name, factory in [("Ridge", REGRESSORS["Ridge"]),
                                  ("XGBoost", REGRESSORS["XGBoost"])]:
                m = factory(0)
                m.fit(X[tr], y_e[tr])
                yp = m.predict(X[te])
                append_row(csv_path, {"Family": f, "Model": name,
                                      "N": int(te.sum()),
                                      "Target": "e_form_per_atom",
                                      **reg_metrics(y_e[te], yp)})
            log.info("family-out %s=%s N=%d done", col, f, int(te.sum()))
        # metal classification, XGBoost only
        csv_c = out_dir / f"results_family_{col}_is_metal.csv"
        done_c = set()
        if resume and csv_c.exists():
            done_c = set(pd.read_csv(csv_c)["Family"].unique().tolist())
        for f in sorted(fams):
            if f in done_c:
                continue
            te, tr = fam == f, fam != f
            m = CLASSIFIERS["XGBoost"](0)
            m.fit(X[tr], y_m[tr])
            proba = m.predict_proba(X[te])[:, 1]
            append_row(csv_c, {"Family": f, "Model": "XGBoost",
                               "N": int(te.sum()), "Target": "is_metal",
                               **cls_metrics(y_m[te], proba)})
            log.info("family-out-cls %s=%s done", col, f)


def run_conformal(df, X, out_dir: Path, seed=0, alpha=0.10):
    """Split conformal (90% nominal) for e_form_per_atom, XGBoost,
    on a RANDOM 60/20/20 split — and, separately, with the test set
    taken as one held-out B-site family (the honest OOD check).
    Reports coverage and mean interval width for both."""
    y = df["e_form_per_atom"].values
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_tr, n_cal = int(0.6 * len(y)), int(0.2 * len(y))
    tr, cal, te = perm[:n_tr], perm[n_tr:n_tr + n_cal], perm[n_tr + n_cal:]
    rows = []
    for label, te_idx in [("random", te),
                          ("family_out_B_Te",
                           np.where(df["B_site"].values == "Te")[0])]:
        m = REGRESSORS["XGBoost"](seed)
        m.fit(X[tr], y[tr])
        resid = np.abs(y[cal] - m.predict(X[cal]))
        q = float(np.quantile(
            resid, math.ceil((len(resid) + 1) * (1 - alpha)) / len(resid)))
        pred = m.predict(X[te_idx])
        covered = np.abs(y[te_idx] - pred) <= q
        rows.append({"Split": label, "Target": "e_form_per_atom",
                     "Model": "XGBoost", "nominal_coverage": 1 - alpha,
                     "coverage": float(covered.mean()),
                     "mean_width": float(2 * q), "q": q,
                     "N_test": int(len(te_idx))})
        log.info("conformal %s: coverage=%.3f width=%.3f (nominal %.2f)",
                 label, covered.mean(), 2 * q, 1 - alpha)
    pd.DataFrame(rows).to_csv(out_dir / "results_conformal.csv", index=False)


# ------------------------------------------------------------------ figures
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    log.info("wrote %s", path)


def run_figures(out_dir: Path):
    """Three figures, all built from the CSVs (no OOF storage needed):
    1. CV R2 by model and split (e_form). 2. Family-out R2 by B-site
    (XGBoost). 3. Conformal coverage, random vs family-out."""
    cv = pd.read_csv(out_dir / "results_cv_e_form_per_atom.csv")
    piv = cv.groupby(["Split", "Model"])["R2"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv.plot(kind="bar", ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("R2 (mean over seeds)")
    ax.set_title("e_form_per_atom: random vs grouped CV")
    ax.legend(title="Model")
    save_fig(out_dir / "fig_cv_r2.png")

    fam = pd.read_csv(out_dir / "results_family_B_site_e_form.csv")
    xgb = fam[fam["Model"] == "XGBoost"].sort_values("R2")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(xgb["Family"], xgb["R2"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("R2 on held-out B-site (XGBoost)")
    ax.set_title("Leave-one-B-site-out: e_form_per_atom")
    save_fig(out_dir / "fig_family_bsite.png")

    conf = pd.read_csv(out_dir / "results_conformal.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(conf["Split"], conf["coverage"])
    ax.axhline(0.90, color="r", ls="--", label="nominal 0.90")
    ax.set_ylabel("Coverage")
    ax.set_title("Split conformal coverage (e_form_per_atom)")
    ax.legend()
    save_fig(out_dir / "fig_conformal.png")


def run_smoke(df, X, out_dir: Path):
    """Fast sanity check: a RANDOM 2,000-row sample (not the first
    2,000 rows — row order is not a sampling design), Ridge + XGBoost,
    5-fold, e_form_per_atom. Pass = finite R2/RMSE and a row written."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(2000, len(df)), replace=False)
    y = df["e_form_per_atom"].values[idx]
    Xs = X[idx]
    for name, factory in [("Ridge", REGRESSORS["Ridge"]),
                          ("XGBoost", REGRESSORS["XGBoost"])]:
        yp = np.full(len(y), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Xs):
            m = factory(0)
            m.fit(Xs[tr], y[tr])
            yp[te] = m.predict(Xs[te])
        append_row(out_dir / "results_smoke.csv",
                   {"Split": "smoke_random2000", "Target": "e_form_per_atom",
                    "Model": name, "Seed": 0, "N": len(y),
                    **reg_metrics(y, yp)})
        log.info("smoke %s: %s", name, reg_metrics(y, yp))


# ------------------------------------------------------------------ CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Perovskite family-shift benchmark v2 (audited)")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--stage", choices=["data", "features", "smoke", "cv",
                                       "family", "conformal", "figures",
                                       "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--folds", type=int, default=FOLDS)
    p.add_argument("--max-n", type=int, default=None,
                   help="random subsample (seed 0) for a quick test run")
    p.add_argument("--resume", action="store_true",
                   help="skip folds / families already in the output CSVs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("versions: python %s, sklearn %s, xgboost %s, matminer %s, "
             "pymatgen %s", sys.version.split()[0],
             __import__("sklearn").__version__,
             __import__("xgboost").__version__,
             __import__("matminer").__version__,
             pkg_version("pymatgen"))
    log.info("out_dir=%s stage=%s resume=%s max_n=%s",
             args.out_dir, args.stage, args.resume, args.max_n)

    df = X = None
    if args.stage in ("data", "features", "smoke", "cv", "family",
                      "conformal", "all"):
        df = load_castelli(args.out_dir)
        df = build_families(df, args.out_dir)
        df = add_tolerance(df)
        if args.max_n is not None:
            rng = np.random.default_rng(0)
            df = df.iloc[rng.choice(len(df), size=args.max_n,
                                    replace=False)].reset_index(drop=True)
            log.info("subsampled to %d rows (random, seed 0)", len(df))
        save_scalar_parquet(df, args.out_dir / "merged_with_families.parquet")
    if args.stage in ("features", "smoke", "cv", "family", "conformal", "all"):
        if df is None:
            df, X = load_inputs(args.out_dir)
        else:
            X, _ = featurise(df, args.out_dir)
    if args.stage in ("smoke", "all"):
        run_smoke(df, X, args.out_dir)
    if args.stage in ("cv", "all"):
        run_cv(df, X, args.out_dir, seeds=args.seeds, folds=args.folds,
               resume=args.resume)
    if args.stage in ("family", "all"):
        run_family(df, X, args.out_dir, resume=args.resume)
    if args.stage in ("conformal", "all"):
        run_conformal(df, X, args.out_dir)
    if args.stage in ("figures", "all"):
        run_figures(args.out_dir)
    log.info("done stage=%s in %.1f min", args.stage, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
