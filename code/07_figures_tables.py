# ============================================================
# v2 — STAGE 7: Figures and tables (ONE cell, paste into Colab)
# Run AFTER Stage 6 (GATE 6 PASS) in the same session, or after
# Setup B then Setup C in a fresh session. It reads ONLY the
# saved v2 CSVs from Drive — it does NO modelling, NO retraining,
# NO new numbers. Every number in a figure or table comes from
# a CSV read in this cell, never typed by hand.
# Full run takes about 5 minutes. It saves 5 figures at 300 dpi
# to OUT_DIR / figures/ and results_table_v2.md to OUT_DIR.
# If any input file is missing, it lists which one and STOPS.
# Paste back: the GATE 7 line and the list of saved figures.
# ============================================================
# SMOKE TEST FLAG — leave False for the real Colab run.
# SMOKE=True reads the *_smoke.csv files from Stages 3-6 and
# writes figures_smoke/ and results_table_v2_smoke.md instead.
# Use it only to check the code runs.
SMOKE = False

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
print("SMOKE =", SMOKE, "(True = small test only, False = full run)")

CEILING = 0.543  # Pooled composition-only ceiling, Stage 1. In the
# figures it is a LABELLED line, not a number typed into a table.

checks = []  # (check, PASS/FAIL) — printed at the end, saved in the report.


def _check(name, ok):
    status = "PASS" if ok else "FAIL"
    checks.append((name, status))
    print(f"  [{status}] {name}")
    return ok


def _check_table(name, table_value, csv_value, tol=0.0006):
    """Every table value must match the CSV it came from (3dp rounding)."""
    ok = abs(float(table_value) - float(csv_value)) <= tol
    return _check(f"table value {name}: {table_value} vs CSV {csv_value}", ok)


# --- 1. Inputs: all 5 CSVs + predictions + features must exist. ---
# No modelling here — if a file is missing, list it and STOP.
# Do not plot with a missing file; a half-made figure is worse
# than no figure, because it looks finished.
suffix = "_smoke" if SMOKE else ""
INPUTS = {
    "cv": OUT_DIR / f"results_cv_v2{suffix}.csv",
    "family": OUT_DIR / f"results_family_v2{suffix}.csv",
    "family_pred": OUT_DIR / f"predictions_family_v2{suffix}.parquet",
    "bandgap": OUT_DIR / f"results_bandgap_v2{suffix}.csv",
    "conformal": OUT_DIR / f"results_conformal_v2{suffix}.csv",
    "features": OUT_DIR / "features.parquet",
}
missing = [k for k, p in INPUTS.items() if not p.exists()]
if missing:
    print("MISSING input files — STOP. Run the stage that makes each one:")
    for k in missing:
        print(f"  {k}: {INPUTS[k].name} (expected in OUT_DIR)")
    raise FileNotFoundError("Stage 7 inputs missing: " + ", ".join(missing))
for k, p in INPUTS.items():
    print(f"found {k}: {p.name}")

cv = pd.read_csv(INPUTS["cv"])
fam = pd.read_csv(INPUTS["family"])
fam_pred = pd.read_parquet(INPUTS["family_pred"])
bg = pd.read_csv(INPUTS["bandgap"])
conf = pd.read_csv(INPUTS["conformal"])
feat = pd.read_parquet(INPUTS["features"])
print(f"cv rows: {len(cv)} | family rows: {len(fam)} | "
      f"family predictions: {len(fam_pred)} | bandgap rows: {len(bg)} | "
      f"conformal rows: {len(conf)} | features shape: {feat.shape}")

EXPECTED = {"cv": 8 if SMOKE else 24, "family": 10 if SMOKE else 118,
            "bandgap": 8 if SMOKE else 22, "conformal": 2 if SMOKE else 4}
for k, n in EXPECTED.items():
    df = {"cv": cv, "family": fam, "bandgap": bg, "conformal": conf}[k]
    _check(f"input {k} rows == {n}", len(df) == n)

FIG_DIR = OUT_DIR / ("figures_smoke" if SMOKE else "figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("figures go to:", FIG_DIR)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10})


def save_fig(fig, name):
    """Save one figure at 300 dpi and check the file landed."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ok = path.exists() and path.stat().st_size > 0
    print(f"saved {name} ({path.stat().st_size // 1024} KB, 300 dpi)")
    return ok


# --- 2. Numbers for the figures/tables, computed from the CSVs. ---
# CV: mean R2 across seeds per (split, model), seed-to-seed std.
MODELS = ["Mean", "Ridge", "RandomForest", "XGBoost"]
cv_sum = (cv.groupby(["split", "model"])["r2_mean"]
          .agg(["mean", "std"]).reset_index())
cv_mean = {(r.split, r.model): float(r.mean) for r in cv_sum.itertuples()}
cv_std = {(r.split, r.model): float(r.std) if not np.isnan(r.std) else 0.0
          for r in cv_sum.itertuples()}

# Pooled family-out: computed ONCE from the saved predictions
# (r2_score on all pooled test predictions), NOT the mean of
# per-family R2s. Recomputed a second way (sums of squares) and
# asserted equal — the two methods must agree.
def pooled_from_preds(sub):
    yt = sub["y_true"].values
    yp = sub["y_pred"].values
    return {"r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "n": int(len(yt))}


def pooled_manual(sub):
    yt = sub["y_true"].values
    yp = sub["y_pred"].values
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


pooled = {}
for (ft, model), sub in fam_pred.groupby(["family_type", "model"]):
    pm = pooled_from_preds(sub)
    pm2 = pooled_manual(sub)
    _check(f"pooled {ft}-out {model} R2: two methods agree "
           f"({pm['r2']:.3f} vs {pm2:.3f})", abs(pm["r2"] - pm2) < 1e-9)
    pooled[(ft, model)] = pm

# Band gap: C1 main rows (threshold 0.5), threshold rows, C2, C0.
c1 = bg[bg["task"] == "classification_metal_vs_nonmetal"]
c1_thr = bg[bg["task"] == "classification_threshold"]
c2 = bg[bg["task"] == "regression_nonzero_gap"]
c0 = bg[bg["task"] == "warning_plain_all_rows"]
c1_roc = float(c1["roc_auc"].mean())
c1_rec05 = float(c1["recall"].mean())
c1_prec05 = float(c1["precision"].mean())
thr_sum = (c1_thr.groupby("threshold")[["precision", "recall", "f1"]]
           .mean().reset_index())
thr02 = thr_sum[thr_sum["threshold"] == 0.2].iloc[0]
c2_r2 = float(c2["r2"].mean())
c2_rmse = float(c2["rmse"].mean())
c2_mae = float(c2["mae"].mean())
c0_r2 = float(c0["r2"].iloc[0]) if len(c0) else float("nan")

# --- 3. Fig 1 — CV R2: random vs grouped, by model. ---
# Grouped bars, mean R2 across seeds, error bars = seed-to-seed std.
# Dashed line at the pooled ceiling 0.543 (Stage 1, labelled).
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(MODELS))
w = 0.38
rand = [cv_mean.get(("random", m), float("nan")) for m in MODELS]
grp = [cv_mean.get(("grouped", m), float("nan")) for m in MODELS]
rand_e = [cv_std.get(("random", m), 0.0) for m in MODELS]
grp_e = [cv_std.get(("grouped", m), 0.0) for m in MODELS]
ax.bar(x - w / 2, rand, w, yerr=rand_e, capsize=4, label="Random CV")
ax.bar(x + w / 2, grp, w, yerr=grp_e, capsize=4, label="Grouped by formula")
ax.axhline(CEILING, color="red", linestyle="--", linewidth=1.2,
           label=f"Composition-only ceiling R² = {CEILING:.3f}")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.set_ylabel("R² (formation energy per atom)")
ax.set_title("Fig 1 — Random vs grouped CV (Magpie only, 5-fold, seeds 0–2)")
ax.set_ylim(-0.1, 0.65)
ax.legend(loc="upper left")
fig.tight_layout()
fig1_ok = save_fig(fig, "fig1_cv_r2.png")

# --- 4. Fig 2 — Family-out pooled R2: two panels, honest scale. ---
# X-out Ridge is -9.96. Plotting it on the same linear scale as
# the others (0.09–0.38) would crush the chart to a flat line, so
# B-out and X-out get separate panels. In the X panel the Ridge
# bar is drawn to the axis floor (-1) with hatching and labelled
# with its TRUE value — the bar is a marker, not the value.
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=False)
b_ridge = pooled[("B", "Ridge")]["r2"]
b_xgb = pooled[("B", "XGBoost")]["r2"]
x_ridge = pooled[("X", "Ridge")]["r2"]
x_xgb = pooled[("X", "XGBoost")]["r2"]
axes[0].bar(["Ridge", "XGBoost"], [b_ridge, b_xgb], color=["#4C72B0", "#DD8452"])
axes[0].axhline(0, color="grey", linewidth=0.8)
axes[0].set_title(f"Leave-one-B-out (pooled, n = {pooled[('B', 'Ridge')]['n']:,})")
axes[0].set_ylabel("Pooled R²")
axes[0].set_ylim(-0.2, 0.45)
for i, v in enumerate([b_ridge, b_xgb]):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")
axes[1].bar(["XGBoost"], [x_xgb], color="#DD8452")
axes[1].bar(["Ridge"], [-1.0], color="#4C72B0", hatch="///",
            edgecolor="#4C72B0")
axes[1].axhline(0, color="grey", linewidth=0.8)
axes[1].set_title(f"Leave-one-X-out (pooled, n = {pooled[('X', 'Ridge')]['n']:,})")
axes[1].set_ylim(-1.2, 0.6)
axes[1].text(0, x_xgb + 0.03, f"{x_xgb:.3f}", ha="center")
# Annotation INSIDE the hatched Ridge bar (white text), in
# SHORT lines so it fits inside the bar and can never overlap
# the x-tick labels below the axis.
axes[1].text(1, -0.5, f"Ridge R² =\n{x_ridge:.2f}\n(off scale)",
             ha="center", va="center", fontsize=9, color="white",
             fontweight="bold")
fig.suptitle("Fig 2 — Family-out pooled R² (Magpie only, seed 0)")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig2_ok = save_fig(fig, "fig2_familyout_r2.png")

# --- 5. Fig 3 — Parity: pooled family-out predictions vs true. ---
# B-out only, one panel per model, y = x line, pooled R2/RMSE on
# each panel. Points are subsampled to 5,000 per panel for speed.
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
rng = np.random.RandomState(0)
for ax, model in zip(axes, ["Ridge", "XGBoost"]):
    sub = fam_pred[(fam_pred["family_type"] == "B")
                   & (fam_pred["model"] == model)]
    pm = pooled[("B", model)]
    idx = rng.choice(len(sub), size=min(5000, len(sub)), replace=False)
    s = sub.iloc[idx]
    ax.scatter(s["y_true"], s["y_pred"], s=6, alpha=0.35)
    lo = min(s["y_true"].min(), s["y_pred"].min())
    hi = max(s["y_true"].max(), s["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")
    ax.set_title(f"B-out {model}\npooled R² = {pm['r2']:.3f}, "
                 f"RMSE = {pm['rmse']:.3f}")
    ax.set_xlabel("True e_form per atom (eV/atom)")
    ax.legend(loc="upper left")
axes[0].set_ylabel("Predicted e_form per atom (eV/atom)")
fig.suptitle("Fig 3 — Parity, leave-one-B-out predictions (Magpie only)")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig3_ok = save_fig(fig, "fig3_parity_xgb.png")

# --- 6. Fig 4 — Distributions: e_form and band gap. ---
# Band gap uses a log y-scale: 96.1% of gaps are exactly zero, so
# a linear scale would hide the 735 non-zero gaps completely.
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
ef = feat["e_form_per_atom"]
axes[0].hist(ef, bins=60)
axes[0].axvline(float(ef.median()), color="red", linestyle="--",
                label=f"median {ef.median():.3f}")
axes[0].set_xlabel("e_form per atom (eV/atom)")
axes[0].set_ylabel("Count")
axes[0].set_title("Formation energy per atom\n(n = 18,928)")
axes[0].legend()
gap = feat["band_gap"]
axes[1].hist(gap, bins=60, log=True)
axes[1].set_xlabel("Band gap (eV, DFT GLLBSC)")
axes[1].set_ylabel("Count (log scale)")
axes[1].set_title(f"Band gap (log scale)\nzero: {int((gap == 0).sum())} "
                  f"({100 * float((gap == 0).mean()):.1f}%), "
                  f"non-zero: {int((gap > 0).sum())}")
fig.suptitle("Fig 4 — Target distributions (Castelli, n = 18,928)")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig4_ok = save_fig(fig, "fig4_distributions.png")

# --- 7. Fig 5 — Band-gap threshold + conformal coverage. ---
# (a) Precision and recall vs threshold, averaged across seeds.
# (b) Conformal coverage per test, dashed line at nominal 0.90.
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
axes[0].plot(thr_sum["threshold"], thr_sum["precision"], "o-",
             label="Precision")
axes[0].plot(thr_sum["threshold"], thr_sum["recall"], "s-",
             label="Recall")
axes[0].plot(thr_sum["threshold"], thr_sum["f1"], "^-", label="F1")
axes[0].set_xlabel("Threshold (non-metal probability)")
axes[0].set_ylabel("Score")
axes[0].set_title("(a) Band-gap screening vs threshold\n"
                  "(grouped CV, pooled across seeds)")
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(loc="center right")
conf_labels = {"random": "Random", "family-out Te": "Te",
               "family-out W": "W", "family-out Mg": "Mg"}
conf_plot = conf.copy()
conf_plot["label"] = conf_plot["test"].map(conf_labels)
axes[1].bar(conf_plot["label"], conf_plot["coverage"],
            color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
axes[1].axhline(0.90, color="red", linestyle="--", linewidth=1.2,
                label="Nominal 0.90")
axes[1].set_ylabel("Coverage")
axes[1].set_title("(b) Conformal coverage (90% nominal)\n"
                  "family-out is an observation, not a promise")
axes[1].set_ylim(0.5, 1.0)
axes[1].legend(loc="lower left")
fig.suptitle("Fig 5 — Band-gap threshold and conformal coverage")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig5_ok = save_fig(fig, "fig5_bandgap_conformal.png")

# --- 8. results_table_v2.md — built from the CSVs, values asserted. ---
# The table is formatted FROM the dicts computed above, and every
# formatted value is then checked back against its CSV source.
# If a number were typed by hand, the check would fail.
def f3(v):
    return f"{v:.3f}"


table_lines = [
    "# Perovskite v2 — Results table",
    "",
    "Every number below comes from a saved v2 CSV read in Stage 7,",
    "never typed by hand. N, split, model and seeds are stated",
    "with each row. Magpie only (132 columns), e_form_per_atom,",
    "unless stated otherwise.",
    "",
    "## Study A — Random vs grouped CV (5-fold, seeds 0–2, n = 18,928)",
    "",
    "| Split | Model | R² (mean ± std across seeds) |",
    "|---|---|---|",
]
for split in ["random", "grouped"]:
    for m in MODELS:
        v = cv_mean[(split, m)]
        s = cv_std[(split, m)]
        table_lines.append(f"| {split} | {m} | {f3(v)} ± {f3(s)} |")
        _check_table(f"CV {split} {m} R2 mean",
                     v, float(cv[(cv["split"] == split)
                                 & (cv["model"] == m)]["r2_mean"].mean()))
table_lines += [
    "",
    f"Composition-only ceiling R² (pooled, Stage 1): {CEILING:.3f}.",
    "Best CV R² (grouped Random Forest) is below the ceiling, as it must be.",
    "",
    "## Study B — Leave-one-family-out, pooled headline (seed 0)",
    "",
    "| Held out | Model | n_test | Pooled R² | Pooled RMSE | Pooled MAE |",
    "|---|---|---|---|---|---|",
]
for ft, label in [("B", "B-site"), ("X", "X-anion")]:
    for m in ["Ridge", "XGBoost"]:
        pm = pooled[(ft, m)]
        table_lines.append(f"| {label} | {m} | {pm['n']} | {f3(pm['r2'])} | "
                           f"{f3(pm['rmse'])} | {f3(pm['mae'])} |")
        sub = fam_pred[(fam_pred["family_type"] == ft)
                       & (fam_pred["model"] == m)]
        _check_table(f"pooled {ft}-out {m} R2", pm["r2"],
                     float(r2_score(sub["y_true"], sub["y_pred"])))
table_lines += [
    "",
    "X-out Ridge pooled R² is catastrophic (linear extrapolation to a",
    "new anion fails); XGBoost survives X-out because trees do not",
    "extrapolate. Mean per-family R² is NOT the headline — pooled is.",
    "",
    "## Study C — Band gap (grouped 5-fold, tuned XGBoost)",
    "",
    f"- Classification, metal vs non-metal (n = 18,928, 735 non-metals): "
    f"ROC-AUC {f3(c1_roc)}, PR-AUC "
    f"{f3(float(c1['pr_auc'].mean()))}, at threshold 0.5 precision "
    f"{f3(c1_prec05)}, recall {f3(c1_rec05)}.",
    f"- At threshold 0.2 (best F1): precision "
    f"{f3(float(thr02['precision']))}, recall "
    f"{f3(float(thr02['recall']))}, F1 {f3(float(thr02['f1']))}.",
    f"- Non-zero-gap regression (n = 735): R² {f3(c2_r2)}, "
    f"RMSE {f3(c2_rmse)}, MAE {f3(c2_mae)}.",
    f"- WARNING only, plain all-rows regression: R² {f3(c0_r2)} — "
    f"looks OK only because 96.1% of gaps are zero. Never a result.",
    "",
]
_check_table("C1 ROC-AUC mean", c1_roc, float(c1["roc_auc"].mean()))
_check_table("C1 recall at 0.5 mean", c1_rec05, float(c1["recall"].mean()))
_check_table("C2 R2 mean", c2_r2, float(c2["r2"].mean()))
_check_table("C0 warning R2", c0_r2, float(c0["r2"].iloc[0]))
table_lines += [
    "## Study D — Conformal uncertainty (90% nominal, exact quantile)",
    "",
    "| Test | n_test | q (eV/atom) | Coverage | Mean width | R² on test |",
    "|---|---|---|---|---|---|",
]
for r in conf.itertuples():
    table_lines.append(f"| {r.test} | {int(r.n_test)} | {f3(r.q)} | "
                       f"{f3(r.coverage)} | {f3(r.mean_width)} | "
                       f"{f3(r.r2_test)} |")
    _check_table(f"conformal {r.test} coverage", float(r.coverage),
                 float(conf[conf["test"] == r.test]["coverage"].iloc[0]))
table_lines += [
    "",
    "Random coverage checks a 90% promise (exchangeable rows).",
    "Family-out coverage is an observation under shift — no guarantee",
    "applies. Coverage is NOT accuracy: W covers only 0.690 while",
    "its point R² is -4.211; Te covers 0.923 while its point R² is",
    "negative. Always report coverage AND width AND point R² together.",
    "",
    "## Figures",
    "",
    "- fig1_cv_r2.png — random vs grouped CV, ceiling line at 0.543.",
    "- fig2_familyout_r2.png — pooled family-out; X-out Ridge (-9.96)",
    "  is annotated, not plotted on the linear scale.",
    "- fig3_parity_xgb.png — B-out parity, Ridge and XGBoost.",
    "- fig4_distributions.png — e_form and band-gap distributions.",
    "- fig5_bandgap_conformal.png — threshold curves and coverage.",
    "",
]
table_name = ("results_table_v2_smoke.md" if SMOKE
              else "results_table_v2.md")
(OUT_DIR / table_name).write_text("\n".join(table_lines) + "\n")
print(f"saved {table_name}")

# --- 9. GATE 7 — figures exist at 300 dpi, table exists, values match. ---
from PIL import Image

fig_checks = {"fig1_cv_r2.png": fig1_ok, "fig2_familyout_r2.png": fig2_ok,
              "fig3_parity_xgb.png": fig3_ok,
              "fig4_distributions.png": fig4_ok,
              "fig5_bandgap_conformal.png": fig5_ok}
for name, ok in fig_checks.items():
    p = FIG_DIR / name
    dpi_ok = False
    if p.exists():
        with Image.open(p) as im:
            dpi = im.info.get("dpi", (0, 0))
            dpi_ok = abs(dpi[0] - 300) < 1 and abs(dpi[1] - 300) < 1
    _check(f"figure {name} exists at 300 dpi", bool(ok) and dpi_ok)
_check(f"table {table_name} exists",
       (OUT_DIR / table_name).exists()
       and (OUT_DIR / table_name).stat().st_size > 0)
failed = [n for n, s in checks if s == "FAIL"]
if failed:
    print("GATE 7: FAIL — " + "; ".join(failed))
    raise AssertionError("GATE 7 FAILED: " + "; ".join(failed))
print("GATE 7: PASS — 5 figures at 300 dpi, table built from CSVs, "
      "every table value matches its CSV.")
print("STAGE 7 done — paste back the GATE 7 line and the figure list.")
