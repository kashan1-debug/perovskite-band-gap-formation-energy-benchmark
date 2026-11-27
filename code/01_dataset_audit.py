# ============================================================
# v2 — STAGE 1: Dataset audit (ONE cell, paste into Colab)
# Run AFTER Setup B then Setup C in a fresh session.
# No modelling in this cell. It only proves the data is right.
# It takes about 2 minutes. If GATE 1 says FAIL, STOP — do not featurise.
# Safe to run twice: it backs up audited.parquet before overwriting it.
# Paste back: the GATE 1 line and the two ceiling lines.
# ============================================================
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Composition

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

gates = []  # (check, metric, value, expected, PASS/FAIL) — saved to audit_report.md


def _gate(check, metric, value, expected, ok):
    status = "PASS" if ok else "FAIL"
    gates.append((check, metric, str(value), str(expected), status))
    print(f"  [{status}] {check} — {metric}: {value} (expected {expected})")
    return ok


# --- 1. Load Castelli ONLY. No MatBench, no merge, ever. (D1) ---
df = load_dataset("castelli_perovskites")
raw_shape = df.shape  # check BEFORE adding columns — df grows below
print("shape:", raw_shape, "(expected (18928, 10))")
print("columns:", list(df.columns))
print("missing total:", int(df.isna().sum().sum()), "(expected 0)")
_gate("1_records_intact", "shape", raw_shape, "(18928, 10)", raw_shape == (18928, 10))
_gate("1_records_intact", "columns_include",
      "formula/e_form/gap gllbsc/structure/vbm/cbm", "all present",
      all(c in df.columns for c in ["formula", "e_form", "gap gllbsc",
                                     "structure", "vbm", "cbm"]))
_gate("1_records_intact", "missing_total", int(df.isna().sum().sum()), 0,
      int(df.isna().sum().sum()) == 0)
df = df.rename(columns={"gap gllbsc": "band_gap"})

# --- 2. Structures valid, every structure has 5 sites (D2: verify BEFORE dividing) ---
valid = [hasattr(s, "num_sites") and s.num_sites > 0 for s in df["structure"]]
df["nsites"] = [s.num_sites for s in df["structure"]]
print("valid Structure rows:", int(sum(valid)), "(expected 18928)")
print("nsites counts:", df["nsites"].value_counts().to_dict(), "(expected {5: 18928})")
_gate("2_structure_valid", "valid_Structure_rows", int(sum(valid)), 18928, all(valid))
_gate("3_nsites", "nsites_all_5", df["nsites"].value_counts().to_dict(), "{5: 18928}",
      set(df["nsites"].unique()) == {5} and len(df) == 18928)

# --- 3. Target: e_form is per 5-atom CELL, so e_form_per_atom = e_form / nsites ---
# Matbench docs: "Heat of formation of the entire 5-atom perovskite cell,
# in eV" (RPBE GGA-DFT; oxygen reference is water vapour, not O2).
# The Castelli entry alone just says "heat of formation in eV" — the
# per-cell definition comes from the Matbench entry for the same 18,928
# structures, confirmed live on the matminer dataset page on 2026-11-25.
df["e_form_per_atom"] = df["e_form"] / df["nsites"]
ef = df["e_form_per_atom"]
print("e_form_per_atom mean %.3f std %.3f median %.3f min %.3f max %.3f"
      % (ef.mean(), ef.std(), ef.median(), ef.min(), ef.max()),
      "(v1: 0.294 / 0.149 / 0.272 / -0.128 / 1.032)")
print("negative e_form_per_atom (NOT a stability claim):",
      int((ef < 0).sum()), "of", len(df), "(v1: 74)")
_gate("4_target", "e_form_per_atom_mean", round(float(ef.mean()), 3), "~0.294",
      abs(float(ef.mean()) - 0.294) < 0.002)
_gate("4_target", "e_form_per_atom_std", round(float(ef.std()), 3), "~0.149",
      abs(float(ef.std()) - 0.149) < 0.002)
_gate("4_target", "negative_e_form_per_atom_rows", int((ef < 0).sum()), 74,
      int((ef < 0).sum()) == 74)
print("band_gap == 0:", int((df["band_gap"] == 0).sum()), "(expected 18193)",
      "| > 0:", int((df["band_gap"] > 0).sum()), "(expected 735)",
      "| in 1.3-1.6 eV:", int(((df["band_gap"] >= 1.3) & (df["band_gap"] <= 1.6)).sum()),
      "(expected 77)")
nz = df.loc[df["band_gap"] > 0, "band_gap"]
print("non-zero gap: n=%d mean %.3f median %.3f max %.3f"
      % (len(nz), nz.mean(), nz.median(), nz.max()), "(v1: 735 / 1.824 / 1.4 / 7.0)")
print("gap is direct:", df["gap is direct"].value_counts().to_dict(),
      "(v1: True 18331, False 597)")
_gate("4_target", "band_gap_zero", int((df["band_gap"] == 0).sum()), 18193,
      int((df["band_gap"] == 0).sum()) == 18193)
_gate("4_target", "band_gap_nonzero", int((df["band_gap"] > 0).sum()), 735,
      int((df["band_gap"] > 0).sum()) == 735)

# --- 4. Three IDs (D4) and A/B/X from STRUCTURE SITE ORDER only (D5) ---
df["formula_ordered"] = df["formula"]  # Castelli formula, A-B-X order
df["composition"] = [s.composition for s in df["structure"]]
df["formula_reduced"] = [c.reduced_formula for c in df["composition"]]
df["composition_group"] = df["formula_reduced"]  # grouping key for CV


def _parse(structure):
    sp = [str(site.specie) for site in structure.sites]
    if len(sp) != 5:
        return None, None, None
    x_amt = {}
    for e in sp[2:]:
        x_amt[e] = x_amt.get(e, 0) + 1
    x_label = "".join(f"{e}{n if n > 1 else ''}" for e, n in sorted(x_amt.items()))
    return sp[0], sp[1], x_label


parsed = [_parse(s) for s in df["structure"]]
df["A_site"], df["B_site"], df["X_anion"] = zip(*parsed)
unparseable = int(df["A_site"].isna().sum())
print("A/B/X unparseable:", unparseable, "(expected 0)")
for f, a, b, x in list(zip(df["formula_ordered"], df["A_site"],
                           df["B_site"], df["X_anion"]))[:5]:
    print(f"  {f} -> A={a} B={b} X={x}")

# Ordered-formula check, on ALL rows. A==B diagonal rows (e.g. BiBiFO2:
# Bi appears twice, once as A and once as B) are NOT mismatches —
# site order still distinguishes A from B by position; count them apart.
n_diagonal = 0
n_mismatch = 0
for formula, a, b, structure in zip(df["formula_ordered"], df["A_site"],
                                    df["B_site"], df["structure"]):
    sp = [str(site.specie) for site in structure.sites]
    comp = Composition(formula)
    amt = {str(e): int(comp[e]) for e in comp.elements}
    x_sum = sum(amt.get(e, 0) for e in set(sp[2:]))
    if a == b and amt.get(a, 0) == 2 and x_sum == 3:
        n_diagonal += 1
    elif amt.get(a, 0) == 1 and amt.get(b, 0) == 1 and x_sum == 3:
        pass
    else:
        n_mismatch += 1
print("A==B diagonal rows (not mismatches):", n_diagonal, "(expected 364 = 52 x 7)")
print("ordered-formula TRUE mismatches:", n_mismatch, "(expected 0)")
_gate("7_ABX_parse", "unparseable_rows", unparseable, 0, unparseable == 0)
_gate("7_ABX_parse", "ordered_formula_mismatch_rows", n_mismatch, 0, n_mismatch == 0)

A_FAMILIES = {
    "alkali": {"Li", "Na", "K", "Rb", "Cs"},
    "alkaline-earth": {"Mg", "Ca", "Sr", "Ba"},
    "lanthanide": {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
                   "Ho", "Er", "Tm", "Yb", "Lu", "Sc", "Y"},
}
df["A_family"] = df["A_site"].apply(
    lambda a: next((fam for fam, els in A_FAMILIES.items() if a in els), "other"))
print("B_site unique:", df["B_site"].nunique(),
      "| B counts:", sorted(df["B_site"].value_counts().unique().tolist()),
      "(expected 52, [364])")
print("X_anion unique:", df["X_anion"].nunique(),
      "| X counts:", sorted(df["X_anion"].value_counts().unique().tolist()),
      "(expected 7, [2704])")
print("A_family counts:", df["A_family"].value_counts().to_dict(),
      "(v1: other 14560, alkali 1820, alkaline-earth 1456, lanthanide 1092)")
_gate("7_ABX_parse", "B_site_unique", int(df["B_site"].nunique()), 52,
      int(df["B_site"].nunique()) == 52)
_gate("7_ABX_parse", "B_site_count_each",
      sorted(df["B_site"].value_counts().unique().tolist()), [364],
      set(df["B_site"].value_counts().unique()) == {364})
_gate("7_ABX_parse", "X_anion_unique", int(df["X_anion"].nunique()), 7,
      int(df["X_anion"].nunique()) == 7)
_gate("7_ABX_parse", "X_anion_count_each",
      sorted(df["X_anion"].value_counts().unique().tolist()), [2704],
      set(df["X_anion"].value_counts().unique()) == {2704})

# --- 5. Duplicate audit — the key table (D3: keep all 18,928 rows) ---
g = df.groupby("formula_reduced")["e_form_per_atom"]
n_i = g.count()
sizes = n_i.value_counts().sort_index()
print("unique ordered formulas:", df["formula_ordered"].nunique(), "(expected 18928)")
print("unique reduced formulas:", df["formula_reduced"].nunique(), "(expected 9646)")
print("reduced groups with >1 row:", int((n_i > 1).sum()), "(expected 9282)")
print("group sizes (size: count):", sizes.to_dict())
_gate("5_duplicates", "unique_ordered_formulas", int(df["formula_ordered"].nunique()),
      18928, int(df["formula_ordered"].nunique()) == 18928)
_gate("5_duplicates", "unique_reduced_formulas", int(df["formula_reduced"].nunique()),
      9646, int(df["formula_reduced"].nunique()) == 9646)
_gate("5_duplicates", "groups_with_gt1_row", int((n_i > 1).sum()), 9282,
      int((n_i > 1).sum()) == 9282)

# Within-group spread. Two ceiling numbers, same data, different definitions:
# (a) POOLED ceiling — the true group-mean ceiling. A composition-only
#     model can at best predict each composition's group mean, and its
#     error is the sum of squared deviations from the group mean, divided
#     by N (all rows). ceiling = 1 - pooled_var / total_var.
#     pooled_var = sum over groups of (sum of squared deviations) / N.
#     NOTE: do NOT divide by sum(n-1) here — that is the unbiased
#     variance estimator, not the group-mean predictor's error, and it
#     gives a wrong ceiling of ~0.07.
# (b) MEAN-STD cross-check — 1 - (mean sample std / overall std)^2.
#     This reproduces the Colab Diagnostic 2 number (~0.46). It is
#     stricter but it is NOT the group-mean ceiling: it squares the mean
#     of the stds instead of averaging the variances. Keep it only as
#     the cross-check.
within_var0 = g.var(ddof=0)
pooled_var = float((within_var0 * n_i).sum() / len(df))
total_var = float(ef.var(ddof=0))
ceiling_pooled = 1 - pooled_var / total_var
mean_std_ddof1 = float(g.std(ddof=1).mean())
ceiling_mean_std = 1 - (mean_std_ddof1 / float(ef.std())) ** 2
print("mean within-group std (ddof=1, groups >1 row): %.3f (v1 diagnostic: 0.109)"
      % mean_std_ddof1)
print("overall std: %.3f (v1: 0.149)" % float(ef.std()))
print("max within-group range: %.3f eV/atom (v1 audit: 0.728)"
      % float((g.max() - g.min()).max()))
print("composition-only R2 ceiling (POOLED, group-mean predictor): %.3f" % ceiling_pooled)
print("  cross-check (Colab diagnostic, mean sample-std): %.3f" % ceiling_mean_std)
print("  -> no composition-only model can beat the pooled ceiling:")
print("     A/B-swapped crystals share one Magpie input but have different energies.")
_gate("6_duplicate_targets", "mean_within_group_std", round(mean_std_ddof1, 3),
      "~0.109", abs(mean_std_ddof1 - 0.109) < 0.005)
_gate("6_duplicate_targets", "ceiling_R2_pooled", round(ceiling_pooled, 3),
      "~0.543", abs(ceiling_pooled - 0.543) < 0.03)
_gate("6_duplicate_targets", "ceiling_R2_mean_std", round(ceiling_mean_std, 3),
      "~0.46", abs(ceiling_mean_std - 0.46) < 0.03)

# --- 6. Save audited.parquet (structure/composition dropped) + column list + report ---
audited = df.drop(columns=["structure", "composition"])
audit_path = OUT_DIR / "audited.parquet"
if audit_path.exists():
    shutil.copy2(audit_path, OUT_DIR / "audited_backup.parquet")
    print("backup saved: audited_backup.parquet")
audited.to_parquet(audit_path, index=False)
print("saved audited.parquet:", audited.shape)

# Alignment check: reload and compare row by row.
reloaded = pd.read_parquet(audit_path)
align = bool(
    len(reloaded) == len(audited)
    and (reloaded["formula_ordered"].values == audited["formula_ordered"].values).all()
    and np.allclose(reloaded["e_form"].values, audited["e_form"].values)
    and np.allclose(reloaded["e_form_per_atom"].values,
                    audited["e_form_per_atom"].values)
    and (reloaded.index.values == audited.index.values).all()
)
print("alignment check (formulas, e_form, indices match row by row):", align)
_gate("8_reproducible_report", "saved_file_reopens_same_shape", reloaded.shape,
      str(audited.shape), reloaded.shape == audited.shape)
_gate("8_reproducible_report", "alignment_row_by_row", align, True, align)
with open(OUT_DIR / "column_list.txt", "w") as fh:
    fh.write("\n".join(audited.columns) + "\n")
print("saved column_list.txt:", len(audited.columns), "columns")

report_lines = [
    "# Perovskite v2 — Stage 1 dataset audit report",
    "",
    "Castelli only (`castelli_perovskites`, matminer). No MatBench, no merge.",
    "No models, no features, no predictions in this stage.",
    "",
    "## Target definition",
    "",
    "e_form is the heat of formation of the entire 5-atom perovskite cell, "
    "in eV, as calculated by RPBE GGA-DFT (Matbench perovskites docs). "
    "Oxygen reference state is water vapour, not O2. The Castelli matminer "
    "entry alone just says 'heat of formation in eV' and does not state "
    "per-cell; the per-cell definition is confirmed by the Matbench entry "
    "for the same 18,928 structures and by a live browser check of the "
    "official matminer dataset page on 2026-11-25. "
    "Therefore e_form_per_atom = e_form / nsites, with nsites verified "
    "to be 5 for every row before dividing.",
    "",
    "## Gates",
    "",
]
for check, metric, value, expected, status in gates:
    report_lines.append(f"- [{status}] {check} — {metric}: {value} (expected {expected})")
report_lines += [
    "",
    "## Key numbers",
    "",
    f"- Rows: {len(df)}, columns in audited.parquet: {audited.shape[1]}",
    f"- e_form_per_atom: mean {ef.mean():.3f}, std {ef.std():.3f}, "
    f"median {ef.median():.3f}, min {ef.min():.3f}, max {ef.max():.3f}, "
    f"negative rows {int((ef < 0).sum())}",
    f"- band_gap: zero {int((df['band_gap'] == 0).sum())}, "
    f"non-zero {int((df['band_gap'] > 0).sum())}, "
    f"in 1.3-1.6 eV {int(((df['band_gap'] >= 1.3) & (df['band_gap'] <= 1.6)).sum())}",
    f"- Unique ordered formulas {df['formula_ordered'].nunique()}, "
    f"unique reduced {df['formula_reduced'].nunique()}, "
    f"groups with >1 row {int((n_i > 1).sum())}",
    f"- Within-group std (mean) {mean_std_ddof1:.3f}, "
    f"composition-only ceiling R2 (pooled) {ceiling_pooled:.3f}, "
    f"(mean-std cross-check) {ceiling_mean_std:.3f}, "
    f"max within-group range {float((g.max() - g.min()).max()):.3f}",
    f"- A/B/X parse: unparseable {unparseable}, "
    f"ordered-formula mismatches {n_mismatch}, "
    f"A==B diagonal rows {n_diagonal} (not mismatches)",
    f"- B_site {df['B_site'].nunique()} x 364 rows; "
    f"X_anion {df['X_anion'].nunique()} x 2704 rows; "
    f"A_family {df['A_family'].value_counts().to_dict()}",
    "",
]
failures = [f"{c} — {m}: {v} (expected {e})"
            for c, m, v, e, s in gates if s == "FAIL"]
if failures:
    report_lines.append("## Verdict")
    report_lines.append("")
    report_lines.append("A GATE FAILED — STOP. Do not featurise or model.")
    report_lines.append("")
    for f in failures:
        report_lines.append(f"- {f}")
else:
    report_lines.append("## Verdict")
    report_lines.append("")
    report_lines.append("ALL GATES PASS — dataset audited. Next: Stage 2 features.")
(OUT_DIR / "audit_report.md").write_text("\n".join(report_lines) + "\n")
print("saved audit_report.md")

# --- GATE 1: stop here if anything fails. Do NOT featurise. ---
if failures:
    print("GATE 1: FAIL —", "; ".join(failures))
    raise AssertionError("GATE 1 FAILED: " + "; ".join(failures))
print("GATE 1: PASS — dataset audited. Next: Stage 2 features.")
print("STAGE 1 done — paste back the GATE 1 line and the two ceiling lines.")
