# ============================================================
# v2 — STAGE 2: Magpie features (ONE cell, paste into Colab)
# Run AFTER Stage 1 (GATE 1 PASS) in the same session, or after
# Setup B then Setup C in a fresh session. It reloads
# audited.parquet from Drive — it does NOT redo Stage 1.
# It takes about 2-3 minutes. If GATE 2 says FAIL, STOP — do not model.
# Safe to run twice: it backs up features.parquet before overwriting it.
# No models, no predictions, no CV in this cell.
# Paste back: the GATE 2 line, the Magpie count, and the saved shape.
# ============================================================
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty

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

gates = []  # (check, metric, value, expected, PASS/FAIL) — saved to features_report.md


def _gate(check, metric, value, expected, ok):
    status = "PASS" if ok else "FAIL"
    gates.append((check, metric, str(value), str(expected), status))
    print(f"  [{status}] {check} — {metric}: {value} (expected {expected})")
    return ok


# --- 1. Reload audited.parquet from Stage 1 (do NOT redo Stage 1) ---
audit_path = OUT_DIR / "audited.parquet"
if not audit_path.exists():
    raise FileNotFoundError(
        "audited.parquet not found in OUT_DIR. Run Stage 1 first, "
        "then this cell again."
    )
audited = pd.read_parquet(audit_path)
print("audited.parquet shape:", audited.shape, "(expected (18928, 18))")
print("audited columns:", list(audited.columns))
_gate("1_audited_input", "audited_shape", audited.shape, "(18928, 18)",
      audited.shape == (18928, 18))
_gate("1_audited_input", "B_site_and_target_present",
      "B_site" in audited.columns and "e_form_per_atom" in audited.columns,
      True,
      "B_site" in audited.columns and "e_form_per_atom" in audited.columns)

# --- 2. Reload Castelli structures ONLY to get compositions for featurising ---
# audited.parquet has no Structure objects (parquet cannot save them),
# so the compositions must come from a fresh Castelli load. This is NOT
# redoing Stage 1: nothing is re-derived, only the compositions are read.
# Verify row-by-row alignment BEFORE featurising — if the two frames do
# not line up, the Magpie features would be attached to the wrong rows.
raw = load_dataset("castelli_perovskites")
print("Castelli reload shape:", raw.shape, "(expected (18928, 10))")
compositions = [s.composition for s in raw["structure"]]
align = bool(
    len(raw) == len(audited)
    and (raw["formula"].values == audited["formula_ordered"].values).all()
    and np.allclose(raw["e_form"].values, audited["e_form"].values)
)
print("alignment check (formula_ordered, e_form match row by row):", align)
_gate("2_alignment", "castelli_reload_shape", raw.shape, "(18928, 10)",
      raw.shape == (18928, 10))
_gate("2_alignment", "alignment_row_by_row", align, True, align)
if not align:
    raise AssertionError(
        "ALIGNMENT FAILED: Castelli reload does not match audited.parquet "
        "row by row. STOP — do not featurise."
    )

# --- 3. Magpie featurisation (D6: Magpie only, 132 columns) ---
# Magpie = elemental-property statistics over the composition only.
# It cannot see the crystal structure, and it cannot tell A from B —
# that is exactly why the composition-only ceiling from Stage 1 exists.
t0 = time.time()
feat = ElementProperty.from_preset("magpie")
comp_df = pd.DataFrame({"composition": compositions})
mag = feat.featurize_dataframe(comp_df.copy(), col_id="composition")
elapsed = time.time() - t0
print(f"featurisation took {elapsed:.0f} s (v1 in Colab: ~2 min 20 s)")
magpie_cols = [c for c in mag.columns if c.startswith("MagpieData")]
print("Magpie columns:", len(magpie_cols), "(expected 132)")
print("first 5 Magpie columns:", magpie_cols[:5])
magpie_df = mag[magpie_cols].reset_index(drop=True)
nan_total = int(magpie_df.isna().sum().sum())
print("NaN cells in Magpie features:", nan_total, "(expected 0)")
_gate("3_magpie", "magpie_columns", len(magpie_cols), 132,
      len(magpie_cols) == 132)
_gate("3_magpie", "magpie_nan_cells", nan_total, 0, nan_total == 0)
_gate("3_magpie", "rows_match_audited", len(magpie_df), len(audited),
      len(magpie_df) == len(audited))

# --- 4. Leakage assert, BEFORE saving or modelling (D6) ---
# X must be Magpie columns ONLY. These DFT columns must never be in X:
# vbm and cbm define the band gap directly (gap = cbm - vbm), fermi
# level / fermi width / mu_b are DFT outputs, and e_form / band_gap /
# e_form_per_atom are targets. Including any of them is cheating.
FORBIDDEN = ["vbm", "cbm", "fermi level", "fermi width", "mu_b",
             "e_form", "band_gap", "e_form_per_atom"]
leak = [c for c in magpie_cols if c in FORBIDDEN]
print("leakage columns in X (should be []):", leak)
print("NOTE: e_form / band_gap / vbm / cbm DO appear in the full features")
print("frame below as target / source columns — that is fine. They are")
print("NOT in the Magpie feature set, and must never be used as inputs.")
_gate("4_leakage", "leakage_columns_in_X", leak, "[]", leak == [])

# --- 5. Build the features frame: audited columns + Magpie columns ---
features = pd.concat([audited.reset_index(drop=True), magpie_df], axis=1)
print("features shape:", features.shape,
      f"(expected (18928, {len(audited.columns) + 132}))")
dup_cols = [c for c in features.columns if list(features.columns).count(c) > 1]
print("duplicate column names:", dup_cols, "(expected [])")
_gate("5_build", "features_shape", features.shape,
      f"(18928, {len(audited.columns) + 132})",
      features.shape == (18928, len(audited.columns) + 132))
_gate("5_build", "duplicate_columns", dup_cols, "[]", dup_cols == [])

# --- 6. The 148 vs 150 discrepancy — investigate, do NOT hide it ---
# v1 expected (18928, 150) but Colab printed (18928, 148), with Magpie
# still 132. Print every non-Magpie column and diff it against the
# audited columns, so the 2-column difference is named, not hidden.
# Why 148 in v1 is now explained: v1 featurised a frame that still
# contained the Structure and Composition object columns, so its
# 150-column frame included them; dropping those 2 object columns
# before saving left 148. In v2, audited.parquet ALREADY excludes
# Structure/Composition, so audited (18) + Magpie (132) = 150 is the
# correct saved shape here. If this cell prints 148, that means 2
# audited columns went missing — find them below, do not just accept it.
non_magpie = [c for c in features.columns if not c.startswith("MagpieData")]
missing = [c for c in audited.columns if c not in non_magpie]
extra = [c for c in non_magpie if c not in audited.columns]
print("non-Magpie columns in features.parquet:", len(non_magpie),
      "(expected", len(audited.columns), ")")
print("  non-Magpie column list:", non_magpie)
print("  missing vs audited columns:", missing, "(expected [])")
print("  extra vs audited columns:", extra, "(expected [])")
_gate("6_shape_audit", "non_magpie_columns", len(non_magpie),
      len(audited.columns), len(non_magpie) == len(audited.columns))
_gate("6_shape_audit", "missing_audited_columns", missing, "[]", missing == [])
_gate("6_shape_audit", "extra_columns", extra, "[]", extra == [])
# Object check: parquet cannot save pymatgen Structure/Composition
# OBJECTS. Plain string columns (formula_ordered, A_site, ...) are
# also stored as dtype "object" in pandas, but pyarrow saves those
# fine as strings — they are NOT the v1 bug. So check the actual
# danger: that no Structure/Composition object survived, and that
# every object-dtype column holds plain strings.
obj_cols = [c for c in features.columns
            if str(features[c].dtype) == "object"]
bad_obj = [c for c in obj_cols
           if not isinstance(features[c].iloc[0], str)]
print("object-dtype columns (all plain strings, fine for parquet):", obj_cols)
print("object columns holding non-string objects (must be []):", bad_obj)
print("'structure' / 'composition' columns present (must be False):",
      ("structure" in features.columns, "composition" in features.columns))
_gate("6_shape_audit", "object_dtype_columns", obj_cols,
      "string columns only", bad_obj == []
      and "structure" not in features.columns
      and "composition" not in features.columns)

# --- 7. Save features.parquet (backup first, safe to run twice) ---
feat_path = OUT_DIR / "features.parquet"
if feat_path.exists():
    shutil.copy2(feat_path, OUT_DIR / "features_backup.parquet")
    print("backup saved: features_backup.parquet")
features.to_parquet(feat_path, index=False)
print("saved features.parquet:", features.shape)
with open(OUT_DIR / "column_list_features.txt", "w") as fh:
    fh.write("\n".join(features.columns) + "\n")
print("saved column_list_features.txt:", len(features.columns), "columns")

# --- 8. Reload and verify the saved file ---
reloaded = pd.read_parquet(feat_path)
reloaded_magpie = [c for c in reloaded.columns if c.startswith("MagpieData")]
reloaded_nan = int(reloaded[reloaded_magpie].isna().sum().sum())
align2 = bool(
    len(reloaded) == len(audited)
    and (reloaded["formula_ordered"].values
         == audited["formula_ordered"].values).all()
    and np.allclose(reloaded["e_form_per_atom"].values,
                    audited["e_form_per_atom"].values)
    and (reloaded["B_site"].values == audited["B_site"].values).all()
)
print("reloaded shape:", reloaded.shape, "(expected", str(features.shape) + ")")
print("reloaded Magpie columns:", len(reloaded_magpie), "(expected 132)")
print("reloaded NaN cells in Magpie:", reloaded_nan, "(expected 0)")
print("alignment check vs audited.parquet:", align2)
_gate("8_reload_verify", "saved_file_reopens_same_shape", reloaded.shape,
      str(features.shape), reloaded.shape == features.shape)
_gate("8_reload_verify", "reloaded_magpie_columns", len(reloaded_magpie), 132,
      len(reloaded_magpie) == 132)
_gate("8_reload_verify", "reloaded_magpie_nan_cells", reloaded_nan, 0,
      reloaded_nan == 0)
_gate("8_reload_verify", "alignment_vs_audited", align2, True, align2)

# --- 9. Save the Stage 2 report ---
report_lines = [
    "# Perovskite v2 — Stage 2 Magpie features report",
    "",
    "Magpie only (132 columns). No models, no predictions in this stage.",
    "Features = audited.parquet columns + Magpie columns. "
    "No DFT column is in the Magpie feature set.",
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
    f"- Rows: {len(features)}, saved shape: {features.shape}",
    f"- Magpie columns: {len(magpie_cols)}, NaN cells: {nan_total}",
    f"- Leakage columns in X: {leak}",
    f"- Non-Magpie columns: {len(non_magpie)}; "
    f"missing vs audited: {missing}; extra vs audited: {extra}",
    f"- Featurisation time: {elapsed:.0f} s",
    "",
    "## The 148 vs 150 note",
    "",
    "v1 printed (18928, 148) in Colab although 150 was expected. "
    "In v1 the featurised frame still contained the Structure and "
    "Composition object columns, so dropping those 2 before saving "
    "left 148. In v2 audited.parquet already excludes them, so "
    "audited (18) + Magpie (132) = 150 is the correct saved shape. "
    "If a future run prints 148, 2 audited columns went missing — "
    "see the missing/extra lists above, do not just accept it.",
    "",
]
failures = [f"{c} — {m}: {v} (expected {e})"
            for c, m, v, e, s in gates if s == "FAIL"]
if failures:
    report_lines.append("## Verdict")
    report_lines.append("")
    report_lines.append("A GATE FAILED — STOP. Do not model.")
    report_lines.append("")
    for f in failures:
        report_lines.append(f"- {f}")
else:
    report_lines.append("## Verdict")
    report_lines.append("")
    report_lines.append("ALL GATES PASS — features ready. Next: Stage 3 CV.")
(OUT_DIR / "features_report.md").write_text("\n".join(report_lines) + "\n")
print("saved features_report.md")

# --- GATE 2: stop here if anything fails. Do NOT model. ---
if failures:
    print("GATE 2: FAIL —", "; ".join(failures))
    raise AssertionError("GATE 2 FAILED: " + "; ".join(failures))
print("GATE 2: PASS — features ready. Next: Stage 3 CV.")
print("STAGE 2 done — paste back the GATE 2 line, the Magpie count, "
      "and the saved shape.")
