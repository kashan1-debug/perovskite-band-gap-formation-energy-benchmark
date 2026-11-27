# Perovskite v2 — Stage 1 dataset audit report

Castelli only (`castelli_perovskites`, matminer). No MatBench, no merge.
No models, no features, no predictions in this stage.

## Target definition

e_form is the heat of formation of the entire 5-atom perovskite cell, in eV, as calculated by RPBE GGA-DFT (Matbench perovskites docs). Oxygen reference state is water vapour, not O2. The Castelli matminer entry alone just says 'heat of formation in eV' and does not state per-cell; the per-cell definition is confirmed by the Matbench entry for the same 18,928 structures and by a live browser check of the official matminer dataset page on 2026-11-25. Therefore e_form_per_atom = e_form / nsites, with nsites verified to be 5 for every row before dividing.

## Gates

- [PASS] 1_records_intact — shape: (18928, 10) (expected (18928, 10))
- [PASS] 1_records_intact — columns_include: formula/e_form/gap gllbsc/structure/vbm/cbm (expected all present)
- [PASS] 1_records_intact — missing_total: 0 (expected 0)
- [PASS] 2_structure_valid — valid_Structure_rows: 18928 (expected 18928)
- [PASS] 3_nsites — nsites_all_5: {5: 18928} (expected {5: 18928})
- [PASS] 4_target — e_form_per_atom_mean: 0.294 (expected ~0.294)
- [PASS] 4_target — e_form_per_atom_std: 0.149 (expected ~0.149)
- [PASS] 4_target — negative_e_form_per_atom_rows: 74 (expected 74)
- [PASS] 4_target — band_gap_zero: 18193 (expected 18193)
- [PASS] 4_target — band_gap_nonzero: 735 (expected 735)
- [PASS] 7_ABX_parse — unparseable_rows: 0 (expected 0)
- [PASS] 7_ABX_parse — ordered_formula_mismatch_rows: 0 (expected 0)
- [PASS] 7_ABX_parse — B_site_unique: 52 (expected 52)
- [PASS] 7_ABX_parse — B_site_count_each: [364] (expected [364])
- [PASS] 7_ABX_parse — X_anion_unique: 7 (expected 7)
- [PASS] 7_ABX_parse — X_anion_count_each: [2704] (expected [2704])
- [PASS] 5_duplicates — unique_ordered_formulas: 18928 (expected 18928)
- [PASS] 5_duplicates — unique_reduced_formulas: 9646 (expected 9646)
- [PASS] 5_duplicates — groups_with_gt1_row: 9282 (expected 9282)
- [PASS] 6_duplicate_targets — mean_within_group_std: 0.109 (expected ~0.109)
- [PASS] 6_duplicate_targets — ceiling_R2_pooled: 0.543 (expected ~0.543)
- [PASS] 6_duplicate_targets — ceiling_R2_mean_std: 0.458 (expected ~0.46)
- [PASS] 8_reproducible_report — saved_file_reopens_same_shape: (18928, 18) (expected (18928, 18))
- [PASS] 8_reproducible_report — alignment_row_by_row: True (expected True)

## Key numbers

- Rows: 18928, columns in audited.parquet: 18
- e_form_per_atom: mean 0.294, std 0.149, median 0.272, min -0.128, max 1.032, negative rows 74
- band_gap: zero 18193, non-zero 735, in 1.3-1.6 eV 77
- Unique ordered formulas 18928, unique reduced 9646, groups with >1 row 9282
- Within-group std (mean) 0.109, composition-only ceiling R2 (pooled) 0.543, (mean-std cross-check) 0.458, max within-group range 0.728
- A/B/X parse: unparseable 0, ordered-formula mismatches 0, A==B diagonal rows 364 (not mismatches)
- B_site 52 x 364 rows; X_anion 7 x 2704 rows; A_family {'other': 14560, 'alkali': 1820, 'alkaline-earth': 1456, 'lanthanide': 1092}

## Verdict

ALL GATES PASS — dataset audited. Next: Stage 2 features.
