# Perovskite v2 — Stage 2 Magpie features report

Magpie only (132 columns). No models, no predictions in this stage.
Features = audited.parquet columns + Magpie columns. No DFT column is in the Magpie feature set.

## Gates

- [PASS] 1_audited_input — audited_shape: (18928, 18) (expected (18928, 18))
- [PASS] 1_audited_input — B_site_and_target_present: True (expected True)
- [PASS] 2_alignment — castelli_reload_shape: (18928, 10) (expected (18928, 10))
- [PASS] 2_alignment — alignment_row_by_row: True (expected True)
- [PASS] 3_magpie — magpie_columns: 132 (expected 132)
- [PASS] 3_magpie — magpie_nan_cells: 0 (expected 0)
- [PASS] 3_magpie — rows_match_audited: 18928 (expected 18928)
- [PASS] 4_leakage — leakage_columns_in_X: [] (expected [])
- [PASS] 5_build — features_shape: (18928, 150) (expected (18928, 150))
- [PASS] 5_build — duplicate_columns: [] (expected [])
- [PASS] 6_shape_audit — non_magpie_columns: 18 (expected 18)
- [PASS] 6_shape_audit — missing_audited_columns: [] (expected [])
- [PASS] 6_shape_audit — extra_columns: [] (expected [])
- [PASS] 6_shape_audit — object_dtype_columns: ['formula', 'formula_ordered', 'formula_reduced', 'composition_group', 'A_site', 'B_site', 'X_anion', 'A_family'] (expected string columns only)
- [PASS] 8_reload_verify — saved_file_reopens_same_shape: (18928, 150) (expected (18928, 150))
- [PASS] 8_reload_verify — reloaded_magpie_columns: 132 (expected 132)
- [PASS] 8_reload_verify — reloaded_magpie_nan_cells: 0 (expected 0)
- [PASS] 8_reload_verify — alignment_vs_audited: True (expected True)

## Key numbers

- Rows: 18928, saved shape: (18928, 150)
- Magpie columns: 132, NaN cells: 0
- Leakage columns in X: []
- Non-Magpie columns: 18; missing vs audited: []; extra vs audited: []
- Featurisation time: 175 s

## The 148 vs 150 note

v1 printed (18928, 148) in Colab although 150 was expected. In v1 the featurised frame still contained the Structure and Composition object columns, so dropping those 2 before saving left 148. In v2 audited.parquet already excludes them, so audited (18) + Magpie (132) = 150 is the correct saved shape. If a future run prints 148, 2 audited columns went missing — see the missing/extra lists above, do not just accept it.

## Verdict

ALL GATES PASS — features ready. Next: Stage 3 CV.
