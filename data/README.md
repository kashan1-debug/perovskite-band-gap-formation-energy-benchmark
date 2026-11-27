# Data

No data files are stored in this repository. The dataset is public and is
fetched inside the workflow (Stage 1, `code/01_dataset_audit.py`).

## Source

**Castelli perovskites** via matminer:

```python
from matminer.datasets import load_dataset
df = load_dataset("castelli_perovskites")
```

- 18,928 rows, one per ABX₃ compound — a complete factorial grid:
  52 A-site cations × 52 B-site metals × 7 X-site anions.
- Columns include `formula`, `structure`, `e_form`, `gap gllbsc`,
  `vbm`, `cbm`, `mu_b`, `fermi level`, `fermi width`, `gap is direct`.
- `e_form` is the heat of formation of the whole 5-atom cell (RPBE GGA-DFT),
  in eV. The target used throughout is `e_form_per_atom = e_form / 5`.
  The oxygen reference state is water vapour, not O₂ — see
  `docs/workflow.md` (decision D2) and the manuscript §3.

## Why no data files here

The raw dataset is a few megabytes and is fetched reproducibly from matminer,
so there is no need to vendor it. The generated feature files are large
(`features.parquet`, `features_siteaware.parquet`, `predictions_*.parquet`)
and are excluded by `.gitignore`; regenerate them with Stages 2, 3b, 4 and 9b,
or ask the author for copies.
