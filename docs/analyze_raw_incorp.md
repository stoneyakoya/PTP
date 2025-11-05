### Raw incorporation analysis (pSILAC TMT)

This analysis inspects raw `corrRatio_*h` time series and `comb.K` for each `Peptidoform` as-is (PTMs preserved), and compares PTM vs non-PTM behavior for identical peptide sequences.

#### What it does
- Aggregates `corrRatio_*h` and `comb.K` by `Peptidoform` and PTM state.
- Saves aggregated table to `data/interim/incorp_by_peptidoform.csv`.
- Visualizes overall `comb.K` distribution and PTM vs non-PTM K density.
- Compares PTM vs non-PTM for the same `Peptidoform`:
  - Delta K (with PTM - no PTM)
  - Correlation and L2 between time-series shapes
  - Exports stats and example overlays

#### Expected columns in raw CSV
- `Peptidoform` (string)
- PTM indicators: `ac-Nterm`, `ac-K`, `ph-STY` (0/1)
- Time series: `corrRatio_0h`, `corrRatio_1h`, `corrRatio_3h`, ..., `corrRatio_48h`, `corrRatio_infin.h` (present subset is auto-detected)
- Kinetic parameter: `comb.K` (peptidoform-level; duplicates like `comb.K.1` are ignored)

#### Run
```bash
python scripts/analyze_raw_incorp.py \
  --raw_csv /home/ishino/PTP/data/raw/pSILAC_TMT.csv \
  --out_interim /home/ishino/PTP/data/interim/incorp_by_peptidoform.csv \
  --fig_dir /home/ishino/PTP/reports/figures/raw_incorp \
  --table_dir /home/ishino/PTP/reports/tables/raw_incorp
```

#### Outputs
- Tables
  - `reports/tables/raw_incorp/ptm_vs_nonptm_stats.csv`
  - `data/interim/incorp_by_peptidoform.csv`
- Figures
  - `reports/figures/raw_incorp/k_distribution.png`
  - `reports/figures/raw_incorp/k_density_by_ptm.png`
  - `reports/figures/raw_incorp/delta_k_distribution.png`
  - `reports/figures/raw_incorp/ptm_vs_nonptm_top_deltaK.png`

#### Notes
- Aggregation uses mean across evidence rows for the same `Peptidoform` and PTM state.
- PTM state is `has_ptm = (ac-Nterm + ac-K + ph-STY) > 0`.
- Time points are auto-detected from headers matching `corrRatio_*h`.
- If `comb.K` appears multiple times due to repeated header groups, the first occurrence is used (peptidoform level).


