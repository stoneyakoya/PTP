### Repository structure and outputs

This project follows a Cookiecutter Data Science–style layout for data and reports. Use this document as the single source of truth for where inputs/outputs live.

#### Top-level layout
- `data/raw/`: immutable source data (do not edit)
- `data/interim/`: intermediate CSVs produced during preprocessing
- `data/processed/`: optional finalized datasets (not always used)
- `data/protein_embeddings/`: residue-level protein embeddings (ESM2, `.npz`)
- `data/peptide_embeddings/`: peptide-region embeddings with `[CLS]` prepended (`.npz`)
- `data/dataset/`: split datasets for training/validation/testing (CSV)
- `reports/figures/`: plots and images
  - `reports/figures/dataset/`: time-series box plots
  - `reports/figures/dataset/clustering/`: clustering sample plots per cluster
- `reports/tables/`: tabular summaries (e.g., dataset pre/post stats)
- `reports/results/`: evaluation artifacts (CSV metrics, logs)
  - `reports/results/baseline/`: baseline outputs (per-run CSV metrics)
  - `reports/results/timeseq/`: TimeSeq training results (per-arch, per-fold; baseline-like summaries)
  - `reports/results/aaseq/`: AASeq training results (per-arch, per-fold; baseline-like summaries)
- `scripts/`: preprocessing, clustering, dataset creation
- `src/`: training/evaluation code
  - `src/timeseq.py`: CLI runner for TimeSeq (multitask or turnover), exports baseline-like results
  - `src/aaseq.py`: CLI runner for AASeq (multitask or turnover), exports baseline-like results
- `docs/`: documentation (methodology and guides)
  - `docs/models.md`: Model architectures, inputs, objectives, training/evaluation, CLI

#### Core scripts and their outputs
- `scripts/preprocess_data.py`
  - Reads `data/raw/pSILAC_TMT.csv`
  - Summarizes peptides and computes turnover series
  - Normalization: clipping or MinMax
  - Fetches protein sequences, filters peptides that occur more than once within the same protein
  - Generates ESM2 protein embeddings and extracts peptide embeddings
  - Writes time-series plots to `reports/figures/dataset/`
  - Writes pre/post dataset stats to `reports/tables/dataset_stats_prepost.csv`
  - Writes intermediates to `data/interim/`

- `scripts/create_dataset.py`
  - Runs time-series clustering (`Cluster` column)
  - Relabels clusters automatically from fastest → slowest (configurable)
  - Saves clustering figures to `reports/figures/dataset/clustering/`
  - Creates K-fold or standard splits under `data/dataset/`

- `src/baseline.py`
  - Reads split datasets from `data/dataset/sampling/fold_{k}/...` by default
    - To use normal splits, switch paths to `data/dataset/normal/fold_{k}/...`
  - Writes evaluation results (per-fold and aggregated) to
    `reports/results/baseline/{INPUT}_{TARGET}_{VARIANT}`
    - If `variant=late` and an hour is specified, suffix `_t{hour}` is appended
  - Files: `results.csv`, `avg_results.csv`, `std_results.csv`

#### How to run
```bash
python scripts/preprocess_data.py --raw_csv data/raw/pSILAC_TMT.csv
python scripts/create_dataset.py
```

#### Notes
- Peptide uniqueness is enforced within the same protein sequence (exactly one occurrence).
- Clustering label order is determined by a configurable kinetic score and relabeled to 0..K-1.
- Figures go under `reports/figures/`; tables under `reports/tables/`; evaluation CSVs/logs under `reports/results/`.

