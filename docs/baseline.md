# Baseline (XGB / RF) for peptide time-series

This baseline predicts one of the precomputed targets per peptide over 8 timepoints:
- Label loss (`loss`)
- Label incorporation (`incorporation`)
- Turnover rate (`turnover` = incorporation / (incorporation + loss))

Two variants are supported:
- `full`: multi-output regression to predict all 8 timepoints simultaneously
- `late`: single-output regression for a specific timepoint (e.g., 48h)

You can also train both Label loss and Incorporation jointly and evaluate Turnover derived from predictions:
- `--target both` trains two models (loss, incorporation) and computes metrics on turnover = inc/(inc+loss)

## Inputs
- **Feature type**: `AAC` (local amino acid composition) or `ESM2` (sequence embeddings summarized)

### TimeSeq/AASeq runners (baseline-like outputs)

- TimeSeq (ESM2のみ):
  - マルチタスク（loss+incorp）:
    - `python -m src.timeseq --input_type ESM2 --arch LSTM --target both --dataset sampling --folds 10`
  - Incorporationのみ（incorp-only）:
    - `python -m src.timeseq --input_type ESM2 --arch LSTM --target incorp --dataset sampling --folds 10`
  - Turnover直接予測:
    - `python -m src.timeseq --input_type ESM2 --arch LSTM --target turnover --dataset normal --folds 10`
  - 出力: `reports/results/timeseq/{INPUT}_{TARGET}_{ARCH}_{DATASET}/`

- AASeq (ESM2のみ):
  - マルチタスク（loss+incorp）:
    - `python -m src.aaseq --input_type ESM2 --arch Transformer --target both --dataset sampling --folds 10`
  - Incorporationのみ（incorp-only）:
    - `python -m src.aaseq --input_type ESM2 --arch Transformer --target incorp --dataset sampling --folds 10`
  - Turnover直接予測:
    - `python -m src.aaseq --input_type ESM2 --arch Transformer --target turnover --dataset normal --folds 10`
  - 出力: `reports/results/aaseq/{INPUT}_{TARGET}_{ARCH}_{DATASET}/`
- **Dataset location**:
  - Default: `data/dataset/sampling/fold_{k}/`（sampled データセット）
  - 切替方法: `src/baseline.py` 冒頭の読み込みパスを
    - sampled: `data/dataset/sampling/fold_{k}/...`
    - normal:  `data/dataset/normal/fold_{k}/...`
    に変更してください（or 後でCLI化可能）。

## Usage

```bash
# Full 8-step prediction of Label loss with AAC features
python -m src.baseline --input_type AAC --target loss --variant full --folds 10

# Full 8-step prediction of Turnover with ESM2 features
python -m src.baseline --input_type ESM2 --target turnover --variant full --folds 10

# Late-timepoint (48h) prediction of Label incorporation with AAC features
python -m src.baseline --input_type AAC --target incorporation --variant late --late_hour 48 --folds 10

# Train both loss & incorporation and evaluate turnover (8-step)
python -m src.baseline --input_type AAC --target both --variant full --folds 10

# Train both loss & incorporation and evaluate turnover at 48h only
python -m src.baseline --input_type AAC --target both --variant late --late_hour 48 --folds 10
```

## Outputs
- **Results path**: `reports/results/baseline/{INPUT}_{TARGET}_{VARIANT}`
  - When `variant=late` and `--late_hour` is specified, suffix `_t{hour}` is appended.
- Files:
  - `results.csv`: per-fold metrics (RMSE and R2 for XGB and RF)
  - `avg_results.csv`: mean over folds
  - `std_results.csv`: standard deviation over folds
  - `overall_summary.csv`: merged mean/std table per model
  - `summary.md`: quick Markdown table (mean±std)

## Notes
- When `input_type=AAC`, features are 20-D local AAC around the peptide.
- When `input_type=ESM2`, per-peptide features are aggregated from residue embeddings.
- `turnover` uses the dataset-provided `Target_TurnoverRate` prepared in preprocessing.

## Direct prediction of turnover rate
`turnover` を直接ターゲットとして学習・評価する例。

```bash
# 8点同時（ESM2特徴量）
python -m src.baseline --input_type ESM2 --target turnover --variant full --folds 10

# 48hのみ（AAC特徴量）
python -m src.baseline --input_type AAC --target turnover --variant late --late_hour 48 --folds 10
```
