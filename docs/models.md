### Model overview

This document summarizes the implemented model families, inputs/targets, training objectives, and evaluation/output conventions. It is designed to be close to a paper-ready methods section while staying implementation-faithful.

#### Tasks and targets
- Let y_loss[t], y_incorp[t], y_turn[t] denote the label loss, label incorporation, and turnover at timestep t ∈ {1,3,6,10,16,24,34,48}.
- Turnover is derived as y_turn[t] = y_incorp[t] / (y_incorp[t] + y_loss[t]) with a small epsilon for numerical stability in code paths that do not predict turnover directly.

#### Datasets and splits
- Base: data/dataset/{sampling|normal|customized}/fold_k/{train_fold.csv, validation_fold.csv, test_fold.csv}
- Creation: scripts/create_dataset.py produces three splits:
  - normal: full dataset, K-fold by protein (7:2:1 per fold)
  - sampling: per-cluster balanced by unique peptides before K-fold
  - customized: optional cluster down-sampling then a single train/val/test split

#### Feature configurations
- ESM2 peptide embeddings: Per peptide-timepoint npz containing peptido_embedding (residue-level, 1280-d). For TimeSeq, token-mean pooling is used to obtain a per-timepoint vector.
- Lag features (AASeq only): For each timepoint, 7-step lag vectors are concatenated and broadcast across residues in the sequence to produce (seq_len, 1280 + 7) tensors per task head.

### Architectures

#### TimeSeq family (sequence over time, per-peptide)
Input: X ∈ R^{B×T×D}, where T=8 timepoints, D=1280 (ESM2 mean-pooled).

- LSTM (src/models/LSTM.py)
  - Shared LSTM encoder: LSTM(input_dim=D, hidden_dim=H, num_layers=L, dropout)
  - Two task-specific linear heads: fc_loss, fc_incorporation ∈ R^{H→1}
  - Optional direct turnover head: fc_turnover ∈ R^{H→1}
  - Activations: Sigmoid/ReLU/GELU (configurable), default ReLU in runners
  - Outputs: (ŷ_loss, ŷ_incorp) or (ŷ_loss, ŷ_incorp, ŷ_turn)

- Transformer (src/models/Transformer_based.py → Transformer_TimeSeq)
  - Input projection: Linear(D→E) + LayerNorm + ReLU + Dropout + Linear(E→E)
  - Fixed sinusoidal positional encoding for T=8
  - TransformerEncoder: n_layers × (MHSA, FFN), d_model=E, n_head=H
  - Task heads: fc_loss, fc_incorporation (E→1), optional fc_turnover (E→1)
  - Activation: Sigmoid/ReLU/GELU (configurable)
  - Outputs: (ŷ_loss, ŷ_incorp) or (ŷ_loss, ŷ_incorp, ŷ_turn)

Remarks: TimeSeq uses no lag features; each timepoint feature is the mean-pooled ESM2 embedding for the peptide at that timepoint.

#### AASeq family (sequence over residues, per-timepoint features concatenated with lags)
Input: For each batch example, two tensors for loss/incorporation streams: X_loss, X_incorp ∈ R^{B×L×(1280+7)}, where L is the padded residue length (e.g., 26 or 30). The 7 dims correspond to lag features broadcast to all residues.

- Transformer_AASeq (src/models/Transformer_based.py)
  - Two parallel encoders (loss-stream and incorporation-stream):
    - Per-stream input linear (1280+7 → E)
    - n_layers of custom encoder blocks with MHSA and FFN
    - Global average pooling over residues → E
  - Two regression heads: fc_loss(E→1), fc_incorporation(E→1)
  - Optional direct turnover head: fc_turnover(E→1) using mean of pooled embeddings
  - Return attention scores per layer for analysis (available in forward)
  - Activation: Sigmoid/ReLU/GELU/LeakyReLU (configurable)

### Objectives and training modes

- Multitask ("both")
  - Loss: MSE over ŷ_loss vs y_loss and ŷ_incorp vs y_incorp, with early-time weighting w(t)=3 (train)/2 (val) for t≤6 as implemented in src/train_test.py.
  - Turnover is evaluated as post-hoc ratio ŷ_incorp/(ŷ_incorp+ŷ_loss).

- Turnover-only ("turnover")
  - If a direct turnover head exists, optimize MSE(ŷ_turn, y_turn); otherwise derive predictions via ŷ_incorp/(ŷ_incorp+ŷ_loss).
  - Loss/Inc metrics are not reported in this mode.

- Incorporation-only ("incorp")
  - Optimize MSE(ŷ_incorp, y_incorp) with the same early-time weighting scheme as multitask.
  - Loss head is not optimized and Loss metrics are not reported in this mode.

Regularization and optimization
- Optimizer: Adam with weight decay 1e-2 (configurable)
- Scheduler: ReduceLROnPlateau on validation loss (factor 0.5, patience 3)
- Early stopping: patience and threshold configurable (default threshold 0.01 in runners)
- Gradient clipping: max-norm 1.0

Data loading and batching
- K-fold data loaders per split, default batch size 8 (recommended to scale up as GPU permits)
- For AASeq, lag features are optionally perturbed with light Gaussian noise (lag_noise) during training for robustness.

### Evaluation and reporting

Per fold (src/train_test.py):
- Metrics: RMSE and R² for task(s) enabled by the training mode. In turnover-only: only Turnover metrics.
- Cluster-wise metrics: RMSE/R² per cluster, saved to cluster_scores.csv.
- Full per-timepoint predictions saved to test_results.csv.

Across folds (src/timeseq.py and src/aaseq.py):
- Aggregate DataFrame results.csv under reports/results/{timeseq|aaseq}/{INPUT}_{TARGET}_{ARCH}_{DATASET}/
- Per-metric mean/std saved to avg_results.csv, std_results.csv, and merged overall_summary.csv; summary.md includes a markdown preview.

### Command-line interface

- TimeSeq
  - Multitask: `python -m src.timeseq --input_type ESM2 --arch LSTM --target both --dataset sampling --folds 10`
  - Incorporation-only: `python -m src.timeseq --input_type ESM2 --arch LSTM --target incorp --dataset sampling --folds 10`
  - Turnover-only: `python -m src.timeseq --input_type ESM2 --arch Transformer --target turnover --dataset normal --folds 10`

- AASeq
  - Multitask: `python -m src.aaseq --input_type ESM2 --arch Transformer --target both --dataset sampling --folds 10`
  - Incorporation-only: `python -m src.aaseq --input_type ESM2 --arch Transformer --target incorp --dataset sampling --folds 10`
  - Turnover-only: `python -m src.aaseq --input_type ESM2 --arch Transformer --target turnover --dataset normal --folds 10`

### Reproducibility notes and limitations
- TimeSeq uses mean-pooled ESM2 per timepoint; incorporating explicit temporal embeddings or lag-derived features is a future extension.
- In turnover-only mode, reporting of loss/inc metrics is suppressed to avoid misinterpretation from untrained heads.
- The AASeq model processes two parallel streams (loss/inc) which enables attention-based interpretability per stream.


