### データ分割（fold）ポリシーと統計

#### 対象コード
- `scripts/create_dataset.py`
  - `normal_split_dataset`
  - `enhanced_k_fold_split_dataset`
  - `sampling_by_cluster`, `reduce_cluster_size_by_peptide`

#### 分割の原則
- タンパク質単位で分割（同一タンパク質が train/val/test 間で重複しない）
- 共有ペプチドは前処理で除外済み（ユニークペプチドのみ）
- 埋め込みファイルが存在しない行は除外

#### K-Fold 構成（推奨）
- `n_splits=10`, `train:val:test = 7:2:1` を各 fold 内で達成
- 各 fold で `train_fold.csv`, `validation_fold.csv`, `test_fold.csv` を保存
- `stats_fold.csv` には以下を記録
  - タンパク質数（train/val/test）
  - ユニークペプチド数（train/val/test）
  - 行数（train/val/test）
  - クラスタ別のユニークペプチド数（train/val/test）

全 fold の一覧は `kfold_summary.csv` に集約されます。

#### クラスタ分布のバランス
- `timeseries_clustering` により `Cluster` 列を付与した後、fold ごとのクラスタ分布を `stats_fold.csv` に記録
- 需要に応じて `sampling_by_cluster` で最小クラスタ規模に合わせたサンプリングを実施
- 特定クラスタを意図的に縮小する場合は `reduce_cluster_size_by_peptide` を使用

#### 実行例
```bash
python scripts/create_dataset.py
```

#### `python scripts/create_dataset.py` の出力物
- `reports/figures/dataset/clustering/`
  - 各クラスタのサンプル時系列図（Cluster 0..K-1）
- `reports/figures/dataset/timeseries_boxplots_clip_arch.png`
  - Label loss / Label incorporation / Turnover ratio をまとめた3段パネル図（clip_arch）
- `reports/tables/`
  - `dataset_characteristics.csv`（クラスタ付与後のユニークタンパク質数・ユニークペプチド数・クラスタ別ユニークペプチド数）
  - `kfold_summary.csv`（foldごとのタンパク質数/ペプチド数/行数およびクラスタ別ユニークペプチド数）
  - `fold_*/stats_fold.csv`（各foldの詳細統計、上記と同等の単体ファイル）
- `data/dataset/normal/fold_*/`
  - `train_fold.csv`, `validation_fold.csv`, `test_fold.csv`
- （オプション）`data/dataset/sampling/` および `data/dataset/customized/`
  - sampling・customized 設定時の同様の出力

