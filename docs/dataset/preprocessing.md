### 前処理（ペプチド集約・正規化・クリッピング）

#### 対象コード
- `scripts/preprocess_data.py` の `PrepareTarget`
  - `summarize_peptide`
  - `filter_unique_peptides`
  - `data_normalize`

#### パイプライン概要
1. `summarize_peptide`
   - 原始 CSV（ヘッダ行に注意）を読み込み
   - ペプチド表記から括弧内修飾を削除し `Cleaned_Peptidoform` を作成
   - レプリケート平均（ピボット→平均）
   - タイムポイントごとに `TurnoverRate_t = Incorporation_t / (Loss_t + Incorporation_t)` を計算
   - NaN 行を除去

2. 同一タンパク質内ユニークペプチドの抽出
   - `PrepareInput.run` 内で、`Cleaned_Peptidoform` が対応する `Protein Sequence` にちょうど1回だけ出現する行のみ残す
   - 同一タンパク質内で複数回出現する（重複位置を持つ）ペプチドは除外（帰属の曖昧性を防止）

3. `data_normalize`
   - 方法: `MinMax` または `clip_arch`
   - デフォルト: `clip_arch`（1h〜48h の各メトリクスを [0,1] にクリップ）
   - 0h, infin.h は分布が特殊なため一部正規化対象外

#### クリッピング方針
- タンパク質代謝回転の比率系指標は外れ値の影響が大きいため、[0,1] へのクリップをデフォルトとする
- 必要に応じ `MinMax` を選択可能（再現性のためシード固定推奨）

#### 可視化
- `timeseries_info(metric, normalize_method)` で箱ひげ図を保存（`data/data_analyze/`）

#### 実行例
```bash
python scripts/preprocess_data.py --raw_csv data/raw/pSILAC_TMT.csv
```

#### `python scripts/preprocess_data.py` の出力物
- `reports/tables/dataset_stats_prepost.csv`
  - raw と aggregated（集約後）の統計比較テーブル
- `data/interim/`（取得をスキップしない場合に順次作成・再利用）
  - `df_with_ProteinSeq.csv`
  - `df_with_ProteinSeq_ProteinEmbedPath.csv`
  - `df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath.csv`
  - `df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag.csv`
- 埋め込み（取得をスキップしない場合）
  - `data/protein_embeddings/{UniProt}.npz`
  - `data/peptide_embeddings/{peptide}_{UniProt}.npz`
- UniProt キャッシュ（取得をスキップしない場合）
  - `data/uniprot/{UniProt}.json`

注: `--skip_seq_embed` を付けると UniProt 取得と埋め込み生成は行いません。既存の `data/interim/*.csv` があればそれを再利用し、上記のテーブルのみを作成します。可視化画像は `scripts/create_dataset.py` でまとめて出力されます。

