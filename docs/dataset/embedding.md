### 埋め込み生成（ESM2）

この文書では、ペプチドから対応するタンパク質配列を取得し、ESM2 によりタンパク質埋め込みを生成し、対応するペプチド領域の埋め込みを抽出するまでの手順をまとめます。

#### 対象コード
- `scripts/preprocess_data.py` の `PrepareInput`
  - `get_protein_sequence`
  - `generate_protein_embedding`
  - `extract_peptide_embedding`

#### 前提
- 入力は `PrepareTarget` により整形済みの DataFrame（列例: `Cleaned_Peptidoform`, `UniProt IDs`, `Protein names` など）
- `Cleaned_Peptidoform` は修飾括弧を除去済み
- ユニークペプチドのみを使用（共有ペプチドは除外）

#### 手順概要
1. タンパク質配列の取得 (`get_protein_sequence`)
   - `UniProt IDs` の最初の ID を代表として使用
   - 事前ダウンロード済み JSON（`data/uniprot/*.json`）から配列を抽出、なければ UniProt API から取得
   - 配列未取得行は除外

2. タンパク質埋め込みの生成 (`generate_protein_embedding`)
   - モデル: `esm2_t33_650M_UR50D`
   - 各配列をトークナイズし、残基レベル埋め込み（[CLS], 各残基, [EOS]）を `.npz` で保存（`data/protein_embeddings/{UniProt}.npz`）

3. ペプチド領域埋め込みの抽出 (`extract_peptide_embedding`)
   - 対応するタンパク質埋め込みをロード
   - 文字列検索でペプチド開始位置を特定（`protein_sequence.find(peptidoform)`）
   - ESM のトークン配列は [CLS] オフセット分ずれるため、`[start+1 : end+1]` をスライス
   - `[CLS]` ベクトルを先頭に付与して保存（`data/peptide_embeddings/{peptide}_{UniProt}.npz`）

#### 注意点 / ベストプラクティス
- 共有ペプチドは原則除外（ユニークペプチドのみ）。配列が複数タンパク質に一致する場合は学習データから外す。
- 長さ不整合（`len(protein_seq)+2 != len(residue_embeddings)`）は警告出力。
- 既存ファイルは再計算せずスキップして計算を節約。

#### 実行例
```bash
python scripts/preprocess_data.py --raw_csv data/raw/pSILAC_TMT.csv
```

中間生成物は `data/tmp/` に保存されます。



