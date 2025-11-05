# プロジェクト運用ガイドライン（docs/guideline)

## 目的
このドキュメントは、本プロジェクトの一連の流れ（データ取得→前処理→データセット作成→学習/評価→可視化/解析→ベースライン）と、実行コマンド・出力場所の要点を簡潔にまとめたものです。詳細は `README.md` / `README-ja.md` も参照してください。

## 全体フロー
1. データ取得（pSILAC-TMT, UniProt）
2. 前処理（特徴量作成・正規化・埋め込み生成）
3. データセット作成（分割・クラスタリング・サンプリング）
4. 学習・評価（モデル選択、保存、メトリクス算出）
5. 解析・可視化（クラスタ傾向・寄与・アトリビューションなど）
6. ベースライン評価（ツリー系モデルによる比較）

## 前提条件
- Python 3.11.4（`conda create -p ./venv python==3.11.4`）
- PyTorch（環境に合わせて公式手順を参照）
- 依存関係：`pip install -r requirements.txt`

## データ配置
- 原データ（pSILAC-TMT CSV）: `data/raw/pSILAC_TMT.csv`
- 参考: UniProt 由来の配列等は前処理で取得・生成されます

## コマンド早見表
- 前処理
  ```bash
  python scripts/preprocess_data.py --raw_csv data/raw/pSILAC_TMT.csv
  ```
  - 出力の一部: `data/data_analyze/`, 埋め込み/特徴量, 中間ファイル など

- データセット作成（分割・クラスタリング等）
  ```bash
  python scripts/create_dataset.py
  ```
  - 代表的な出力:
    - `data/dataset/normal/`（通常の train/val/test）
    - `data/dataset/sampling/`（クラスタバランス調整後の分割）
    - `data/dataset/customized/`（サイズ調整済みカスタム）

- 学習と評価（深層学習系）
  ```bash
  python src/train_test.py --batch_size 8 --lr 0.001 --num_epochs 100
  ```
  - 出力: モデル `data/models/…`, 結果 `data/results/…`, 図 `data/plts/…`
  - その他の引数は `README.md` の表を参照

- ベースライン（木系: XGBoost / RandomForest）
  ```bash
  python src/baseline.py
  ```
  - デフォルト: `input_type = "AAC"`, 10-fold 実行
  - 出力: `data/results/00_baseline_AAC/{results.csv, avg_results.csv, std_results.csv}`
  - ESM2 を使う場合はファイル内の `input_type = "ESM2"` に変更

## ディレクトリ要点
- `scripts/`: 前処理・データセット作成ユーティリティ
- `src/datasets/`: データローダ（TimeSeq, AAC, AASeq など）
- `src/models/`: モデル定義（LSTM/GRU/Transformer 等）
- `src/train_test.py`: 学習/評価エントリポイント
- `src/analyze/`: 解析スクリプト（クラスタ付与、アトリビューション可視化等）
- `src/baseline.py`: XGBoost/RF によるベースライン
- `data/`: 原データ、加工済データ、分割済データ、可視化、結果など

## 運用上のヒント
- 実験は `result_save_dir` を分け、`README.md` に載っている引数で再現性を担保
- 乱数シードや fold 設定はスクリプト内の定数で管理（必要なら引数化）
- 大きな前処理（埋め込み生成など）はキャッシュ化し、パスを固定

## 参考
- `README.md`（英語）／`README-ja.md`（日本語）
- `data/data_analyze/` の図表・中間物
- `data/results/` のメトリクス・予測物
