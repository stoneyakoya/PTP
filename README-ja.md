# PTP（pSILAC-TMT パイプライン）

## 概要

PTP は pSILAC-TMT データの前処理から学習・評価までを一貫して行うパイプラインです。

- 前処理スクリプトでのデータ整形
- 学習・評価用スクリプト
- 実験系および計算パイプラインのユーティリティ

## 前提条件

- Python 3.11.4
- Git
- conda

## セットアップ手順

1. リポジトリを取得

   ```bash
   git clone https://github.com/stoneyakoya/PTP
   cd PTP
   ```

2. 仮想環境を作成

   ```bash
   conda create -p ./venv python==3.11.4
   ```

3. 仮想環境を有効化

   ```bash
   conda activate ./venv
   ```

4. PyTorch のインストール

   PyTorch 公式サイトの環境に応じたコマンドを参照してください（`https://pytorch.org/get-started/locally/`）。

5. 依存パッケージのインストール

   ```bash
   pip install -r requirements.txt
   ```

## データ準備

1. Jana Zecha ら（Nature Communications, 13:165, 2022; PXD023218）の補足データから pSILAC-TMT データを取得します。
2. Supplementary Dataset 1 を CSV としてエクスポートします。
3. `data/raw` に配置します。推奨ファイル名：

   ```bash
   pSILAC_TMT.csv
   ```

注意: データの利用・共有は元論文の規約に従ってください。

## 前処理

CSV を `data/raw` 配下に置いたら、以下を実行します。

```bash
python scripts/preprocess_data.py --raw_csv data/raw/pSILAC_TMT.csv
```

- ターンオーバー関連の集計・正規化（`PrepareTarget`）
- UniProt からのタンパク質配列取得、ESM2 による埋め込み生成、ペプチド埋め込み抽出、ラグ特徴量作成（`PrepareInput`）
- 可視化は `data/data_analyze/` に保存されます

補助: デバッグ出力は `--debug` で有効化できます。

## データセット作成（分割・クラスタリングなど）

以下はクラスター付与、分割（通常分割・KFold など）を行い、`data/dataset/` 以下に保存します。

```bash
python scripts/create_dataset.py
```

- 非対話で動作するように調整済み
- 生成物の一部:
  - `data/dataset/normal`（train/val/test）
  - `data/dataset/sampling`（各クラスタサイズ調整後の分割）
  - `data/dataset/customized`（任意クラスタ縮小）

## 学習と評価

```bash
python src/train_test.py
```

- 既定ではモデルは `data/models`、結果は `data/results` に保存されます
- コマンドライン引数で学習率・バッチサイズ・モデル構成などを変更可能

例：

```bash
python src/train_test.py --batch_size 8 --lr 0.001 --num_epochs 100
```

主な引数は `README.md` の「Adjustable Parameters」を参照してください。

## プロジェクト構成

```plaintext
.
├── data/             # データ（raw/processed）
├── models/           # 学習済みモデル
├── README.md         # 英語版 README
├── README-ja.md      # 日本語版 README（本ファイル）
├── requirements.txt  # 依存関係
├── scripts/          # 前処理・データセット作成などのスクリプト群
├── src/              # 学習・評価・モデル定義などのソースコード
└── venv/             # 仮想環境（VC 管理対象外）
```

## ライセンス・データ利用

- 本リポジトリのコードライセンスに従ってください
- データの取り扱いは必ず元データの利用規約に従ってください

