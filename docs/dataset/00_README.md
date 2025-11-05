### データセット作成ドキュメント概要

本フォルダには、pSILAC-TMT データの前処理、ESM2 による埋め込み生成、学習用データセットの分割ポリシーを日本語で整理しています。

- `embedding.md`: ペプチド→タンパク質配列取得→ESM2 埋め込み生成→対応ペプチド領域の抽出
- `preprocessing.md`: ペプチドサマリー作成、同一タンパク質内ユニーク化、正規化（クリッピング/MinMax）
- `splitting.md`: fold 構成、タンパク質単位分割、クラスタ分布の統計

出力先ポリシー：
- 中間CSV: `data/interim/`
- 可視化: `reports/figures/dataset/` および `reports/figures/dataset/clustering/`
- 集計テーブル: `reports/tables/`

コードの実装は主に `scripts/preprocess_data.py` と `scripts/create_dataset.py` にあります。

