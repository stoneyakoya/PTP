### Dataset analysis (cluster dynamics)

目的: `sampling` / `normal` / `customized` の各データセット分割で、クラスタ別の時間変化が期待どおり（例: very rapid など）になっているかを可視化・要約する。

出力:
- 図: `reports/figures/dataset/analysis_{DATASET}/timecourses_{target}_fold{K}.png`
- 表: `reports/tables/t50_by_cluster_{DATASET}_fold{K}.csv`, `reports/tables/cluster_counts_{DATASET}_fold{K}.csv`

実行例:
```bash
python -m src.analyze.dataset_analysis --dataset_base data/dataset --dataset sampling --fold 0
python -m src.analyze.dataset_analysis --dataset normal --fold 0

# クラスタ番号が fast→slow になっているか検証（t50順）
python -m src.analyze.verify_cluster_order --dataset sampling --fold 0
python -m src.analyze.verify_cluster_order --dataset normal --fold 0 --write_mapping
```

中身:
- cluster-wise timecourse (mean±95%CI) を loss/incorporation/turnover で出力
- turnover の 0.5 到達時間（t50）をペプチド単位で推定し、クラスタごとに平均・標準偏差を集計
- クラスタごとのユニークペプチド数も表で出力
 - `cluster_speed_rank_{DATASET}_fold{K}.csv` で平均t50と順位を出力。`--write_mapping` を付けると対応表JSON（current→fast順）をfoldディレクトリに保存


