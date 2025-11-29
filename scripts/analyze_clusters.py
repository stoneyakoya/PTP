# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/tmp/clustered_peptides_n5.csv')

# 各クラスターのLabel incorporation平均を計算
print('Cluster characteristics:')
print('='*70)

cluster_stats = []
for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    incorp_mean = cluster_data['Target_LabelIncorporation'].mean()
    loss_mean = cluster_data['Target_LabelLoss'].mean()
    turnover_mean = cluster_data['Target_TurnoverRate'].mean()
    n_peptides = cluster_data['Cleaned_Peptidoform'].nunique()
    
    cluster_stats.append({
        'cluster': cluster,
        'incorp_mean': incorp_mean,
        'loss_mean': loss_mean,
        'turnover_mean': turnover_mean,
        'n_peptides': n_peptides
    })
    
    print(f'Cluster {cluster}:')
    print(f'  Label Incorporation mean: {incorp_mean:.4f}')
    print(f'  Label Loss mean:          {loss_mean:.4f}')
    print(f'  Turnover Rate mean:       {turnover_mean:.4f}')
    print(f'  # of peptides:            {n_peptides}')
    print()

# ターンオーバー率でソート
print('='*70)
print('Clusters sorted by Turnover Rate (slow to rapid):')
print('='*70)
sorted_clusters = sorted(cluster_stats, key=lambda x: x['turnover_mean'])

speed_names = ['Very Slow', 'Slow', 'Moderate', 'Rapid', 'Very Rapid']
for i, stat in enumerate(sorted_clusters):
    print(f'{speed_names[i]:12} <- Cluster {stat["cluster"]}: '
          f'TurnoverRate={stat["turnover_mean"]:.4f}, '
          f'n_peptides={stat["n_peptides"]}')

