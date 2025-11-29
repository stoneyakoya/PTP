import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans


def timeseries_clustering(
    df: pd.DataFrame, n_clusters: int, save_path: str
) -> pd.DataFrame:
    """
    Performs time-series clustering on Label Loss and Label Incorporation data, assigns
    each peptide to a cluster, plots the results, and saves the updated DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'Cleaned_Peptidoform', 'timestep',
                           'Target_LabelLoss', and 'Target_LabelIncorporation' columns.
        n_clusters (int): Number of clusters to fit using TimeSeriesKMeans.
        save_path (str): Path to save the DataFrame with newly added cluster information.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'Cluster' column indicating the
                      cluster assignment for each peptide.
    """
    # Pivot tables for Label Loss and Label Incorporation
    label_loss_data = df.pivot(
        index="Cleaned_Peptidoform", columns="timestep", values="Target_LabelLoss"
    ).fillna(0)
    label_incorp_data = df.pivot(
        index="Cleaned_Peptidoform",
        columns="timestep",
        values="Target_LabelIncorporation",
    ).fillna(0)

    # Combine both Label Loss and Incorporation into a single array for clustering
    X = np.hstack([label_loss_data.values, label_incorp_data.values])

    # Perform time-series clustering
    _, y_pred = _time_series_clustering(X, n_clusters)
    
    # Remap cluster IDs to match speed order (0=fastest, n-1=slowest)
    y_pred = _remap_clusters_by_speed(label_incorp_data.values, y_pred, n_clusters)

    # Create a mapping from Cleaned_Peptidoform to cluster labels
    cluster_map = dict(zip(label_loss_data.index, y_pred))
    df["Cluster"] = df["Cleaned_Peptidoform"].map(cluster_map)

    # Plot clusters (individual)
    _plot_clusters(label_loss_data.values, label_incorp_data.values, y_pred, n_clusters, df)
    
    # Plot all clusters in one figure (combined)
    _plot_clusters_combined(label_loss_data.values, label_incorp_data.values, y_pred, n_clusters, df)

    # Save the updated DataFrame
    df.to_csv(save_path, index=False)
    print("DataFrame with assigned cluster information has been saved.")

    return df


def _time_series_clustering(X: np.ndarray, n_clusters: int):
    """
    Fits a TimeSeriesKMeans model to the data and returns the predictions.

    Args:
        X (np.ndarray): The combined time-series data for Label Loss and Incorporation.
        n_clusters (int): Number of clusters for the TimeSeriesKMeans.

    Returns:
        tuple: (X, y_pred) where y_pred is the cluster assignment for each sample.
    """
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=0)
    y_pred = model.fit_predict(X)
    return X, y_pred


def _remap_clusters_by_speed(X_incorp: np.ndarray, y_pred: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Remap cluster IDs so that:
    - Cluster 0 = Very Rapid (fastest)
    - Cluster 1 = Rapid
    - ...
    - Cluster n-1 = Very Slow (slowest)
    
    Args:
        X_incorp (np.ndarray): Array of Label Incorporation time-series data.
        y_pred (np.ndarray): Original cluster labels from K-means.
        n_clusters (int): Number of clusters.
    
    Returns:
        np.ndarray: Remapped cluster labels.
    """
    # Calculate mean early incorporation rate for each cluster
    cluster_speeds = []
    for cluster in range(n_clusters):
        cluster_data = X_incorp[y_pred == cluster]
        early_incorp = np.mean(cluster_data[:, :3])  # First 3 timepoints
        cluster_speeds.append((cluster, early_incorp))
    
    # Sort by speed (descending: fast to slow)
    sorted_clusters = sorted(cluster_speeds, key=lambda x: x[1], reverse=True)
    
    # Create remapping: old_cluster_id -> new_cluster_id
    remap_dict = {}
    for new_id, (old_id, speed) in enumerate(sorted_clusters):
        remap_dict[old_id] = new_id
        print(f"  Old Cluster {old_id} -> New Cluster {new_id} (early incorp: {speed:.4f})")
    
    # Apply remapping
    y_pred_remapped = np.array([remap_dict[label] for label in y_pred])
    
    return y_pred_remapped


def _assign_cluster_names(n_clusters: int) -> dict:
    """
    Assigns cluster names based on cluster IDs (already sorted by speed).
    Cluster 0 = Very Rapid (fastest)
    Cluster n-1 = Very Slow (slowest)
    
    Args:
        n_clusters (int): Number of clusters.
    
    Returns:
        dict: Mapping from cluster ID to descriptive name.
    """
    # Define speed names based on number of clusters
    if n_clusters == 3:
        speed_names = ["Rapid", "Moderate", "Slow"]
    elif n_clusters == 4:
        speed_names = ["Very Rapid", "Rapid", "Moderate", "Slow"]
    elif n_clusters == 5:
        speed_names = ["Very Rapid", "Rapid", "Moderate", "Slow", "Very Slow"]
    elif n_clusters == 6:
        speed_names = ["Very Rapid", "Rapid", "Moderately Rapid", "Moderately Slow", "Slow", "Very Slow"]
    else:
        # For other numbers, use generic numbering
        speed_names = [f"Speed Level {n_clusters-i}" for i in range(n_clusters)]
    
    # Create mapping (cluster ID already matches speed order)
    cluster_names = {}
    for cluster_id in range(n_clusters):
        cluster_names[cluster_id] = speed_names[cluster_id]
        print(f"Cluster {cluster_id} = {speed_names[cluster_id]}")
    
    return cluster_names


def _plot_clusters(
    X_loss: np.ndarray,
    X_incorp: np.ndarray,
    y_pred: np.ndarray,
    n_clusters: int,
    df: pd.DataFrame = None,
    sampling_num: int = 50,
    save_dir: str = "data/data_analyze/clustering",
) -> None:
    """
    Plots sample time-series data for each cluster, including Label Loss (solid line)
    and Label Incorporation (dashed line).

    Args:
        X_loss (np.ndarray): Array of Label Loss time-series data.
        X_incorp (np.ndarray): Array of Label Incorporation time-series data.
        y_pred (np.ndarray): Cluster labels for each peptide.
        n_clusters (int): Number of clusters.
        df (pd.DataFrame): Optional DataFrame to calculate cluster statistics.
        sampling_num (int): Number of peptide samples to plot per cluster. Defaults to 50.
        save_dir (str): Directory to save the plots.
    """
    # n=clusters のサブディレクトリに保存
    save_subdir = os.path.join(save_dir, f"n={n_clusters}")
    os.makedirs(save_subdir, exist_ok=True)
    cluster_counts = Counter(y_pred)

    # Generate a color map for clusters
    colors = plt.cm.get_cmap("tab10", n_clusters)
    
    # Automatically assign cluster names based on cluster IDs (already sorted)
    print("\nCluster names (sorted by speed):")
    print("="*60)
    cluster_names = _assign_cluster_names(n_clusters)
    print("="*60)
    
    # タイムポイントラベル
    time_labels = ["1h", "3h", "6h", "10h", "16h", "24h", "34h", "48h"]

    for cluster in range(n_clusters):
        # フォント設定 - Arial指定
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
        plt.rcParams['ps.fonttype'] = 42   # TrueType fonts for EPS
        
        fig, ax = plt.subplots(figsize=(12, 7))

        # Extract data belonging to the current cluster
        cluster_mask = (y_pred == cluster)
        cluster_data_loss_all = X_loss[cluster_mask]
        cluster_data_incorp_all = X_incorp[cluster_mask]
        
        # Select samples that show cluster characteristics with natural variation
        if len(cluster_data_loss_all) > sampling_num:
            # Combine loss and incorp for analysis
            cluster_combined = np.hstack([cluster_data_loss_all, cluster_data_incorp_all])
            
            # Calculate cluster MEDIAN (more robust than mean)
            cluster_median = np.median(cluster_combined, axis=0)
            
            # Calculate distance to median for each sample
            distances_to_median = np.linalg.norm(cluster_combined - cluster_median, axis=1)
            
            # Normalize distance to [0, 1]
            distances_norm = (distances_to_median - distances_to_median.min()) / (distances_to_median.max() - distances_to_median.min() + 1e-10)
            
            # Select samples with a range of distances:
            # - Some very typical (close to median)
            # - Some with moderate variation
            # This gives a better sense of cluster spread while maintaining clarity
            
            # Stratified sampling: 70% from typical range, 30% from moderate variation
            n_typical = int(sampling_num * 0.7)
            n_varied = sampling_num - n_typical
            
            # Typical samples: closest 30% to median
            typical_threshold = np.percentile(distances_norm, 30)
            typical_mask = distances_norm <= typical_threshold
            typical_indices = np.where(typical_mask)[0]
            
            # Varied samples: middle range (30-70 percentile)
            varied_mask = (distances_norm > typical_threshold) & (distances_norm <= np.percentile(distances_norm, 70))
            varied_indices = np.where(varied_mask)[0]
            
            # Random selection from each group
            if len(typical_indices) >= n_typical:
                selected_typical = np.random.choice(typical_indices, n_typical, replace=False)
            else:
                selected_typical = typical_indices
                
            if len(varied_indices) >= n_varied:
                selected_varied = np.random.choice(varied_indices, n_varied, replace=False)
            else:
                selected_varied = varied_indices
            
            representative_indices = np.concatenate([selected_typical, selected_varied])
            
            cluster_data_loss = cluster_data_loss_all[representative_indices]
            cluster_data_incorp = cluster_data_incorp_all[representative_indices]
        else:
            cluster_data_loss = cluster_data_loss_all
            cluster_data_incorp = cluster_data_incorp_all

        # Plot each peptide's time-series in the cluster
        for i, (series_loss, series_incorp) in enumerate(
            zip(cluster_data_loss, cluster_data_incorp)
        ):
            ax.plot(
                series_loss,
                color=colors(cluster),
                linestyle="-",
                alpha=0.7,
                linewidth=1.2,
            )
            ax.plot(
                series_incorp,
                color=colors(cluster),
                linestyle="--",
                alpha=0.7,
                linewidth=1.2,
            )

        # クラスター名を取得
        cluster_label = cluster_names.get(cluster, f"Cluster {cluster}")
        
        ax.set_title(
            f"Cluster {cluster} {cluster_label}",
            fontsize=24,
            fontweight='bold',
            pad=15
        )
        ax.set_xlabel("Time", fontsize=20, fontweight='bold')
        ax.set_ylabel("Normalized Value", fontsize=20, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # X軸にタイムポイントラベルを設定
        n_points = cluster_data_loss.shape[1] if len(cluster_data_loss) > 0 else len(time_labels)
        ax.set_xticks(range(n_points))
        ax.set_xticklabels(time_labels[:n_points], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # PNG と PDF で保存
        save_path_png = os.path.join(save_subdir, f"cluster_{cluster}_styled.png")
        save_path_pdf = os.path.join(save_subdir, f"cluster_{cluster}_styled.pdf")
        plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=600, bbox_inches='tight', format='pdf')
        print(f"Saved: {save_path_png}")
        print(f"Saved: {save_path_pdf}")
        plt.close()


def _plot_clusters_combined(
    X_loss: np.ndarray,
    X_incorp: np.ndarray,
    y_pred: np.ndarray,
    n_clusters: int,
    df: pd.DataFrame = None,
    sampling_num: int = 50,
    save_dir: str = "data/data_analyze/clustering",
) -> None:
    """
    Plot all clusters in a single figure for easy comparison.
    Layout: 2 rows x 3 columns (or adjusted based on n_clusters)
    """
    save_subdir = os.path.join(save_dir, f"n={n_clusters}")
    os.makedirs(save_subdir, exist_ok=True)
    
    # フォント設定 - Arial指定
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Automatically assign cluster names based on cluster IDs (already sorted)
    print("\nCluster names (sorted by speed):")
    cluster_names = _assign_cluster_names(n_clusters)
    
    # Generate a color map for clusters
    colors = plt.cm.get_cmap("tab10", n_clusters)
    
    # タイムポイントラベル
    time_labels = ["1h", "3h", "6h", "10h", "16h", "24h", "34h", "48h"]
    
    # Calculate layout: rows x cols
    if n_clusters <= 3:
        nrows, ncols = 1, n_clusters
        figsize = (7 * n_clusters, 6)
    elif n_clusters <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    else:
        nrows = (n_clusters + 2) // 3
        ncols = 3
        figsize = (18, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes for easy iteration
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    for cluster in range(n_clusters):
        ax = axes[cluster]
        
        # Extract data belonging to the current cluster
        cluster_mask = (y_pred == cluster)
        cluster_data_loss_all = X_loss[cluster_mask]
        cluster_data_incorp_all = X_incorp[cluster_mask]
        
        # Select representative samples (same logic as individual plots)
        if len(cluster_data_loss_all) > sampling_num:
            cluster_combined = np.hstack([cluster_data_loss_all, cluster_data_incorp_all])
            cluster_median = np.median(cluster_combined, axis=0)
            distances_to_median = np.linalg.norm(cluster_combined - cluster_median, axis=1)
            distances_norm = (distances_to_median - distances_to_median.min()) / (distances_to_median.max() - distances_to_median.min() + 1e-10)
            
            n_typical = int(sampling_num * 0.7)
            n_varied = sampling_num - n_typical
            
            typical_threshold = np.percentile(distances_norm, 30)
            typical_mask = distances_norm <= typical_threshold
            typical_indices = np.where(typical_mask)[0]
            
            varied_mask = (distances_norm > typical_threshold) & (distances_norm <= np.percentile(distances_norm, 70))
            varied_indices = np.where(varied_mask)[0]
            
            if len(typical_indices) >= n_typical:
                selected_typical = np.random.choice(typical_indices, n_typical, replace=False)
            else:
                selected_typical = typical_indices
                
            if len(varied_indices) >= n_varied:
                selected_varied = np.random.choice(varied_indices, n_varied, replace=False)
            else:
                selected_varied = varied_indices
            
            representative_indices = np.concatenate([selected_typical, selected_varied])
            
            cluster_data_loss = cluster_data_loss_all[representative_indices]
            cluster_data_incorp = cluster_data_incorp_all[representative_indices]
        else:
            cluster_data_loss = cluster_data_loss_all
            cluster_data_incorp = cluster_data_incorp_all
        
        # Plot each peptide's time-series in the cluster
        for i, (series_loss, series_incorp) in enumerate(
            zip(cluster_data_loss, cluster_data_incorp)
        ):
            ax.plot(
                series_loss,
                color=colors(cluster),
                linestyle="-",
                alpha=0.6,
                linewidth=0.8,
            )
            ax.plot(
                series_incorp,
                color=colors(cluster),
                linestyle="--",
                alpha=0.6,
                linewidth=0.8,
            )
        
        # クラスター名を取得
        cluster_label = cluster_names.get(cluster, f"Cluster {cluster}")
        
        ax.set_title(
            f"Cluster {cluster} {cluster_label}",
            fontsize=16,
            fontweight='bold',
            pad=10
        )
        ax.set_xlabel("Time", fontsize=12, fontweight='bold')
        ax.set_ylabel("Normalized Value", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # X軸にタイムポイントラベルを設定
        n_points = cluster_data_loss.shape[1] if len(cluster_data_loss) > 0 else len(time_labels)
        ax.set_xticks(range(n_points))
        ax.set_xticklabels(time_labels[:n_points], fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save combined figure
    save_path_png = os.path.join(save_subdir, f"all_clusters_combined.png")
    save_path_pdf = os.path.join(save_subdir, f"all_clusters_combined.pdf")
    plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(save_path_pdf, dpi=600, bbox_inches='tight', format='pdf')
    print(f"Saved combined plot: {save_path_png}")
    print(f"Saved combined plot: {save_path_pdf}")
    plt.close()


if __name__ == "__main__":
    """
    直接実行用のエントリーポイント
    既存のクラスタリング結果から画像のみを再生成します
    
    使い方:
        python clustering_data.py [--recluster] [n_clusters]
        
    例:
        python clustering_data.py                # 既存のクラスタ結果を使用（n=5）
        python clustering_data.py 4              # 既存のクラスタ結果を使用（n=4）
        python clustering_data.py --recluster    # クラスタリングを再実行（n=5）
        python clustering_data.py --recluster 4  # クラスタリングを再実行（n=4）
    """
    import sys
    
    # Parse arguments
    recluster = False
    n_clusters = 5
    
    args = [arg for arg in sys.argv[1:] if arg != '--recluster']
    if '--recluster' in sys.argv:
        recluster = True
    
    if len(args) > 0:
        n_clusters = int(args[0])
    
    # Check for existing clustered data
    clustered_data_path = f"data/tmp/clustered_peptides_n{n_clusters}.csv"
    
    if os.path.exists(clustered_data_path) and not recluster:
        # Use existing clustering results (faster, consistent)
        print(f"Using existing clustering results from: {clustered_data_path}")
        print("(Use --recluster flag to recompute clustering)")
        df_clustered = pd.read_csv(clustered_data_path)
        
        # Check if Cluster column exists
        if 'Cluster' not in df_clustered.columns:
            print("Error: 'Cluster' column not found in existing data!")
            print("Running clustering from scratch...")
            recluster = True
        else:
            # Extract data for plotting
            label_loss_data = df_clustered.pivot(
                index="Cleaned_Peptidoform", columns="timestep", values="Target_LabelLoss"
            ).fillna(0)
            label_incorp_data = df_clustered.pivot(
                index="Cleaned_Peptidoform",
                columns="timestep",
                values="Target_LabelIncorporation",
            ).fillna(0)
            
            # Get cluster labels
            cluster_map = df_clustered.groupby('Cleaned_Peptidoform')['Cluster'].first()
            y_pred = cluster_map.values
            
            # Plot individual clusters
            _plot_clusters(
                label_loss_data.values, 
                label_incorp_data.values, 
                y_pred, 
                n_clusters,
                df_clustered
            )
            
            # Plot combined clusters
            _plot_clusters_combined(
                label_loss_data.values,
                label_incorp_data.values,
                y_pred,
                n_clusters,
                df_clustered
            )
            
            print(f"\n{'='*60}")
            print(f"Images regenerated successfully!")
            print(f"Images saved to: data/data_analyze/clustering/n={n_clusters}/")
            print(f"{'='*60}")
    
    if not os.path.exists(clustered_data_path) or recluster:
        # Perform clustering from scratch
        print(f"Performing clustering with n_clusters={n_clusters}")
        
        # データファイルのパス
        data_path = "data/tmp/df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag.csv"
        
        # データの読み込み
        print(f"Loading data from: {data_path}")
        if not os.path.exists(data_path):
            print(f"Error: {data_path} not found!")
            print("Please run preprocess_data.py first.")
            sys.exit(1)
        
        df = pd.read_csv(data_path)
        
        # クラスタリング実行
        df_clustered = timeseries_clustering(
            df=df,
            n_clusters=n_clusters,
            save_path=clustered_data_path,
        )
        
        print(f"\n{'='*60}")
        print(f"Clustering complete!")
        print(f"Results saved to: {clustered_data_path}")
        print(f"Images saved to: data/data_analyze/clustering/n={n_clusters}/")
        print(f"{'='*60}")
