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

    # Create a mapping from Cleaned_Peptidoform to cluster labels
    cluster_map = dict(zip(label_loss_data.index, y_pred))
    df["Cluster"] = df["Cleaned_Peptidoform"].map(cluster_map)

    # Plot clusters
    _plot_clusters(label_loss_data.values, label_incorp_data.values, y_pred, n_clusters)

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


def _plot_clusters(
    X_loss: np.ndarray,
    X_incorp: np.ndarray,
    y_pred: np.ndarray,
    n_clusters: int,
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
        sampling_num (int): Number of peptide samples to plot per cluster. Defaults to 50.
        save_dir (str): Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    cluster_counts = Counter(y_pred)

    # Generate a color map for clusters
    colors = plt.cm.get_cmap("tab10", n_clusters)

    for cluster in range(n_clusters):
        plt.figure(figsize=(10, 6))

        # Extract data belonging to the current cluster (up to sampling_num samples)
        cluster_data_loss = X_loss[y_pred == cluster][:sampling_num]
        cluster_data_incorp = X_incorp[y_pred == cluster][:sampling_num]

        # Plot each peptide's time-series in the cluster
        for i, (series_loss, series_incorp) in enumerate(
            zip(cluster_data_loss, cluster_data_incorp)
        ):
            plt.plot(
                series_loss,
                color=colors(cluster),
                linestyle="-",
                alpha=0.8,
            )
            plt.plot(
                series_incorp,
                color=colors(cluster),
                linestyle="--",
                alpha=0.8,
            )

        # Save the plot
        save_path = os.path.join(save_dir, f"cluster_{cluster}.png")
        plt.title(
            f"Cluster {cluster} - Sample Time Series ({cluster_counts[cluster]} Peptides)"
        )
        plt.xlabel("Time Points")
        plt.ylabel("Values")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
