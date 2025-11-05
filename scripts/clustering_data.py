import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans


def timeseries_clustering(
    df: pd.DataFrame,
    n_clusters: int,
    save_path: str,
    relabel_by_speed: bool = True,
    relabel_metric: str = "avg_turnover",  # "avg_turnover" | "avg_incorp" | "initial_slope" | "delta" | "k_loss" | "k_incorp" | "k_mean"
    ascending: bool = False,  # False: high->low
    feature_mode: str = "both",  # "both" | "incorp"
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

    # Ensure time columns are sorted numerically (e.g., 1,3,6,...,48)
    try:
        loss_cols_sorted = sorted([int(c) for c in label_loss_data.columns])
        incorp_cols_sorted = sorted([int(c) for c in label_incorp_data.columns])
        label_loss_data = label_loss_data[loss_cols_sorted]
        label_incorp_data = label_incorp_data[incorp_cols_sorted]
    except Exception:
        # If conversion fails, fall back to existing order
        pass

    # Select features for clustering
    if feature_mode == "incorp":
        X = label_incorp_data.values
    else:
        # Combine both Label Loss and Incorporation
        X = np.hstack([label_loss_data.values, label_incorp_data.values])

    # Perform time-series clustering
    _, y_pred = _time_series_clustering(X, n_clusters)

    # Optionally relabel clusters by a speed/kinetic score
    if relabel_by_speed:
        # Compute per-peptide score
        if relabel_metric == "delta":
            # (inc_end - inc_start) + (loss_start - loss_end)
            inc_start = label_incorp_data.values[:, 0]
            inc_end = label_incorp_data.values[:, -1]
            loss_start = label_loss_data.values[:, 0]
            loss_end = label_loss_data.values[:, -1]
            peptide_scores = (inc_end - inc_start) + (loss_start - loss_end)
        elif relabel_metric in ("avg_turnover", "initial_slope"):
            # Build turnover matrix = inc/(inc+loss)
            inc_vals = label_incorp_data.values
            loss_vals = label_loss_data.values
            denom = np.clip(inc_vals + loss_vals, a_min=1e-8, a_max=None)
            turnover = inc_vals / denom
            if relabel_metric == "avg_turnover":
                peptide_scores = np.nanmean(turnover, axis=1)
            else:  # initial_slope
                # slope over the first 3 time points using linear regression
                try:
                    time_cols = label_incorp_data.columns
                    x = np.array([int(t) for t in time_cols], dtype=float)
                except Exception:
                    x = np.arange(turnover.shape[1], dtype=float)
                end_idx = min(3, turnover.shape[1])
                x_fit = x[:end_idx]
                slopes = []
                for row in turnover[:, :end_idx]:
                    try:
                        m, _ = np.polyfit(x_fit, row, deg=1)
                    except Exception:
                        m = 0.0
                    slopes.append(m)
                peptide_scores = np.array(slopes, dtype=float)
        elif relabel_metric == "avg_incorp":
            # Average incorporation level as speed proxy
            peptide_scores = np.nanmean(label_incorp_data.values, axis=1)
        elif relabel_metric in ("k_loss", "k_incorp", "k_mean") and {
            "loss_K",
            "incorporation_K",
        }.issubset(df.columns):
            # 1 peptide : 1 K（同一ペプチドで重複するので代表統計を使用）
            k_df = (
                df[["Cleaned_Peptidoform", "loss_K", "incorporation_K"]]
                .drop_duplicates()
                .set_index("Cleaned_Peptidoform")
                .reindex(label_loss_data.index)
            )
            if relabel_metric == "k_loss":
                peptide_scores = k_df["loss_K"].astype(float).values
            elif relabel_metric == "k_incorp":
                peptide_scores = k_df["incorporation_K"].astype(float).values
            else:
                peptide_scores = (
                    0.5 * k_df["loss_K"].astype(float).values
                    + 0.5 * k_df["incorporation_K"].astype(float).values
                )
        else:
            # Fallback to delta
            inc_start = label_incorp_data.values[:, 0]
            inc_end = label_incorp_data.values[:, -1]
            loss_start = label_loss_data.values[:, 0]
            loss_end = label_loss_data.values[:, -1]
            peptide_scores = (inc_end - inc_start) + (loss_start - loss_end)

        # Median score per original cluster label
        scores_per_cluster = {}
        for c in range(n_clusters):
            cluster_scores = peptide_scores[y_pred == c]
            if len(cluster_scores) == 0:
                # push empty clusters to the end/start depending on order
                scores_per_cluster[c] = np.nan
            else:
                scores_per_cluster[c] = float(np.median(cluster_scores))

        # Sort clusters by score
        # Handle NaN by placing them at the end
        valid_items = [(c, s) for c, s in scores_per_cluster.items() if not np.isnan(s)]
        invalid_items = [(c, s) for c, s in scores_per_cluster.items() if np.isnan(s)]
        ordered_valid = sorted(valid_items, key=lambda x: x[1], reverse=not ascending)
        ordered = [c for c, _ in ordered_valid] + [c for c, _ in invalid_items]
        old_to_new = {old: new for new, old in enumerate(ordered)}
        y_pred = np.array([old_to_new[int(lbl)] for lbl in y_pred])

    # Create a mapping from Cleaned_Peptidoform to cluster labels
    cluster_map = dict(zip(label_loss_data.index, y_pred))
    df["Cluster"] = df["Cleaned_Peptidoform"].map(cluster_map)

    # Plot clusters
    _plot_clusters(label_loss_data.values, label_incorp_data.values, y_pred, n_clusters)

    # Save the updated DataFrame
    df.to_csv(save_path, index=False)
    print("DataFrame with assigned cluster information has been saved.")

    return df


def relabel_clusters_in_df(
    df: pd.DataFrame,
    relabel_metric: str = "avg_turnover",
    ascending: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Relabel existing df["Cluster"] by a kinetic metric using median-per-cluster scores.
    Returns (updated_df, old_to_new_mapping).
    """
    # Build pivots
    label_loss_data = df.pivot(
        index="Cleaned_Peptidoform", columns="timestep", values="Target_LabelLoss"
    ).fillna(0)
    label_incorp_data = df.pivot(
        index="Cleaned_Peptidoform",
        columns="timestep",
        values="Target_LabelIncorporation",
    ).fillna(0)
    # Sort columns numerically if possible
    try:
        loss_cols_sorted = sorted([int(c) for c in label_loss_data.columns])
        incorp_cols_sorted = sorted([int(c) for c in label_incorp_data.columns])
        label_loss_data = label_loss_data[loss_cols_sorted]
        label_incorp_data = label_incorp_data[incorp_cols_sorted]
    except Exception:
        pass

    # Compute peptide scores 
    inc_vals = label_incorp_data.values
    loss_vals = label_loss_data.values
    denom = np.clip(inc_vals + loss_vals, a_min=1e-8, a_max=None)
    turnover = inc_vals / denom
    if relabel_metric == "avg_turnover":
        peptide_scores = np.nanmean(turnover, axis=1)
    elif relabel_metric == "avg_incorp":
        peptide_scores = np.nanmean(inc_vals, axis=1)
    elif relabel_metric == "initial_slope":
        try:
            time_cols = label_incorp_data.columns
            x = np.array([int(t) for t in time_cols], dtype=float)
        except Exception:
            x = np.arange(turnover.shape[1], dtype=float)
        end_idx = min(3, turnover.shape[1])
        x_fit = x[:end_idx]
        slopes = []
        for row in turnover[:, :end_idx]:
            try:
                m, _ = np.polyfit(x_fit, row, deg=1)
            except Exception:
                m = 0.0
            slopes.append(m)
        peptide_scores = np.array(slopes, dtype=float)
    else:
        peptide_scores = np.nanmean(turnover, axis=1)

    # Align cluster labels on the pivot index
    cluster_map = df.drop_duplicates("Cleaned_Peptidoform").set_index(
        "Cleaned_Peptidoform"
    )["Cluster"].to_dict()
    y_pred = np.array([cluster_map.get(p, -1) for p in label_loss_data.index])
    valid_mask = y_pred >= 0
    y_pred = y_pred[valid_mask]
    peptide_scores = peptide_scores[valid_mask]

    # Compute cluster median scores and order
    scores_per_cluster = {}
    for c in np.unique(y_pred):
        scores = peptide_scores[y_pred == c]
        if len(scores) == 0:
            scores_per_cluster[int(c)] = np.nan
        else:
            scores_per_cluster[int(c)] = float(np.nanmedian(scores))
    valid_items = [(c, s) for c, s in scores_per_cluster.items() if not np.isnan(s)]
    invalid_items = [(c, s) for c, s in scores_per_cluster.items() if np.isnan(s)]
    ordered_valid = sorted(valid_items, key=lambda x: x[1], reverse=not ascending)
    ordered = [c for c, _ in ordered_valid] + [c for c, _ in invalid_items]
    old_to_new = {old: new for new, old in enumerate(ordered)}

    # Apply mapping
    df = df.copy()
    df["Cluster"] = df["Cluster"].map(lambda x: old_to_new.get(int(x), x))
    return df, old_to_new


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
    save_dir: str = "reports/figures/dataset/clustering",
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
        print(f"Saved clustering plot: {save_path}")
        plt.close()


def plot_clusters_from_df(
    df: pd.DataFrame,
    n_clusters: int | None = None,
    save_dir: str = "reports/figures/dataset/clustering",
    sampling_num: int = 50,
) -> None:
    """
    Regenerate clustering plots from a DataFrame that already has a 'Cluster' column,
    without re-running k-means.
    """
    # Build pivots for loss/incorporation (index = peptide, columns = timestep)
    label_loss_data = df.pivot(
        index="Cleaned_Peptidoform", columns="timestep", values="Target_LabelLoss"
    ).fillna(0)
    label_incorp_data = df.pivot(
        index="Cleaned_Peptidoform",
        columns="timestep",
        values="Target_LabelIncorporation",
    ).fillna(0)

    # Sort time columns numerically if possible
    try:
        loss_cols_sorted = sorted([int(c) for c in label_loss_data.columns])
        incorp_cols_sorted = sorted([int(c) for c in label_incorp_data.columns])
        label_loss_data = label_loss_data[loss_cols_sorted]
        label_incorp_data = label_incorp_data[incorp_cols_sorted]
    except Exception:
        pass

    # Align y_pred to pivot index
    cluster_map = df.drop_duplicates("Cleaned_Peptidoform").set_index(
        "Cleaned_Peptidoform"
    )["Cluster"].to_dict()
    y_pred = np.array([cluster_map.get(p, -1) for p in label_loss_data.index])
    # Filter out entries with invalid cluster labels
    valid_mask = y_pred >= 0
    X_loss = label_loss_data.values[valid_mask]
    X_incorp = label_incorp_data.values[valid_mask]
    y_pred = y_pred[valid_mask]

    if n_clusters is None:
        try:
            n_clusters = int(df["Cluster"].nunique())
        except Exception:
            n_clusters = 0

    _plot_clusters(
        X_loss=X_loss,
        X_incorp=X_incorp,
        y_pred=y_pred.astype(int),
        n_clusters=n_clusters,
        sampling_num=sampling_num,
        save_dir=save_dir,
    )
