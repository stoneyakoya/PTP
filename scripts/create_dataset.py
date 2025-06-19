import os
import random

import numpy as np
import pandas as pd
from clustering_data import timeseries_clustering
from preprocess_data import PrepareInput, PrepareTarget
from sklearn.model_selection import KFold


def preprocess_main():
    """Main preprocessing function."""
    # Prepare input & target data
    raw_data_path = "data/raw/pSILAC_TMT.csv"
    target_df = PrepareTarget(raw_data_path).run()
    input_target_df = PrepareInput(target_df).run()

    # Add cluster information
    n_clusters = 5
    save_path = "data/tmp/input_target_df_add_cluster_info.csv"
    if not os.path.exists(save_path):
        print("Creating input_target_df_add_cluster_info...")
        input_target_df_add_cluster_info = timeseries_clustering(
            df=input_target_df,
            n_clusters=n_clusters,
            save_path=save_path,
        )
    else:
        print(f"Loading existing file: {save_path}")
        input_target_df_add_cluster_info = pd.read_csv(save_path)

    # Remove rows with NaN in Peptido_Embedding_Path
    input_target_df_add_cluster_info = input_target_df_add_cluster_info[
        input_target_df["Peptido_Embedding_Path"].apply(
            lambda x: isinstance(x, str) and os.path.exists(x)
        )
    ].reset_index(drop=True)

    print(input_target_df_add_cluster_info)
    print(input_target_df_add_cluster_info.columns)
    input()

    # Data info: check the number of unique peptides in each cluster
    for cluster_id, group in input_target_df_add_cluster_info.groupby("Cluster"):
        unique_peptide_count = group["Cleaned_Peptidoform"].nunique()
        print(
            f"Cluster={cluster_id}, Number of unique peptides= {unique_peptide_count}"
        )

    # Normal dataset
    dataset_dir = "data/dataset/normal"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Number of Proteins in the dataset
    print(
        f"Number of unique proteins in the entire DataFrame: {len(input_target_df_add_cluster_info['UniProt IDs'].str.split(';').str[0].unique())}"
    )
    # Number of Peptides
    print(
        f"Number of unique peptides in the entire DataFrame: {len(input_target_df_add_cluster_info['Cleaned_Peptidoform'].unique())}"
    )

    enhanced_k_fold_split_dataset(input_target_df_add_cluster_info, dataset_dir)

    # Sampling dataset
    dataset_dir = "data/dataset/sampling"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    sampled_df = sampling_by_cluster(input_target_df_add_cluster_info)

    # Number of Proteins in the dataset
    print(
        f"Number of unique proteins in the entire DataFrame: {len(sampled_df['UniProt IDs'].str.split(';').str[0].unique())}"
    )
    # Number of Peptides
    print(
        f"Number of unique peptides in the entire DataFrame: {len(sampled_df['Cleaned_Peptidoform'].unique())}"
    )
    enhanced_k_fold_split_dataset(sampled_df, dataset_dir)

    # Customized dataset: reduce target clusters
    clusters_to_reduce = []
    if not clusters_to_reduce:
        print("choose cluster(s) to reduce")
    dataset_dir = "data/dataset/customized"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    customized_df = reduce_cluster_size_by_peptide(
        input_target_df_add_cluster_info,
        clusters_to_reduce=clusters_to_reduce,
    )

    normal_split_dataset(customized_df, dataset_dir)


def normal_split_dataset(
    df: pd.DataFrame,
    save_dir: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Splits the dataset into training, validation, and test sets (by protein).
    Ensures that the same protein does not appear in both train/validation and test sets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        save_dir (str): Directory to save the split files.
        train_ratio (float): Ratio of training data out of the entire dataset. Defaults to 0.7.
        validation_ratio (float): Ratio of validation data out of the entire dataset. Defaults to 0.2.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            A tuple of (train_df, validation_df, test_df).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_save_path = os.path.join(save_dir, "train.csv")
    val_save_path = os.path.join(save_dir, "validation.csv")
    test_save_path = os.path.join(save_dir, "test.csv")

    # Set random seed
    random.seed(seed)

    # Get unique protein names
    unique_proteins = list(set(df["Protein names"].to_list()))

    # Shuffle the list of unique proteins
    random.shuffle(unique_proteins)

    # Calculate split indices for train+val vs test, and for train vs val
    train_val_cutoff = int(len(unique_proteins) * (train_ratio + validation_ratio))
    train_cutoff = int(len(unique_proteins) * train_ratio)

    train_val_proteins = unique_proteins[:train_val_cutoff]  # train + validation
    test_proteins = unique_proteins[train_val_cutoff:]  # test

    train_proteins = train_val_proteins[:train_cutoff]  # training
    validation_proteins = train_val_proteins[train_cutoff:]  # validation

    # Filter out rows that have invalid or missing embeddings
    df = df[df.apply(_has_valid_embedding, axis=1)].reset_index(drop=True)

    # Create train/validation/test DataFrames
    train_df = df[df["Protein names"].isin(train_proteins)].reset_index(drop=True)
    validation_df = df[df["Protein names"].isin(validation_proteins)].reset_index(
        drop=True
    )
    test_df = df[df["Protein names"].isin(test_proteins)].reset_index(drop=True)

    print(f"Number of training proteins: {len(train_proteins)}")
    print(f"Number of validation proteins: {len(validation_proteins)}")
    print(f"Number of test proteins: {len(test_proteins)}")

    print(f"Number of rows in training data: {len(train_df)}")
    print(f"Number of rows in validation data: {len(validation_df)}")
    print(f"Number of rows in test data: {len(test_df)}")

    train_df.to_csv(train_save_path, index=False)
    validation_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    return train_df, validation_df, test_df


def _has_valid_embedding(row: pd.Series) -> bool:
    """
    Checks if the embedding file exists and the embedding vector is valid (non-empty).

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        bool: True if the embedding is valid, False otherwise.
    """
    peptido_embedding_path = row["Peptido_Embedding_Path"]
    # print(type(peptido_embedding_path))
    if not os.path.exists(peptido_embedding_path):
        return False
    try:
        data = np.load(peptido_embedding_path)
        peptido_embedding = data["peptido_embedding"]
        return len(peptido_embedding) > 0
    except Exception as e:
        print(
            f"Error: An exception occurred while loading {peptido_embedding_path} - {e}"
        )
        return False


def sampling_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs sampling per cluster to match the size of the smallest cluster.
    The sampling is done at the peptide level: once a peptide is chosen,
    all of its time points are included.

    Args:
        df (pd.DataFrame): A DataFrame containing cluster information.

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    # Count unique peptides per cluster
    cluster_peptide_counts = df.groupby("Cluster")["Cleaned_Peptidoform"].nunique()
    min_peptide_count = cluster_peptide_counts.min()

    sampled_dfs = []

    for cluster in cluster_peptide_counts.index:
        cluster_df = df[df["Cluster"] == cluster]
        cluster_count = len(df["Cluster"].unique())
        print(f"Number of unique clusters in the entire DataFrame: {cluster_count}")

        unique_peptides = cluster_df["Cleaned_Peptidoform"].unique()

        # Randomly sample the peptides
        sampled_peptides = np.random.choice(
            unique_peptides, size=min_peptide_count, replace=False
        )

        # Retrieve all rows related to the sampled peptides
        sampled_cluster_df = cluster_df[
            cluster_df["Cleaned_Peptidoform"].isin(sampled_peptides)
        ]
        sampled_dfs.append(sampled_cluster_df)

    # Merge all sampled subsets
    sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
    print(f"Length of the DataFrame after sampling: {len(sampled_df)}")
    return sampled_df


def enhanced_k_fold_split_dataset(
    df: pd.DataFrame,
    save_dir: str,
    n_splits: int = 10,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    K-Fold split by protein, using 7:2:1 (train:val:test) ratio.
    Saves split datasets and stats (protein/peptide counts per split).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(df)
    # input()
    df = df[df.apply(_has_valid_embedding, axis=1)].reset_index(drop=True)
    unique_proteins = df["Protein names"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    summary = []

    for fold_num, (train_val_index, test_index) in enumerate(kf.split(unique_proteins)):
        test_proteins = unique_proteins[test_index]
        train_val_proteins = unique_proteins[train_val_index]

        np.random.seed(random_state + fold_num)
        train_val_proteins_shuffled = np.random.permutation(train_val_proteins)

        train_cutoff = int(len(train_val_proteins) * train_ratio)
        validation_cutoff = int(
            len(train_val_proteins) * (train_ratio + validation_ratio)
        )

        train_proteins = train_val_proteins_shuffled[:train_cutoff]
        validation_proteins = train_val_proteins_shuffled[
            train_cutoff:validation_cutoff
        ]

        train_df = df[df["Protein names"].isin(train_proteins)].reset_index(drop=True)
        validation_df = df[df["Protein names"].isin(validation_proteins)].reset_index(
            drop=True
        )
        test_df = df[df["Protein names"].isin(test_proteins)].reset_index(drop=True)

        save_dir_fold = os.path.join(save_dir, f"fold_{fold_num}")

        if not os.path.exists(save_dir_fold):
            os.makedirs(save_dir_fold)

        train_path = os.path.join(save_dir_fold, "train_fold.csv")
        val_path = os.path.join(save_dir_fold, "validation_fold.csv")
        test_path = os.path.join(save_dir_fold, "test_fold.csv")
        stats_path = os.path.join(save_dir_fold, "stats_fold.csv")

        train_df.to_csv(train_path, index=False)
        validation_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        stats = {
            "Fold": fold_num,
            "Train Proteins": len(train_proteins),
            "Validation Proteins": len(validation_proteins),
            "Test Proteins": len(test_proteins),
            "Train Peptides": train_df["Cleaned_Peptidoform"].nunique(),
            "Validation Peptides": validation_df["Cleaned_Peptidoform"].nunique(),
            "Test Peptides": test_df["Cleaned_Peptidoform"].nunique(),
            "Train Rows": len(train_df),
            "Validation Rows": len(validation_df),
            "Test Rows": len(test_df),
        }
        summary.append(stats)

        pd.DataFrame([stats]).to_csv(stats_path, index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(save_dir, "kfold_summary.csv"), index=False)
    return summary_df


def reduce_cluster_size_by_peptide(
    df: pd.DataFrame,
    clusters_to_reduce=None,
    fraction: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly downsamples specific cluster(s) at the peptide level according to the specified fraction.
    For example, fraction=0.5 will remove half of the unique peptides (and all rows associated
    with those peptides) in the targeted cluster(s). Non-targeted clusters remain unchanged.

    If multiple clusters are specified (e.g., [1, 3]), each is processed in sequence.

    Args:
        df (pd.DataFrame): DataFrame with cluster information.
        clusters_to_reduce (int or list[int], optional): The cluster ID(s) to be reduced.
            If None, defaults to [3].
        fraction (float, optional): The fraction of unique peptides to keep in the specified clusters.
            Defaults to 0.5.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        pd.DataFrame: A DataFrame where specified clusters have been reduced to the fraction indicated.
    """
    # If a single int is provided, wrap it in a list
    if isinstance(clusters_to_reduce, int):
        clusters_to_reduce = [clusters_to_reduce]
    elif clusters_to_reduce is None:
        # Default: cluster 3 only
        clusters_to_reduce = [3]

    # For each target cluster, sample peptides and remove the remaining ones
    for cluster_id in clusters_to_reduce:
        # Separate target cluster vs others
        target_df = df[df["Cluster"] == cluster_id]
        other_df = df[df["Cluster"] != cluster_id]

        unique_peptides = target_df["Cleaned_Peptidoform"].unique()
        np.random.seed(random_state)
        sample_size = int(len(unique_peptides) * fraction)
        sampled_peptides = np.random.choice(
            unique_peptides, size=sample_size, replace=False
        )

        # Keep rows only for the sampled peptides
        target_sampled_df = target_df[
            target_df["Cleaned_Peptidoform"].isin(sampled_peptides)
        ]

        # Re-combine with other clusters
        df = pd.concat([other_df, target_sampled_df], axis=0).reset_index(drop=True)

        print(
            f"\n[Cluster {cluster_id}]: Before sampling => rows: {len(target_df)}, unique peptides: {len(unique_peptides)}"
        )
        print(
            f"[Cluster {cluster_id}]: After sampling  => rows: {len(target_sampled_df)}, unique peptides: {len(sampled_peptides)}"
        )
        print(f"[Cluster {cluster_id}]: Updated DataFrame size => rows: {len(df)}")

    return df


def create_K_dataset():
    return


if __name__ == "__main__":
    preprocess_main()
