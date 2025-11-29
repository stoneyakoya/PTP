import json
import os
import random
import re
import time

import esm
import esm.pretrained
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

"""
Preprocessing Steps
  PrepareTarget -> PrepareInput

PrepareTarget
- summarize_peptide:
  This step processes the raw dataset by:
    1. Cleaning peptide strings by removing parenthetical modifications.
    2. Aggregating replicate data for each peptide and computing the mean across time points.
    3. Calculating turnover rates for each time point based on label incorporation and label loss values.
    4. Producing a cleaned, NaN-free DataFrame summarizing peptide-level data.

- data_normalize:
  Normalizes the processed peptide data to ensure consistent value scaling:
    - MinMax scaling: Scales values across time points to the [0, 1] range using MinMaxScaler.
    - Clipping (clip_arch): Directly clips each time point's values to the range [0, 1].
  This step standardizes the data for downstream analysis or model training.

PrepareInput
- get_protein_sequence:
  Retrieves protein sequences from UniProt for each peptide in the raw dataset. The function:
    1. Extracts the first UniProt ID from each peptide.
    2. Fetches corresponding JSON data from the UniProt database.
    3. Extracts and saves the protein sequence locally for further processing.

- generate_protein_embedding:
  Utilizes the ESM2 model to generate embedding representations for the retrieved protein sequences. 
  These embeddings provide a vectorized representation capturing structural and functional characteristics of the proteins.

- extract_peptide_embedding:
  Extracts peptide-level embeddings from the larger protein embedding by identifying the corresponding region 
  of the peptide sequence within the protein.

- create_lag_feature:
  Constructs lag features for turnover data (label loss, label incorporation, and related metrics). 
  Lag features enable temporal dependencies to be modeled effectively, improving the prediction of turnover dynamics over time.
"""


class PrepareTarget:
    def __init__(self, data_path: str):
        """
        Initialize the PrepareTargets class.

        Args:
            data_path (str): Path to the CSV file containing TMT data.
        """
        self.data_path = data_path

    def summarize_peptide(self) -> pd.DataFrame:
        """
        Aggregates data on a per-peptide basis.
        1. Reads the data from the CSV file.
        2. Cleans up peptide strings by removing any parenthetical text (e.g., modifications).
        3. Converts the data into a pivot table.
        4. Calculates the mean across replicates for each time point.
        5. Computes turnover rate for each time point.

        Returns:
            pd.DataFrame: A DataFrame with turnover rate and related columns for each peptide.
        """
        # Read the CSV data; header=1 means the second row is treated as the header
        df = pd.read_csv(self.data_path, header=1)

        # if remove_PTM is True, remove peptides with modifications
        df = df[~df["Peptidoform"].str.contains(r"\(.*\)", na=False)]

        # Clean the peptide string by removing parenthetical text
        df["Cleaned_Peptidoform"] = df["Peptidoform"].apply(
            lambda x: re.sub(r"\(.*?\)", "", x)
        )

        # Create a pivot table, averaging across replicates
        df_pivot = df.pivot_table(
            index=[
                "Cleaned_Peptidoform",
                "Peptidoform",
                "Replicate",
                "UniProt IDs",
                "Protein names",
            ],
            columns="TurnoverType",
            values=[
                "corrRatio_0h",
                "corrRatio_1h",
                "corrRatio_3h",
                "corrRatio_6h",
                "corrRatio_10h",
                "corrRatio_16h",
                "corrRatio_24h",
                "corrRatio_34h",
                "corrRatio_48h",
                "corrRatio_infin.h",
                "K",
            ],
            aggfunc="mean",
        )

        # Define time points
        time_points = [
            "0h",
            "1h",
            "3h",
            "6h",
            "10h",
            "16h",
            "24h",
            "34h",
            "48h",
            "infin.h",
        ]

        # Create a DataFrame to store the mean replicate data
        mean_replicate_df = pd.DataFrame(index=df_pivot.index)

        for time_point in time_points:
            label_loss_key = ("corrRatio_" + time_point, "Label loss")
            label_incorp_key = ("corrRatio_" + time_point, "Label incorporation")

            mean_replicate_df[f"Label loss_{time_point}"] = df_pivot[label_loss_key]
            mean_replicate_df[f"Label incorporation_{time_point}"] = df_pivot[
                label_incorp_key
            ]

        # Include K (turnover rate)
        mean_replicate_df["loss_K"] = df_pivot[("K", "Label loss")]
        mean_replicate_df["incorporation_K"] = df_pivot[("K", "Label incorporation")]

        # Group by peptide/protein info, take the mean across replicates, then reset index
        mean_replicate_df = (
            mean_replicate_df.groupby(
                [
                    "Cleaned_Peptidoform",
                    "UniProt IDs",
                    "Protein names",
                ]
            )
            .mean()
            .reset_index()
        )

        # Remove rows that contain NaN values (if any replicate is missing data)
        cleaned_mean_replicate_df = mean_replicate_df.dropna(how="any").reset_index(
            drop=True
        )

        # Calculate TurnoverRate for each time point by dividing label incorporation
        # by the sum of label loss and label incorporation
        for time_point in time_points:
            label_loss_col = f"Label loss_{time_point}"
            label_incorp_col = f"Label incorporation_{time_point}"
            turnover_rate_col = f"TurnoverRate_{time_point}"

            cleaned_mean_replicate_df[turnover_rate_col] = cleaned_mean_replicate_df[
                label_incorp_col
            ] / (
                cleaned_mean_replicate_df[label_loss_col]
                + cleaned_mean_replicate_df[label_incorp_col]
            )

        return cleaned_mean_replicate_df

    def data_normalize(
        self, df: pd.DataFrame, normalize_method="clip_arch"
    ) -> pd.DataFrame:
        """
        Normalizes the data.
        There are two methods:
          1. MinMax: Scales data between 0 and 1 using MinMaxScaler.
          2. clip_arch: Clips each time point's values within [0, 1] directly.

        Args:
            df (pd.DataFrame): The input DataFrame to normalize.
            normalize_method (str): The normalization method to use. Defaults to "clip_arch".

        Returns:
            pd.DataFrame: A DataFrame with normalized values.
        """
        # Exclude 0h and infin.h from certain normalizations
        time_points = ["1h", "3h", "6h", "10h", "16h", "24h", "34h", "48h"]
        metrics = ["Label loss", "Label incorporation", "TurnoverRate"]

        df = df.copy()  # 安全のためコピー
        if normalize_method == "MinMax":
            scaler = MinMaxScaler()
            for metric in metrics:
                for tp in time_points:
                    col = f"{metric}_{tp}"
                    if col in df.columns:
                        df[[col]] = scaler.fit_transform(df[[col]])

        elif normalize_method == "clip_arch":
            for metric in metrics:
                for tp in time_points:
                    col = f"{metric}_{tp}"
                    if col in df.columns:
                        df[col] = df[col].clip(lower=0, upper=1)
        return df

    def timeseries_info(
        self,
        df: pd.DataFrame,
        metric: str,
        normalize_method="clip_arch",
        save_dir: str = "data/data_analyze/boxplot",
    ):
        """
        指定された metric（例: TurnoverRate, Label loss, Label incorporation）に対して
        各タイムポイントの箱ひげ図を描画する

        Parameters
        ----------
        df : pd.DataFrame
            summarize_peptide の出力
        metric : str
            可視化するメトリクス（TurnoverRate, Label loss, Label incorporation）
        """
        os.makedirs(save_dir, exist_ok=True)
        
        time_points = [
            "0h",
            "1h",
            "3h",
            "6h",
            "10h",
            "16h",
            "24h",
            "34h",
            "48h",
            "infin.h",
        ]

        # long-form に変換
        data = []
        for time in time_points:
            col = f"{metric}_{time}"
            if col not in df.columns:
                continue
            for value in df[col]:
                data.append({"Time": time, "Value": value})

        plot_df = pd.DataFrame(data)

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

        # 可視化
        fig, ax = plt.subplots(figsize=(14, 7))  # 横幅を小さく
        
        # グラデーションカラーでboxplotを作成
        n_times = len(plot_df['Time'].unique())
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_times))
        
        box = sns.boxplot(
            data=plot_df, 
            x="Time", 
            y="Value", 
            palette=colors,
            linewidth=1.5,
            ax=ax,
            showfliers=False  # 外れ値（黒いポツポツ）を非表示
        )
        
        ax.set_title(f"{metric} over Time", fontsize=24, fontweight='bold', pad=15)
        ax.set_ylabel(metric, fontsize=22, fontweight='bold')
        ax.set_xlabel("Time point", fontsize=22, fontweight='bold')
        ax.tick_params(axis='x', labelsize=18, rotation=0)  # X軸目盛り大きく
        ax.tick_params(axis='y', labelsize=18)  # Y軸目盛り大きく
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.02, 1.02)  # 下限-0.2、上限1.2
        
        plt.tight_layout()
        
        # PNG と PDF で保存
        save_name_png = f"{normalize_method}_{metric}_boxplot_for_paper.png"
        save_name_pdf = f"{normalize_method}_{metric}_boxplot_for_paper.pdf"
        save_path_png = os.path.join(save_dir, save_name_png)
        save_path_pdf = os.path.join(save_dir, save_name_pdf)
        plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=600, bbox_inches='tight', format='pdf')
        print(f"Saved: {save_path_png}")
        print(f"Saved: {save_path_pdf}")
        plt.show()

    def run(self, debug=False) -> pd.DataFrame:
        """
        Orchestrates the peptide summarization and normalization steps.

        Args:
            debug (bool): If True, prints debug information. Defaults to False.

        Returns:
            pd.DataFrame: The final processed DataFrame containing turnover data.
        """
        peptide_df = self.summarize_peptide()
        normalized_df = self.data_normalize(peptide_df)
        # Visualize the time series data
        normalize_method = "clip_arch"
        
        # Individual plots
        self.timeseries_info(normalized_df, "Label loss", normalize_method)
        self.timeseries_info(normalized_df, "Label incorporation", normalize_method)
        self.timeseries_info(normalized_df, "TurnoverRate", normalize_method)
        
        # Combined plot (3 in 1)
        self.timeseries_info_combined(normalized_df, normalize_method)
        
        if debug:
            print(normalized_df)
            print(normalized_df.columns)
        return normalized_df
    
    def timeseries_info_combined(
        self,
        df: pd.DataFrame,
        normalize_method="clip_arch",
        save_dir: str = "data/data_analyze/boxplot",
    ):
        """
        3つのboxplot（Label loss, Label incorporation, Turnover Rate）を
        1つの画像にまとめて表示
        """
        os.makedirs(save_dir, exist_ok=True)
        
        time_points = [
            "0h", "1h", "3h", "6h", "10h", "16h", "24h", "34h", "48h", "infin.h"
        ]
        
        metrics = ["Label loss", "Label incorporation", "TurnoverRate"]
        
        # フォント設定 - Arial指定
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        
        # 3行1列のサブプロット（横幅を狭く）
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # long-form に変換
            data = []
            for time in time_points:
                col = f"{metric}_{time}"
                if col not in df.columns:
                    continue
                for value in df[col]:
                    data.append({"Time": time, "Value": value})
            
            plot_df = pd.DataFrame(data)
            
            # グラデーションカラーでboxplotを作成
            n_times = len(plot_df['Time'].unique())
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_times))
            
            box = sns.boxplot(
                data=plot_df,
                x="Time",
                y="Value",
                palette=colors,
                linewidth=1.5,
                ax=ax,
                showfliers=False
            )
            
            ax.set_title(f"{metric} over Time", fontsize=22, fontweight='bold', pad=15)
            ax.set_ylabel(metric, fontsize=20, fontweight='bold')
            ax.set_xlabel("Time point", fontsize=20, fontweight='bold')
            ax.tick_params(axis='x', labelsize=16, rotation=0)
            ax.tick_params(axis='y', labelsize=16)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylim(-0.02, 1.02)
        
        plt.tight_layout()
        
        # PNG と PDF で保存
        save_name_png = f"{normalize_method}_all_metrics_boxplot_for_paper.png"
        save_name_pdf = f"{normalize_method}_all_metrics_boxplot_for_paper.pdf"
        save_path_png = os.path.join(save_dir, save_name_png)
        save_path_pdf = os.path.join(save_dir, save_name_pdf)
        plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=600, bbox_inches='tight', format='pdf')
        print(f"Saved combined plot: {save_path_png}")
        print(f"Saved combined plot: {save_path_pdf}")
        plt.close()


class PrepareInput:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the PrepareInput class.

        Args:
            df (pd.DataFrame): A DataFrame containing turnover data (e.g., from PrepareTargets).
        """
        self.df = df

    def get_protein_sequence(
        self,
        df_save_path: str,
        uniprot_json_dir: str = "data/uniprot",
    ) -> pd.DataFrame:
        """
        Adds protein sequences to the DataFrame by either extracting them from pre-downloaded JSON
        files or by fetching them from the UniProt API if not available locally.

        Args:
            df_save_path (str): Path to save the updated DataFrame.
            uniprot_json_dir (str): Directory where UniProt JSON files are stored or will be stored.

        Returns:
            pd.DataFrame: A DataFrame that includes the 'Protein Sequence' column.
        """
        print("Adding protein sequences...")
        protein_sequences = []
        unidentified_uniprots = []

        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            uni_id = row["UniProt IDs"].split(";")[0]
            uniprot_json_path = os.path.join(uniprot_json_dir, f"{uni_id}.json")

            # 1. Attempt to extract protein sequence from a local JSON file
            protein_sequence = self._extract_protein_sequence(uni_id, uniprot_json_path)

            if not protein_sequence:
                # If the local file or valid sequence is not found, fetch from UniProt
                print("Fetching from UniProt...")
                protein_sequence = self._get_protein_sequence_from_uniprot(uni_id)

            if not protein_sequence:
                unidentified_uniprots.append(uni_id)

            protein_sequences.append(protein_sequence)

        # Add the protein sequence column to the DataFrame
        new_df = self.df.copy()
        new_df["Protein Sequence"] = protein_sequences

        # Remove rows that do not have valid protein sequences
        cleaned_df = new_df[
            new_df["Protein Sequence"].notna() & (new_df["Protein Sequence"] != "")
        ]
        cleaned_df = cleaned_df.dropna(how="any")
        cleaned_df.to_csv(df_save_path, index=False)

        return cleaned_df

    def _get_protein_sequence_from_uniprot(
        self, uniprot_id: str, save_dir: str = "data/uniprot"
    ) -> str:
        """
        Fetches the protein sequence from UniProt using their public API.

        Args:
            uniprot_id (str): The UniProt ID.
            save_dir (str): Directory to save the downloaded JSON file.

        Returns:
            str: The protein sequence, or an empty string if not found.
        """
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.json"
        # Random wait time (0.5-2.0 seconds) before the request to avoid overloading the server
        wait_time = random.uniform(0.5, 2.0)
        time.sleep(wait_time)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            # Check the JSON structure for 'sequence'
            if "sequence" in data and "sequence" in data["sequence"]:
                print(f"✅ Retrieved from UniProt API: {uniprot_id}")
                with open(os.path.join(save_dir, f"{uniprot_id}.json"), "w") as f:
                    json.dump(data, f)
                return data["sequence"]["sequence"]
            elif "sequence" in data and "value" in data["sequence"]:
                # Some JSON files might store the sequence under "value"
                print(f"✅ Retrieved from UniProt API: {uniprot_id}")
                with open(os.path.join(save_dir, f"{uniprot_id}.json"), "w") as f:
                    json.dump(data, f)
                return data["sequence"]["value"]
            else:
                print("Unexpected JSON format. Please check the content:")
                print(data)
                return ""
        else:
            print(f"{response}")
            print(f"❌ Could not retrieve data from UniProt for: {uniprot_id}")
            return ""

    def _extract_protein_sequence(self, uniprot_id: str, uniprot_json_path: str) -> str:
        """
        Extracts the protein sequence from a local UniProt JSON file.

        Args:
            uniprot_id (str): UniProt ID.
            uniprot_json_path (str): Path to the JSON file.

        Returns:
            str: Protein sequence or an empty string if not found.
        """
        first_uniprot_id = uniprot_id.split(";")[0]
        fallback_path = os.path.join(
            os.path.dirname(uniprot_json_path), f"{first_uniprot_id}.json"
        )

        # Check if JSON file exists
        if os.path.exists(uniprot_json_path):
            with open(uniprot_json_path, "r") as f:
                uniprot_json = json.load(f)
        elif os.path.exists(fallback_path):
            with open(fallback_path, "r") as f:
                uniprot_json = json.load(f)
        else:
            print(f"JSON file not found: {uniprot_json_path}")
            return ""

        # Extract the protein sequence from the JSON structure
        if "sequence" in uniprot_json and "sequence" in uniprot_json["sequence"]:
            return uniprot_json["sequence"]["sequence"]
        elif "sequence" in uniprot_json and "value" in uniprot_json["sequence"]:
            return uniprot_json["sequence"]["value"]
        else:
            print(f"No sequence information found for: {uniprot_id}")
            return ""

    def generate_protein_embedding(
        self,
        df: pd.DataFrame,
        df_save_path: str,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda",
        emb_save_dir: str = "data/protein_embeddings",
    ) -> pd.DataFrame:
        """
        Generates and saves embedding vectors for each protein sequence using ESM2.

        Args:
            df (pd.DataFrame): DataFrame containing 'Protein Sequence'.
            df_save_path (str): CSV file path to save the updated DataFrame.
            model_name (str): The ESM2 model name to use. Defaults to "esm2_t33_650M_UR50D".
            device (str): Device to run inference on ("cuda" or "cpu"). Defaults to "cuda".
            emb_save_dir (str): Directory to save protein embedding files.

        Returns:
            pd.DataFrame: The input DataFrame is returned unchanged, but the embeddings are
                          saved separately as .npz files.
        """
        print("Generating protein embeddings...")

        # Load the ESM2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(device)
        model.eval()

        # Create the embedding directory if it doesn't exist
        if not os.path.exists(emb_save_dir):
            os.makedirs(emb_save_dir)

        # Prepare batch converter for ESM
        batch_converter = alphabet.get_batch_converter()

        success_index = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            sequence = row["Protein Sequence"]
            protein_name = row["Protein names"]
            uniport_id = row["UniProt IDs"].split(";")[0]

            emb_save_path = os.path.join(emb_save_dir, f"{uniport_id}.npz")

            # If an embedding already exists for this UniProt ID, skip
            if os.path.exists(emb_save_path):
                continue

            try:
                # Convert the sequence into a batch format
                data = [(protein_name, sequence)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                # Compute the embeddings
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])
                    token_representations = results["representations"][33].squeeze(0)

                # Save the residue-level embeddings (including [CLS], [EOS] tokens)
                residue_embeddings = token_representations.cpu().numpy()
                np.savez_compressed(
                    emb_save_path, residue_embeddings=residue_embeddings
                )
                success_index.append(i)

            except Exception as e:
                print(f"Failed to process {protein_name} due to {e}")
                continue

        df.to_csv(df_save_path, index=False)
        print(
            f"Protein embedding info saved, and DataFrame with updated paths saved to {df_save_path}."
        )

        return df

    def extract_peptide_embedding(
        self,
        df: pd.DataFrame,
        df_save_path: str,
        protein_embeddings_dir: str = "data/protein_embeddings/",
        peptido_embeddings_dir: str = "data/peptide_embeddings/",
    ) -> pd.DataFrame:
        """
        Extracts peptide-level embeddings from the corresponding protein embeddings.
        Note that protein_embeddings usually have length = protein_seq_length + 2.
        The CLS token embedding (index 0) is prepended to each peptide embedding.

        Args:
            df (pd.DataFrame): DataFrame containing peptide and protein information.
            df_save_path (str): CSV file path to save the updated DataFrame.
            protein_embeddings_dir (str): Directory where protein .npz embeddings are stored.
            peptido_embeddings_dir (str): Directory to save peptide-level .npz embeddings.

        Returns:
            pd.DataFrame: Updated DataFrame with a new column 'Peptido_Embedding_Path'.
        """
        print("Starting peptide embedding extraction...")

        peptido_embedding_paths = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            peptidoform = row["Cleaned_Peptidoform"]
            protein_name = row["Protein names"]
            first_uniport_id = row["UniProt IDs"].split(";")[0]
            protein_sequence = row["Protein Sequence"]

            # Create the peptide embeddings directory if it doesn't exist
            if not os.path.exists(peptido_embeddings_dir):
                os.makedirs(peptido_embeddings_dir)

            # Check if the peptide embedding already exists
            existing_peptido_path = os.path.join(
                peptido_embeddings_dir, f"{peptidoform}_{first_uniport_id}.npz"
            )
            if os.path.exists(existing_peptido_path):
                peptido_embedding_paths.append(existing_peptido_path)
                continue

            # Check if the protein embedding file exists
            embedding_file = os.path.join(
                protein_embeddings_dir, f"{first_uniport_id}.npz"
            )
            if not os.path.exists(embedding_file):
                print(
                    f"Embedding file not found for {protein_name} ({first_uniport_id}). Skipping."
                )
                peptido_embedding_paths.append(np.nan)
                continue

            # Load the protein embedding
            try:
                data = np.load(embedding_file)
                residue_embeddings = data["residue_embeddings"]
            except Exception as e:
                print(f"Failed to load embeddings for {protein_name}: {e}")
                peptido_embedding_paths.append(np.nan)
                continue

            # Verify the length: protein_sequence length + 2 should match residue_embeddings length
            if len(protein_sequence) + 2 != len(residue_embeddings):
                print(protein_sequence)
                print(residue_embeddings[0])
                print("Length mismatch.")
                # We won't necessarily skip here; it's just a warning

            # Find the position of the peptide within the protein sequence
            peptido_start_index = protein_sequence.find(peptidoform)
            if peptido_start_index == -1:
                print(
                    f"Peptidoform {peptidoform} not found in {protein_name}. Skipping."
                )
                peptido_embedding_paths.append(np.nan)
                continue

            peptido_end_index = peptido_start_index + len(peptidoform)

            # Extract the CLS token embedding (index 0)
            cls_token_embedding = residue_embeddings[0, :]

            # Extract the peptide embedding from residue_embeddings, then prepend CLS
            peptido_embedding = residue_embeddings[
                peptido_start_index + 1 : peptido_end_index + 1, :
            ]
            peptido_embedding_with_cls = np.vstack(
                [cls_token_embedding, peptido_embedding]
            )

            # Save the peptide embedding
            peptido_embedding_file = f"{peptidoform}_{first_uniport_id}.npz"
            peptido_embedding_path = os.path.join(
                peptido_embeddings_dir, peptido_embedding_file
            )
            np.savez_compressed(
                peptido_embedding_path, peptido_embedding=peptido_embedding_with_cls
            )

            peptido_embedding_paths.append(peptido_embedding_path)

        # Add a new column to the DataFrame with the path to the peptide embeddings
        df["Peptido_Embedding_Path"] = peptido_embedding_paths

        # Check how many are NaN or empty
        nan_count = df["Peptido_Embedding_Path"].isna().sum()
        empty_count = sum(
            df["Peptido_Embedding_Path"].apply(
                lambda x: x is None or (isinstance(x, np.ndarray) and len(x) == 0)
            )
        )
        print(f"Total entries: {len(df)}")
        print(f"NaN entries in Peptido_Embedding_Path: {nan_count}")
        print(f"Empty embeddings: {empty_count}")

        # Remove rows with NaN or empty embeddings
        df_cleaned = df.dropna(subset=["Peptido_Embedding_Path"])
        df_cleaned = df_cleaned[
            df_cleaned["Peptido_Embedding_Path"].apply(
                lambda x: not (x is None or (isinstance(x, np.ndarray) and len(x) == 0))
            )
        ]

        df_cleaned.to_csv(df_save_path, index=False)
        print(f"Cleaned DataFrame with peptide embeddings saved to: {df_save_path}")

        return df

    def create_lag_feature(self, df: pd.DataFrame, df_save_path: str) -> pd.DataFrame:
        """
        Creates lag features for label loss/incorporation and TurnoverRate.

        Args:
            df (pd.DataFrame): The DataFrame containing time-series data of label loss,
                               label incorporation, and turnover rate.
            df_save_path (str): Path to save the transformed DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame containing lag features and target columns.
        """
        print("Creating lag features...")

        # Define the total number of target time points per peptide
        timesteps = 8

        transformed_data = []

        for _, row in df.iterrows():
            # Metadata
            peptidoform = row["Cleaned_Peptidoform"]
            uniprot_ids = row["UniProt IDs"]
            protein_names = row["Protein names"]
            protein_sequence = row["Protein Sequence"]
            peptide_embedding_path = row["Peptido_Embedding_Path"]

            # Define time-series columns
            turnover_times = [
                "TurnoverRate_1h",
                "TurnoverRate_3h",
                "TurnoverRate_6h",
                "TurnoverRate_10h",
                "TurnoverRate_16h",
                "TurnoverRate_24h",
                "TurnoverRate_34h",
                "TurnoverRate_48h",
            ]
            loss_times = [
                "Label loss_1h",
                "Label loss_3h",
                "Label loss_6h",
                "Label loss_10h",
                "Label loss_16h",
                "Label loss_24h",
                "Label loss_34h",
                "Label loss_48h",
            ]
            incorporation_times = [
                "Label incorporation_1h",
                "Label incorporation_3h",
                "Label incorporation_6h",
                "Label incorporation_10h",
                "Label incorporation_16h",
                "Label incorporation_24h",
                "Label incorporation_34h",
                "Label incorporation_48h",
            ]

            turnover_series = row[turnover_times].values
            loss_series = row[loss_times].values
            incorporation_series = row[incorporation_times].values

            # Extract just the hour labels for reference
            timestep_list = [s.split("_")[1][:-1] for s in turnover_times]

            # For each time point, create a row in the transformed dataset
            for t in range(1, timesteps + 1):
                timestep = timestep_list[t - 1]

                # Target values at time t
                target_turnover = turnover_series[t - 1]
                target_loss = loss_series[t - 1]
                target_incorporation = incorporation_series[t - 1]

                # Lag features (reverse-ordered slices)
                lags_turnover = turnover_series[: t - 1][::-1]
                lags_loss = loss_series[: t - 1][::-1]
                lags_incorporation = incorporation_series[: t - 1][::-1]

                # Pad lag features with zeros (or ones in the case of loss) if needed
                lags_turnover_padded = np.pad(
                    lags_turnover,
                    (0, timesteps - 1 - len(lags_turnover)),
                    mode="constant",
                    constant_values=0,
                )
                lags_loss_padded = np.pad(
                    lags_loss,
                    (0, timesteps - 1 - len(lags_loss)),
                    mode="constant",
                    constant_values=1,
                )
                lags_incorporation_padded = np.pad(
                    lags_incorporation,
                    (0, timesteps - 1 - len(lags_incorporation)),
                    mode="constant",
                    constant_values=0,
                )

                # Collect all features into one row
                transformed_data.append(
                    [
                        peptidoform,
                        timestep,
                        uniprot_ids,
                        protein_names,
                        protein_sequence,
                        peptide_embedding_path,
                    ]
                    + lags_turnover_padded.tolist()
                    + lags_loss_padded.tolist()
                    + lags_incorporation_padded.tolist()
                    + [target_turnover, target_loss, target_incorporation]
                    + [row["loss_K"], row["incorporation_K"]]
                )

        # Create the output DataFrame
        metadata_columns = [
            "Cleaned_Peptidoform",
            "timestep",
            "UniProt IDs",
            "Protein names",
            "Protein Sequence",
            "Peptido_Embedding_Path",
        ]
        lag_turnover_columns = [f"TurnoverRate_t-{i + 1}" for i in range(timesteps - 1)]
        lag_loss_columns = [f"Label loss_t-{i + 1}" for i in range(timesteps - 1)]
        lag_incorp_columns = [
            f"Label incorporation_t-{i + 1}" for i in range(timesteps - 1)
        ]

        columns = (
            metadata_columns
            + lag_turnover_columns
            + lag_loss_columns
            + lag_incorp_columns
            + ["Target_TurnoverRate", "Target_LabelLoss", "Target_LabelIncorporation"]
            + ["loss_K", "incorporation_K"]
        )
        transformed_df = pd.DataFrame(transformed_data, columns=columns)
        transformed_df.to_csv(df_save_path, index=False)

        return transformed_df

    def run(self, debug: bool = False) -> pd.DataFrame:
        """
        Orchestrates the data preparation steps, including:
          1) Retrieving protein sequences,
          2) Generating protein embeddings,
          3) Extracting peptide embeddings,
          4) Creating lag features.

        Args:
            debug (bool): Whether to print debug information. Defaults to False.

        Returns:
            pd.DataFrame: The final DataFrame with lag features.
        """
        # Step 1: Get protein sequences
        df_save_path = "data/tmp/df_with_ProteinSeq.csv"
        if not os.path.exists(df_save_path):
            df_with_ProteinSeq = self.get_protein_sequence(df_save_path)
        else:
            df_with_ProteinSeq = pd.read_csv(df_save_path)

        # Step 2: Generate protein embeddings
        df_save_path = "data/tmp/df_with_ProteinSeq_ProteinEmbedPath.csv"
        if not os.path.exists(df_save_path):
            df_with_ProteinSeq_ProteinEmbedPath = self.generate_protein_embedding(
                df_with_ProteinSeq, df_save_path
            )
        else:
            df_with_ProteinSeq_ProteinEmbedPath = pd.read_csv(df_save_path)

        # Step 3: Extract peptide embeddings
        df_save_path = (
            "data/tmp/df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath.csv"
        )
        if not os.path.exists(df_save_path):
            df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath = (
                self.extract_peptide_embedding(
                    df_with_ProteinSeq_ProteinEmbedPath, df_save_path
                )
            )
        else:
            df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath = pd.read_csv(
                df_save_path
            )

        if debug:
            print("\n==== df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath ====\n")
            print(df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath)
            print(df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath.columns)

        # Step 4: Create lag features
        df_save_path = (
            "data/tmp/df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag.csv"
        )
        if not os.path.exists(df_save_path):
            df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag = (
                self.create_lag_feature(
                    df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath, df_save_path
                )
            )
        else:
            df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag = pd.read_csv(
                df_save_path
            )

        if debug:
            print(
                "\n==== df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag ====\n"
            )
            print(df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag)
            print(df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag.columns)

        return df_with_ProteinSeq_ProteinEmbedPath_PeptideEmbedPath_Lag


if __name__ == "__main__":
    # Example usage
    raw_data_path = "data/raw/pSILAC_TMT.csv"
    target_df = PrepareTarget(raw_data_path).run(debug=True)
    input_target_df = PrepareInput(target_df).run(debug=True)
