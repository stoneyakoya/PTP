import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Datasets_TimeSeq(Dataset):
    """
    TimeSeq用のデータセット:
      - DataFrame には「1行=1タイムポイント」が格納されているが、
      - 同じペプチドIDの行を 8ステップぶんまとめて (8, input_dim) を返す。

    return:
      - "Cleaned_Peptidoform"
      - "timestep"
      - "Peptido_Embedding_Path"
      - "Target_LabelLoss", "Target_LabelIncorporation", "Target_TurnoverRate"
      - "loss_K", "incorporation_K"
      - "Label incorporation_t-1", ...
    """

    def __init__(self, df: pd.DataFrame, input_cls=False):
        """
        Args:
            df: 1行=1タイムポイント形式の DataFrame
            input_cls: 埋め込みに CLSトークンを含むかどうか
        """
        self.input_cls = input_cls

        df = df.rename(columns=lambda c: c.replace(" ", "_").replace("-", "_"))

        # ラグ列を特定 (例: "Label loss_t-1" など)
        self.lag_col_names_loss = [c for c in df.columns if "Label_loss_t_" in c]
        self.lag_col_names_incorp = [
            c for c in df.columns if "Label_incorporation_t_" in c
        ]

        # ペプチド単位で groupby
        grouped = df.groupby("Cleaned_Peptidoform")
        self.peptide_groups = []
        for peptide_id, group_df in grouped:
            # 時系列順にソート (timestep列で昇順)
            group_df = group_df.sort_values("timestep", ascending=True).reset_index(
                drop=True
            )
            if len(group_df) == 8:
                # ちょうど8行あるものだけ self.peptide_groups に追加
                self.peptide_groups.append((peptide_id, group_df))

    def __len__(self):
        # ペプチド単位での数
        return len(self.peptide_groups)

    def __getitem__(self, idx):
        """
        1ペプチド分 (最大 8タイムポイント) を (8, input_dim) にまとめて返す。
        ターゲットも (8,) で返す (Many-to-Many) 例。最後だけ使うなら自由に加工してください。
        """
        peptide_id, group_df = self.peptide_groups[idx]

        # 8ステップ分を蓄えるリスト
        timesteps = []
        peptides = []
        features_x = []
        target_loss_list = []
        target_incorp_list = []
        target_turnover_list = []

        # group_df の各行 ( = 各タイムポイント ) を順番に処理
        for row in group_df.itertuples(index=False):
            # NamedTuple → dict に変換
            row_dict = row._asdict()
            # 1) 埋め込みをロード & 平均プーリング
            embedding_path = row_dict["Peptido_Embedding_Path"]  # 辞書から取得
            embedding_data = np.load(embedding_path)
            if self.input_cls:
                peptido_embedding = embedding_data[
                    "peptido_embedding"
                ]  # (seq_len, 1280) など
            else:
                peptido_embedding = embedding_data["peptido_embedding"][1:]
            # パディング → 平均プーリング
            # peptido_embedding = pad_peptido_embedding(peptido_embedding)  # (26, 1280)
            embedding_mean = peptido_embedding.mean(axis=0)  # (1280,)
            features_x.append(embedding_mean)

            # 3) ターゲットを取得 (row_dict でアクセス)
            target_loss_list.append(row_dict["Target_LabelLoss"])
            target_incorp_list.append(row_dict["Target_LabelIncorporation"])
            target_turnover_list.append(row_dict["Target_TurnoverRate"])
            peptides.append(row_dict["Cleaned_Peptidoform"])
            timesteps.append(row_dict["timestep"])

        # (8, input_dim) に stack
        arr_loss = np.stack(features_x, axis=0)

        # ターゲットも (8,)
        arr_t_loss = np.array(target_loss_list, dtype=np.float32)
        arr_t_incorp = np.array(target_incorp_list, dtype=np.float32)
        arr_t_turnover = np.array(target_turnover_list, dtype=np.float32)

        return {
            "peptide": peptides,
            "timestep": timesteps,
            "features_x": torch.tensor(arr_loss, dtype=torch.float32),
            "target_loss": torch.tensor(arr_t_loss, dtype=torch.float32),  # (seq_len,)
            "target_incorporation": torch.tensor(arr_t_incorp, dtype=torch.float32),
            "target_turnover": torch.tensor(arr_t_turnover, dtype=torch.float32),
        }


class Datasets_AASeq(Dataset):
    """
    複数のターゲット（LabelLoss, LabelIncorporation, TurnoverRate）と
    それに対応するラグ特徴量を持つペプチド時系列データセット。

    Attributes:
        peptides (np.ndarray): ペプチドIDの配列
        embedding_paths (np.ndarray): 各ペプチド埋め込みへのパス
        timesteps (np.ndarray): タイムステップ情報(形: (num_samples,))
        lag_features_loss (np.ndarray): Label Loss用ラグ特徴量
        lag_features_incorporation (np.ndarray): Label Incorporation用ラグ特徴量
        targets_loss (np.ndarray): Label Lossターゲット値
        targets_incorporation (np.ndarray): Label Incorporationターゲット値
        targets_turnover (np.ndarray): TurnoverRateターゲット値
        input_cls (bool): クラス埋め込みを入力に含めるかどうかのフラグ
    """

    def __init__(self, df, input_cls=False):
        self.peptides = df["Cleaned_Peptidoform"].values
        self.embedding_paths = df["Peptido_Embedding_Path"].values
        self.timesteps = df["timestep"].values

        # ラグ特徴量
        self.lag_features_loss = df[
            [col for col in df.columns if "Label loss_t-" in col]
        ].values
        self.lag_features_incorporation = df[
            [col for col in df.columns if "Label incorporation_t-" in col]
        ].values

        # ターゲット値
        self.targets_loss = df["Target_LabelLoss"].values
        self.targets_incorporation = df["Target_LabelIncorporation"].values
        self.targets_turnover = df["Target_TurnoverRate"].values

        self.input_cls = input_cls

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, index, debug=False):
        # 基本情報取得
        peptide = self.peptides[index]
        timestep = self.timesteps[index]

        # ターゲットをテンソル化
        target_loss = torch.tensor(self.targets_loss[index], dtype=torch.float32)
        target_incorporation = torch.tensor(
            self.targets_incorporation[index], dtype=torch.float32
        )
        target_turnover = torch.tensor(
            self.targets_turnover[index], dtype=torch.float32
        )

        # ラグ特徴量をテンソル化
        lag_loss = torch.tensor(self.lag_features_loss[index], dtype=torch.float32)
        lag_incorporation = torch.tensor(
            self.lag_features_incorporation[index], dtype=torch.float32
        )

        # 埋め込みロード＆パディング
        peptido_embedding = self._load_and_pad_embedding(self.embedding_paths[index])

        # ラグ特徴量をシーケンス長に合わせて拡大
        combined_features_loss_tensor = self._combine_embedding_and_lag(
            peptido_embedding, lag_loss
        )
        combined_features_incorporation_tensor = self._combine_embedding_and_lag(
            peptido_embedding, lag_incorporation
        )

        if debug:
            print(f"Index: {index}")
            print(f"Peptide: {peptide}, Timestep: {timestep}")
            print(
                f"Target Loss: {target_loss.item()}, Target Incorporation: {target_incorporation.item()}, Target Turnover: {target_turnover.item()}"
            )
            print(f"Embedding shape: {peptido_embedding.shape}")
            print(f"Lag loss: {lag_loss}, Lag incorporation: {lag_incorporation}")
            print(
                f"Lag loss shape: {lag_loss.shape}, Lag incorporation shape: {lag_incorporation.shape}"
            )
            print(
                f"Combined Loss Features shape: {combined_features_loss_tensor.shape}"
            )
            print(
                f"Combined Incorporation Features shape: {combined_features_incorporation_tensor.shape}"
            )
            input()

        return {
            "peptide": peptide,
            "features_loss": combined_features_loss_tensor,
            "features_incorporation": combined_features_incorporation_tensor,
            "timestep": timestep,
            "target_loss": target_loss,
            "target_incorporation": target_incorporation,
            "target_turnover": target_turnover,
        }

    def _load_and_pad_embedding(self, embedding_path):
        """
        埋め込みをロードし、必要に応じてCLSトークンを取り除き、パディングするヘルパーメソッド。
        """
        embedding_data = np.load(embedding_path)
        if self.input_cls:
            peptido_embedding = embedding_data["peptido_embedding"]
        else:
            peptido_embedding = embedding_data["peptido_embedding"][1:]
        peptido_embedding = pad_peptide_embedding(peptido_embedding)
        return torch.tensor(peptido_embedding, dtype=torch.float32)

    def _combine_embedding_and_lag(self, peptido_embedding, lag_features):
        """
        埋め込みとラグ特徴量を結合するヘルパーメソッド。
        ラグ特徴量をシーケンス長分複製し、(seq_len, embedding_dim + lag_dim)のテンソルを返す。
        """
        seq_len = peptido_embedding.shape[0]
        # (seq_len, lag_dim)
        repeated_lag = lag_features.unsqueeze(0).repeat(seq_len, 1)
        # 結合 (seq_len, embedding_dim + lag_dim)
        combined = torch.cat([peptido_embedding, repeated_lag], dim=1)
        return combined


# 埋め込みベクトルをパディングする関数
def pad_peptide_embedding(embedding, max_length=26):
    current_length = embedding.shape[0]  # 埋め込みベクトルの行数（アミノ酸の数）
    feature_dim = embedding.shape[1]  # 各行の次元数（ここでは1280）
    if current_length < max_length:
        # 0パディングを行う
        padding = np.zeros(
            (max_length - current_length, feature_dim), dtype=embedding.dtype
        )
        padded_embedding = np.vstack([embedding, padding])
    else:
        # 長すぎる場合は切り詰める
        padded_embedding = embedding[:max_length, :]
    return padded_embedding
