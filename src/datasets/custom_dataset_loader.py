import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
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
        self.lag_col_names_incorp = [c for c in df.columns if "Label_incorporation_t_" in c]

        # ペプチド単位で groupby
        grouped = df.groupby("Cleaned_Peptidoform")
        self.peptide_groups = []
        for peptide_id, group_df in grouped:
            # 時系列順にソート (timestep列で昇順)
            group_df = group_df.sort_values("timestep", ascending=True).reset_index(drop=True)
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
        proteins = []
        features_x = []
        target_loss_list = []
        target_incorp_list = []
        target_turnover_list = []
        cluster_list = []
        loss_K = []
        incorp_K = []

        # group_df の各行 ( = 各タイムポイント ) を順番に処理
        for row in group_df.itertuples(index=False):
            # NamedTuple → dict に変換
            row_dict = row._asdict()
            # 1) 埋め込みをロード & 平均プーリング
            embedding_path = row_dict["Peptido_Embedding_Path"]  # 辞書から取得
            embedding_data = np.load(embedding_path)
            if self.input_cls:
                peptido_embedding = embedding_data["peptido_embedding"]  # (seq_len, 1280) など
            else:
                peptido_embedding = embedding_data["peptido_embedding"][1:]
            # パディング → 平均プーリング
            # peptido_embedding = pad_peptido_embedding(peptido_embedding)  # (26, 1280)
            embedding_mean = peptido_embedding.mean(axis=0)  # (1280,)
            # embedding_mean = np.concatenate(
            #     [
            #         peptido_embedding.mean(axis=0),
            #         peptido_embedding.std(axis=0),
            #         peptido_embedding.max(axis=0),
            #         peptido_embedding.min(axis=0),
            #     ]
            # )
            features_x.append(embedding_mean)

            # 3) ターゲットを取得 (row_dict でアクセス)
            target_loss_list.append(row_dict["Target_LabelLoss"])
            target_incorp_list.append(row_dict["Target_LabelIncorporation"])
            target_turnover_list.append(row_dict["Target_TurnoverRate"])
            peptides.append(row_dict["Cleaned_Peptidoform"])
            timesteps.append(row_dict["timestep"])

            # 2) ターゲットKを取得
            loss_K.append(row_dict["loss_K"])
            incorp_K.append(row_dict["incorporation_K"])

        # (8, input_dim) に stack
        arr_loss = np.stack(features_x, axis=0)

        # ターゲットも (8,)
        arr_t_loss = np.array(target_loss_list, dtype=np.float32)
        arr_t_incorp = np.array(target_incorp_list, dtype=np.float32)
        arr_t_turnover = np.array(target_turnover_list, dtype=np.float32)
        cluster = group_df.iloc[0]["Cluster"]
        return {
            "peptide": peptides,
            "timestep": timesteps,
            "features_x": torch.tensor(arr_loss, dtype=torch.float32),
            "target_loss": torch.tensor(arr_t_loss, dtype=torch.float32),  # (seq_len,)
            "target_incorporation": torch.tensor(arr_t_incorp, dtype=torch.float32),
            "target_turnover": torch.tensor(arr_t_turnover, dtype=torch.float32),
            "cluster": torch.tensor(cluster, dtype=torch.float32),
            "loss_K": torch.tensor(loss_K[0], dtype=torch.float32),
            "incorp_K": torch.tensor(incorp_K[0], dtype=torch.float32),
        }


class Datasets_TimeSeq_AAC(Dataset):
    """
    AAC (Amino Acid Composition) による特徴量を使った TimeSeq データセット。
    """

    def __init__(self, df: pd.DataFrame, input_cls=False):
        df = df.rename(columns=lambda c: c.replace(" ", "_").replace("-", "_"))

        self.lag_col_names_loss = [c for c in df.columns if "Label_loss_t_" in c]
        self.lag_col_names_incorp = [c for c in df.columns if "Label_incorporation_t_" in c]

        grouped = df.groupby("Cleaned_Peptidoform")
        self.peptide_groups = []
        for peptide_id, group_df in grouped:
            group_df = group_df.sort_values("timestep", ascending=True).reset_index(drop=True)
            if len(group_df) == 8:
                self.peptide_groups.append((peptide_id, group_df))

    def __len__(self):
        return len(self.peptide_groups)

    def __getitem__(self, idx):
        peptide_id, group_df = self.peptide_groups[idx]

        timesteps = []
        peptides = []
        proteins = []
        features_x = []
        target_loss_list = []
        target_incorp_list = []
        target_turnover_list = []
        cluster_list = []
        loss_K = []
        incorp_K = []

        aa_order = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids

        for row in group_df.itertuples(index=False):
            row_dict = row._asdict()

            # AAC特徴量抽出（ペプチド単位）
            pep_seq = row_dict["Cleaned_Peptidoform"]
            protein_seq = row_dict["Protein_Sequence"]

            # pa = ProteinAnalysis(pep_seq)
            # aac_dict = pa.get_amino_acids_percent()
            # aac_vector = [aac_dict.get(aa, 0.0) for aa in aa_order]  # 20次元

            aac_vector = get_local_aac(protein_seq, pep_seq, flank=0)

            features_x.append(aac_vector)

            # ターゲットとメタ情報
            target_loss_list.append(row_dict["Target_LabelLoss"])
            target_incorp_list.append(row_dict["Target_LabelIncorporation"])
            target_turnover_list.append(row_dict["Target_TurnoverRate"])
            peptides.append(row_dict["Cleaned_Peptidoform"])
            proteins.append(row_dict["Protein_Sequence"])
            timesteps.append(row_dict["timestep"])
            loss_K.append(row_dict["loss_K"])
            incorp_K.append(row_dict["incorporation_K"])

        arr_loss = np.stack(features_x, axis=0)
        arr_t_loss = np.array(target_loss_list, dtype=np.float32)
        arr_t_incorp = np.array(target_incorp_list, dtype=np.float32)
        arr_t_turnover = np.array(target_turnover_list, dtype=np.float32)
        cluster = group_df.iloc[0]["Cluster"]

        return {
            "peptide": peptides,
            "proteins": proteins,
            "timestep": timesteps,
            "features_x": torch.tensor(arr_loss, dtype=torch.float32),  # shape: (8, 20)
            "target_loss": torch.tensor(arr_t_loss, dtype=torch.float32),
            "target_incorporation": torch.tensor(arr_t_incorp, dtype=torch.float32),
            "target_turnover": torch.tensor(arr_t_turnover, dtype=torch.float32),
            "cluster": torch.tensor(cluster, dtype=torch.float32),
            "loss_K": torch.tensor(loss_K[0], dtype=torch.float32),
            "incorp_K": torch.tensor(incorp_K[0], dtype=torch.float32),
        }


def get_local_aac(protein_seq: str, peptide_seq: str, flank: int = 5):
    start_idx = protein_seq.find(peptide_seq)
    if start_idx == -1:
        raise ValueError("Peptide not found in protein sequence")

    end_idx = start_idx + len(peptide_seq)
    start = max(0, start_idx - flank)
    end = min(len(protein_seq), end_idx + flank)
    local_seq = protein_seq[start:end]

    pa = ProteinAnalysis(local_seq)
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    aac_dict = pa.get_amino_acids_percent()
    return [aac_dict.get(aa, 0.0) for aa in aa_order]


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
        self.proteins = df["Protein Sequence"].values
        self.embedding_paths = df["Peptido_Embedding_Path"].values
        self.timesteps = df["timestep"].values

        # ラグ特徴量
        self.lag_features_loss = df[[col for col in df.columns if "Label loss_t-" in col]].values
        self.lag_features_incorporation = df[[col for col in df.columns if "Label incorporation_t-" in col]].values

        # ターゲット値
        self.targets_loss = df["Target_LabelLoss"].values
        self.targets_incorporation = df["Target_LabelIncorporation"].values
        self.targets_turnover = df["Target_TurnoverRate"].values

        self.input_cls = input_cls
        self.clusters = df["Cluster"].values

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, index, debug=False):
        # 基本情報取得
        peptide = self.peptides[index]
        proteins = self.proteins[index]
        timestep = self.timesteps[index]
        cluster = torch.tensor(self.clusters[index], dtype=torch.float32)

        # ターゲットをテンソル化
        target_loss = torch.tensor(self.targets_loss[index], dtype=torch.float32)
        target_incorporation = torch.tensor(self.targets_incorporation[index], dtype=torch.float32)
        target_turnover = torch.tensor(self.targets_turnover[index], dtype=torch.float32)

        # ラグ特徴量をテンソル化
        lag_loss = torch.tensor(self.lag_features_loss[index], dtype=torch.float32)
        lag_incorporation = torch.tensor(self.lag_features_incorporation[index], dtype=torch.float32)

        # 埋め込みロード＆パディング
        peptido_embedding = self._load_and_pad_embedding(self.embedding_paths[index])

        # ラグ特徴量をシーケンス長に合わせて拡大
        combined_features_loss_tensor = self._combine_embedding_and_lag(peptido_embedding, lag_loss)
        combined_features_incorporation_tensor = self._combine_embedding_and_lag(peptido_embedding, lag_incorporation)
        if debug:
            print(f"Index: {index}")
            print(f"Peptide: {peptide}, Timestep: {timestep}")
            print(
                f"Target Loss: {target_loss.item()}, Target Incorporation: {target_incorporation.item()}, Target Turnover: {target_turnover.item()}"
            )
            print(f"Embedding shape: {peptido_embedding.shape}")
            print(f"Lag loss: {lag_loss}, Lag incorporation: {lag_incorporation}")
            print(f"Lag loss shape: {lag_loss.shape}, Lag incorporation shape: {lag_incorporation.shape}")
            print(f"Combined Loss Features shape: {combined_features_loss_tensor.shape}")
            print(f"Combined Incorporation Features shape: {combined_features_incorporation_tensor.shape}")
            input()

        return {
            "peptide": peptide,
            "proteins": proteins,
            "features_loss": combined_features_loss_tensor,
            "features_incorporation": combined_features_incorporation_tensor,
            "timestep": timestep,
            "cluster": cluster,
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


class Datasets_AASeq_AAC(Dataset):
    """
    AAC (Amino Acid Composition) 特徴量とラグ特徴量を組み合わせたデータセット。
    Datasets_AASeq の埋め込み版に対して、AAC特徴量を使用する版。
    """

    def __init__(self, df, input_cls=False):
        df = df.rename(columns=lambda c: c.replace(" ", "_").replace("-", "_"))

        # ラグ列名を特定（時系列順に整列）
        self.lag_col_names_loss = [c for c in df.columns if "Label_loss_t_" in c]
        self.lag_col_names_incorp = [c for c in df.columns if "Label_incorporation_t_" in c]

        def _lag_index(col_name: str) -> int:
            # 末尾の整数インデックスを抽出（例: Label_loss_t_1 -> 1）
            try:
                return int(str(col_name).rsplit("_", 1)[-1])
            except Exception:
                return 0

        self.lag_col_names_loss.sort(key=_lag_index)
        self.lag_col_names_incorp.sort(key=_lag_index)

        # ペプチド単位で groupby
        grouped = df.groupby("Cleaned_Peptidoform")
        self.peptide_groups = []
        for peptide_id, group_df in grouped:
            # 時系列順にソート (timestep列で昇順)
            group_df = group_df.sort_values("timestep", ascending=True).reset_index(drop=True)
            if len(group_df) == 8:
                # ちょうど8行あるものだけ self.peptide_groups に追加
                self.peptide_groups.append((peptide_id, group_df))

    def __len__(self):
        return len(self.peptide_groups)

    def __getitem__(self, idx):
        peptide_id, group_df = self.peptide_groups[idx]

        # 8ステップ分を蓄えるリスト
        timesteps = []
        peptides = []
        proteins = []
        features_x = []
        target_loss_list = []
        target_incorp_list = []
        target_turnover_list = []
        cluster_list = []
        loss_K = []
        incorp_K = []

        # group_df の各行 ( = 各タイムポイント ) を順番に処理
        for row in group_df.itertuples(index=False):
            # NamedTuple → dict に変換
            row_dict = row._asdict()
            
            # AAC特徴量抽出（ペプチド単位）
            pep_seq = row_dict["Cleaned_Peptidoform"]
            protein_seq = row_dict["Protein_Sequence"]
            aac_vector = get_local_aac(protein_seq, pep_seq, flank=0)

            # ラグ特徴量（turnover と同様の比率特徴量）を作成
            # ratio_k = lag_incorp_k / (lag_loss_k + lag_incorp_k)
            lag_ratio_vec = []
            for loss_col, inc_col in zip(self.lag_col_names_loss, self.lag_col_names_incorp):
                loss_val = float(row_dict.get(loss_col, 0.0))
                inc_val = float(row_dict.get(inc_col, 0.0))
                denom = loss_val + inc_val
                ratio = (inc_val / denom) if denom > 0 else 0.0
                lag_ratio_vec.append(ratio)

            # AAC(20) + lag_ratio(num_lags) を結合
            combined_vec = np.concatenate(
                [
                    np.asarray(aac_vector, dtype=np.float32),
                    np.asarray(lag_ratio_vec, dtype=np.float32),
                ],
                axis=0,
            )

            features_x.append(combined_vec)

            # ターゲットとメタ情報
            target_loss_list.append(row_dict["Target_LabelLoss"])
            target_incorp_list.append(row_dict["Target_LabelIncorporation"])
            target_turnover_list.append(row_dict["Target_TurnoverRate"])
            peptides.append(row_dict["Cleaned_Peptidoform"])
            proteins.append(row_dict["Protein_Sequence"])
            timesteps.append(row_dict["timestep"])
            loss_K.append(row_dict["loss_K"])
            incorp_K.append(row_dict["incorporation_K"])

        # (8, 20 + num_lags) に stack
        arr_features = np.stack(features_x, axis=0)

        # ターゲットも (8,)
        arr_t_loss = np.array(target_loss_list, dtype=np.float32)
        arr_t_incorp = np.array(target_incorp_list, dtype=np.float32)
        arr_t_turnover = np.array(target_turnover_list, dtype=np.float32)
        cluster = group_df.iloc[0]["Cluster"]

        return {
            "peptide": peptides,
            "proteins": proteins,
            "timestep": timesteps,
            # "features_x": torch.tensor(arr_features, dtype=torch.float32),  # shape: (8, 20 + num_lags)
            "features_loss": torch.tensor(arr_features, dtype=torch.float32),
            "target_loss": torch.tensor(arr_t_loss, dtype=torch.float32),
            "target_incorporation": torch.tensor(arr_t_incorp, dtype=torch.float32),
            "target_turnover": torch.tensor(arr_t_turnover, dtype=torch.float32),
            "cluster": torch.tensor(cluster, dtype=torch.float32),
            "loss_K": torch.tensor(loss_K[0], dtype=torch.float32),
            "incorp_K": torch.tensor(incorp_K[0], dtype=torch.float32),
        }


# 埋め込みベクトルをパディングする関数
def pad_peptide_embedding(embedding, max_length=26):
    current_length = embedding.shape[0]  # 埋め込みベクトルの行数（アミノ酸の数）
    feature_dim = embedding.shape[1]  # 各行の次元数（ここでは1280）
    if current_length < max_length:
        # 0パディングを行う
        padding = np.zeros((max_length - current_length, feature_dim), dtype=embedding.dtype)
        padded_embedding = np.vstack([embedding, padding])
    else:
        # 長すぎる場合は切り詰める
        padded_embedding = embedding[:max_length, :]
    return padded_embedding


# 残基ごとにattentionで予測するモデルのためのデータセットローダー
class Datasets_AASeqTimeSeq(Dataset):
    """
    ペプチドごとに、
      ・残基ごとの埋め込み (1280次元) を max_length=30 までパディング／トリム
      ・８ステップ (1h,3h,6h,10h,16h,24h,36h,48h) の時刻埋め込みインデックス
      ・対応する turnoverRate のターゲットベクトル (長さ8)
    を返すデータセット。
    """

    def __init__(self, df, max_length: int = 30, time_steps: list = None, input_cls=False):
        """
        df: 各行が (Cleaned_Peptidoform, Peptido_Embedding_Path, timestep, Target_TurnoverRate) をもつDataFrame
        max_length: 残基の最大長 (padding/truncate)
        time_steps: モデルで扱う8ステップの list, 例 [1,3,6,10,16,24,36,48]
        in
        """
        self.input_cls = input_cls
        if time_steps is None:
            time_steps = [1, 3, 6, 10, 16, 24, 34, 48]
        self.max_length = max_length
        self.time2idx = {t: i for i, t in enumerate(time_steps)}  # 時刻→埋め込みインデックス
        self.seq = []
        # １グループ（ペプチド＋埋め込みパス）あたり必ず 8行ずつ時間順に並んでいる前提で処理
        grp = df.groupby(["Cleaned_Peptidoform", "Peptido_Embedding_Path"], sort=False)
        for (pep, emb_path), sub in grp:
            # 素直に timestep 昇順ソートして、8行まとめる
            sub_sorted = sub.sort_values("timestep")
            if len(sub_sorted) != len(time_steps):
                # 8行揃っていなければスキップ or 補完ロジック
                continue

            # ターゲットとタイムステップ index を配列化
            targets_LabelLoss = sub_sorted["Target_LabelLoss"].values.astype(np.float32)  # (8,)
            targets_LabelIncorporation = sub_sorted["Target_LabelIncorporation"].values.astype(np.float32)  # (8,)
            targets_TurnoverRate = sub_sorted["Target_TurnoverRate"].values.astype(np.float32)  # (8,)
            timesteps = sub_sorted["timestep"].map(self.time2idx).values.astype(np.int64)  # (8,)
            cluster = sub_sorted["Cluster"].values.astype(np.float32)  # (8,)
            self.seq.append(
                {
                    "peptide": pep,
                    "emb_path": emb_path,
                    "timesteps": timesteps,
                    "targets_LabelLoss": targets_LabelLoss,
                    "targets_LabelIncorporation": targets_LabelIncorporation,
                    "targets_TurnoverRate": targets_TurnoverRate,
                    "cluster": cluster,
                }
            )
        # デバッグ用
        if len(self.seq) == 0:  # 何も取得できなかった場合
            raise ValueError("No valid peptide sequences found in the DataFrame.")

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        item = self.seq[idx]
        # 埋め込み読み込み & CLS除去＋pad/truncate はそのまま
        emb = np.load(item["emb_path"])["peptido_embedding"]
        if not self.input_cls:
            emb = emb[1:]  # CLSトークン除去
        L = emb.shape[0]

        if L < self.max_length:
            pad = np.zeros((self.max_length - L, emb.shape[1]), dtype=emb.dtype)
            emb = np.vstack([emb, pad])
        else:
            emb = emb[: self.max_length]
        emb = torch.from_numpy(emb).float()  # (max_length, 1280)

        # 時刻インデックス
        t_idx = torch.from_numpy(item["timesteps"]).long()  # (8,)

        # 追加したターゲット
        y_loss = torch.from_numpy(item["targets_LabelLoss"]).float()  # (8,)
        y_inc = torch.from_numpy(item["targets_LabelIncorporation"]).float()  # (8,)
        y_turn = torch.from_numpy(item["targets_TurnoverRate"]).float()  # (8,)
        cluster = torch.from_numpy(item["cluster"]).float()  # (8,)
        length = min(L, self.max_length)
        mask = (torch.arange(self.max_length) < length).to(torch.bool)  # shape=(max_length,)

        return {
            "peptide": item["peptide"],
            "embedding": emb,  # (max_length, 1280)
            "mask": mask,  # (max_length,)
            "t_idx": t_idx,  # (8,)
            "y_loss": y_loss,  # (8,)
            "y_inc": y_inc,  # (8,)
            "y_turn": y_turn,  # (8,)
            "cluster": cluster,  # (8,)
        }


def collate_fn(batch):
    peptides = [b["peptide"] for b in batch]
    embeddings = torch.stack([b["embedding"] for b in batch], dim=0)  # (B, L, 1280)
    masks = torch.stack([b["mask"] for b in batch], dim=0)  # (B, L)
    t_idxs = torch.stack([b["t_idx"] for b in batch], dim=0)  # (B, 8)
    y_loss = torch.stack([b["y_loss"] for b in batch], dim=0)  # (B, 8)
    y_inc = torch.stack([b["y_inc"] for b in batch], dim=0)  # (B, 8)
    y_turn = torch.stack([b["y_turn"] for b in batch], dim=0)  # (B, 8)
    clusters = torch.stack([b["cluster"] for b in batch], dim=0)  # (B, 8)

    return {
        "embedding": embeddings,
        "mask": masks,
        "t_idx": t_idxs,
        "y_loss": y_loss,
        "y_inc": y_inc,
        "y_turn": y_turn,
        "cluster": clusters,
        "peptide": peptides,
    }


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    df_path = "data/dataset/sampling/fold_0/train_fold.csv"
    df = pd.read_csv(df_path)
    df = df[:100]
    print(df)
    df = df.rename(columns=lambda c: c.replace(" ", "_").replace("-", "_"))
    time_steps = [1, 3, 6, 10, 16, 24, 34, 48]
    dataset = Datasets_AASeqTimeSeq(df, max_length=30, time_steps=time_steps, input_cls=False)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Define a simple time embedding layer
    num_steps = len(time_steps)
    time_emb_dim = 16
    time_embedding = nn.Embedding(num_steps, time_emb_dim)

    # Test iteration and embedding addition
    for batch in loader:
        peptides = batch["peptide"]  # list of peptide names length B
        embeddings = batch["embedding"]  # (B, L, 1280)
        mask = batch["mask"]  # (B, L)
        t_idx = batch["t_idx"]  # (B, 8)

        # Confirm DataLoader iteration
        print("Batch peptide.shape:", peptides)
        print("Batch embedding shape:", embeddings.shape)
        print("Batch mask shape:", mask.shape)
        print("Batch time idx shape:", t_idx.shape)

        # Time embedding check
        t_emb = time_embedding(t_idx)  # (B, 8, time_emb_dim)
        print("Time embedding shape:", t_emb.shape)

        # Broadcast time embeddings to residue dimension
        # Here we simply add the first time embedding to the first residue token for demonstration
        combined = embeddings[:, 0, :time_emb_dim] + t_emb[:, 0, :]
        print("Combined shape:", combined.shape)

        # Simulate attention mask usage
        # e.g., for residue-level attention scores
        attn_scores = torch.randn(embeddings.size(0), embeddings.size(1), embeddings.size(1))
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        print("Masked attention scores shape:", attn_scores.shape)

        break  # Only run one batch for test
