import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader

from datasets.custom_dataset_loader import Datasets_AASeq  # adjust import
from models.Transformer_based import Transformer_AASeq  # adjust import


def load_model(model_path, device, input_dim, embed_dim, n_heads, num_layers, dropout, activation_func):
    model = Transformer_AASeq(
        input_dim=input_dim,
        embed_dim=embed_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        activation_func=activation_func,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_attributions(model, dataloader, device, baseline=None):
    model.eval()

    def fusion_forward(loss_feat, inc_feat):
        # モデル本体を呼び出し
        out_loss, out_inc, _, _, _, _ = model(loss_feat, inc_feat)
        # out_loss, out_inc は both (batch, 1)
        # dim=1 を sum すると (batch,) になる
        # print(out_loss.shape, out_inc.shape)
        # input()
        return out_loss.squeeze() + out_inc.squeeze()

    ig = IntegratedGradients(fusion_forward)
    all_attrs, all_clusters, all_peptides = [], [], []

    for batch in dataloader:
        loss_feat = batch["features_loss"].to(device)
        inc_feat = batch["features_incorporation"].to(device)
        cluster = batch["cluster"].cpu().numpy()
        peptides = batch["peptide"]
        # print(peptides)
        # print(cluster)
        # print(loss_feat.shape, inc_feat.shape)

        inputs = (loss_feat, inc_feat)
        if baseline is None:
            baseline = tuple(torch.zeros_like(x) for x in inputs)

        # このとき出力は (batch,) のベクトルなので IG が通る
        attributions = ig.attribute(inputs, baselines=baseline)
        # print(attributions)
        # print(attributions[0].shape, attributions[1].shape)
        # input()
        # attributions はタプル(loss_attr, inc_attr)、両方とも同じ shape
        loss_attr, inc_attr = attributions
        # abs して足し合わせて次元圧縮
        # per_sample_attr = (loss_attr.abs() + inc_attr.abs()).sum(dim=2)  # (batch, feature_dim)
        per_sample_attr = loss_attr.abs().sum(dim=2)  # (batch, feature_dim)
        for i in range(per_sample_attr.size(0)):
            all_attrs.append(per_sample_attr[i].cpu().numpy())
            all_clusters.append(cluster[i])
            all_peptides.append(peptides[i])

    return pd.DataFrame({"Peptide": all_peptides, "Cluster": all_clusters, "Attribution": all_attrs})


def count_amino_acid_frequency(peptides):
    counters = []
    for seq in peptides:
        aa = list(seq)
        freq = Counter(aa)
        counters.append(freq)
    return counters


def compute_dataset_mean_baseline(df, device, batch_size=32):
    """
    DataFrame 全体から features_loss と features_incorporation の平均ベクトルを計算して返す。
    """
    ds_all = Datasets_AASeq(df, input_cls=False)
    loader_all = DataLoader(ds_all, batch_size=batch_size, shuffle=False)

    sum_loss = 0
    sum_inc = 0
    count = 0

    for batch in loader_all:
        loss_feat = batch["features_loss"].to(device)  # (B, L, D)
        inc_feat = batch["features_incorporation"].to(device)  # (B, L, D)
        sum_loss += loss_feat.sum(dim=0)  # (L,D)
        sum_inc += inc_feat.sum(dim=0)  # (L,D)
        count += loss_feat.shape[0]

    mean_loss = sum_loss / count
    mean_inc = sum_inc / count
    return mean_loss, mean_inc


def main():
    compute = True
    if compute:
        # PARAMETERS
        model_path = "data/models/model_2025_0525_224813.pth"
        data_csv = "data/dataset/normal/fold_7/test_fold.csv"
        batch_size = 8
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load data
        df = pd.read_csv(data_csv)

        # ① まず全データの平均ベクトルを計算して baseline を作る
        mean_loss, mean_inc = compute_dataset_mean_baseline(df, device, batch_size=batch_size)

        # バッチサイズ分に expand してタプルに
        baseline = (
            mean_loss.unsqueeze(0).repeat(batch_size, 1, 1).to(device),
            mean_inc.unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        )

        dataset = Datasets_AASeq(df, input_cls=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # load model
        model = load_model(
            model_path, device, input_dim=1287, embed_dim=256, n_heads=4, num_layers=2, dropout=0.5, activation_func="ReLU"
        )
        # compute attributions
        attr_df = compute_attributions(model, loader, device, baseline=baseline)
        attr_df["Attribution"] = attr_df["Attribution"].apply(lambda arr: json.dumps(arr.tolist()))
        # save
        os.makedirs("data/results/contributions", exist_ok=True)
        # 保存時
        attr_df.to_feather("data/results/contributions/attributions_loss.feather")

    # Load aggregated cluster contributions
    attr_df = pd.read_feather("data/results/contributions/attributions_loss.feather")

    attr_df["Attribution"] = attr_df["Attribution"].apply(json.loads)
    clusters = attr_df["Cluster"].unique().tolist()
    for cluster in clusters:
        dfc = attr_df[attr_df["Cluster"] == cluster]

        # --- 追加: AA ごとのすべての寄与度をため込む辞書 ---
        aa_attr_values = defaultdict(list)

        # 元々の合計／頻度カウンタ
        aa_attr_sum = Counter()
        aa_freq_sum = Counter()

        for _, row in dfc.iterrows():
            seq = row["Peptide"]
            if len(seq) > 26:
                seq = seq[:26]
            attr_values = row["Attribution"]  # リスト of float

            for pos, aa in enumerate(seq):
                val = attr_values[pos]
                # 合計と頻度
                aa_attr_sum[aa] += val
                aa_freq_sum[aa] += 1
                # すべての値をリストに蓄積
                aa_attr_values[aa].append(val)

        # --- 平均寄与度（もとのまま） ---
        aa_mean_attr = {aa: aa_attr_sum[aa] / aa_freq_sum[aa] for aa in aa_attr_sum}

        # --- ここから中央値・四分位数の計算 ---
        aa_median_attr = {}
        aa_q1_attr = {}
        aa_q3_attr = {}
        for aa, vals in aa_attr_values.items():
            arr = np.array(vals)
            aa_median_attr[aa] = float(np.median(arr))
            aa_q1_attr[aa] = float(np.percentile(arr, 25))
            aa_q3_attr[aa] = float(np.percentile(arr, 75))

        # Sort amino acids by average attribution descending
        sorted_items = sorted(aa_median_attr.items(), key=lambda x: -x[1])
        aas = [item[0] for item in sorted_items]
        mean_attrs = [item[1] for item in sorted_items]
        freqs = [aa_freq_sum[aa] for aa in aas]

        x = np.arange(len(aas))
        width = 0.4

        fig, ax1 = plt.subplots(figsize=(10, 5))
        bars1 = ax1.bar(x - width / 2, mean_attrs, width, label="Avg Attribution", color="tab:blue")
        ax1.set_xlabel("Amino Acid")
        ax1.set_ylabel("Average Attribution")
        ax1.set_xticks(x)
        ax1.set_xticklabels(aas, rotation=45)

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, freqs, width, label="Frequency", color="tab:orange", alpha=0.7)
        ax2.set_ylabel("Frequency")

        # Combined legend
        handles = [bars1, bars2]
        labels = [h.get_label() for h in handles]
        fig.legend(handles, labels, loc="upper right")

        plt.title(f"Cluster {cluster}: Attribution vs Frequency")
        plt.tight_layout()
        plt.savefig(f"data/results/contributions/loss_cluster_{cluster}_combined.png")
        plt.show()


if __name__ == "__main__":
    main()
