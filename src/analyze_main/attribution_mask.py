import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from datasets.custom_dataset_loader import Datasets_AASeq
from models.Transformer_based import Transformer_AASeq
import pickle
import re
from scipy.stats import mannwhitneyu, brunnermunzel
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
import shap
import random
import ast
from collections import defaultdict
import matplotlib as mpl


COMPUTE = False
COMPUTE_METHOD = 'captum' # 'shap' or 'captum'

# モード切り替え: "peptide" or "protein"
MODE = "protein"  # "peptide" にするとペプチド内インデックスベース,
                  # "protein" にするとタンパク質絶対位置ベースになります。
# タンパク質絶対位置ベースのとき
MAX_SEQ_LEN = 1022  # ESM2 の最大入力長（必要に応じて変える）
# 両端の可視化残基数（N 末端と C 末端）
N_TERM = 10   # N 末端から何残基表示するか
C_TERM = 10   # C 末端から何残基表示するか
# 可視化対象のクラスター番号
TARGET_CLUSTER = 4
# データ・モデル関連パス
MODEL_PATH = "data/models/model_2025_0529_121518.pth"
PICKLE_PATH = f"data/results/contributions/attributions_simple_allfold_cluster{TARGET_CLUSTER}.pickle"
MIDDLE_PCT = 0.3
PATTERNS = {
    'pest': [
        re.compile(r'[PEST]{6,}'),
        ['P','E','S','T']]

}




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

def compute_dataset_baseline(
    df: pd.DataFrame,
    device,
    batch_size: int = 32,
    mode: str = "global",          # "global" | "cluster" | "random" | "zero"
    cluster_col: str = "Cluster",
    target_cluster: int = None,    # mode=="cluster" のとき必須
    n_random: int = 4,             # mode=="random" のとき何個サンプル平均を作るか
):
    """
    ベースライン埋め込みを返す関数。
    戻り値:
        - mode=="zero" 以外:  (mean_loss, mean_inc)  or  [(loss1, inc1), (loss2, inc2), ...]
        - mode=="zero":      (torch.zeros_like(...), torch.zeros_like(...))
    """
    ds_all = Datasets_AASeq(df, input_cls=False)
    loader = DataLoader(ds_all, batch_size=batch_size, shuffle=False)

    if mode == "zero":
        sample_batch = next(iter(loader))
        z_loss = torch.zeros_like(sample_batch["features_loss"][0]).to(device)
        z_inc  = torch.zeros_like(sample_batch["features_incorporation"][0]).to(device)
        return z_loss, z_inc

    # ---------- 平均計算用ヘルパ ----------
    def _mean_over_loader(_loader):
        s_loss, s_inc, n = 0, 0, 0
        for batch in _loader:
            s_loss += batch["features_loss"].to(device).sum(dim=0)
            s_inc  += batch["features_incorporation"].to(device).sum(dim=0)
            n += batch["features_loss"].shape[0]
        return s_loss / n, s_inc / n
    # -------------------------------------

    if mode == "global":
        return _mean_over_loader(loader)

    elif mode == "cluster":
        if target_cluster is None:
            raise ValueError("mode='cluster' では target_cluster を指定してください")
        sub_df = df[df[cluster_col] == target_cluster]
        sub_loader = DataLoader(Datasets_AASeq(sub_df, input_cls=False),
                                batch_size=batch_size, shuffle=False)
        return _mean_over_loader(sub_loader)

    elif mode == "random":
        baselines = []
        for _ in range(n_random):
            sample_df = df.sample(frac=0.1, replace=False)  # 例：全体の10%をランダム抽出
            sample_loader = DataLoader(Datasets_AASeq(sample_df, input_cls=False),
                                       batch_size=batch_size, shuffle=False)
            baselines.append(_mean_over_loader(sample_loader))
        return baselines

    else:
        raise ValueError(f"unsupported mode: {mode}")

def compute_attributions(model, dataloader, device, baseline=None):
    """
    IntegratedGradients を使って寄与度を計算し、
    DataFrame("Peptide", "Protein", "Cluster", "Attribution") を返す。
    """
    model.eval()

    def fusion_forward(loss_feat, inc_feat):
        out_loss, out_inc, _, _, _, _ = model(loss_feat, inc_feat)
        return out_loss.squeeze() + out_inc.squeeze()

    ig = IntegratedGradients(fusion_forward)
    all_attrs, all_clusters, all_peptides, all_proteins = [], [], [], []

    for batch in dataloader:
        loss_feat = batch["features_loss"].to(device)
        inc_feat = batch["features_incorporation"].to(device)
        cluster = batch["cluster"].cpu().numpy()
        peptides = batch["peptide"]
        proteins = batch["proteins"]
        inputs = (loss_feat, inc_feat)
        if baseline is None:
            baseline = tuple(torch.zeros_like(x) for x in inputs)

        attributions = ig.attribute(inputs, baselines=baseline)
        loss_attr, inc_attr = attributions
        # print(loss_attr.shape)
        # print(inc_attr.shape)
        # input()

        per_sample_attr = loss_attr.abs() + inc_attr.abs()
        # 最初の1280次元だけ使う
        per_sample_attr = per_sample_attr[:, :, :1280]  # → shape: (batch, seq_len, 1280)
        per_sample_attr = per_sample_attr.max(dim=2).values  # shape: (batch, seq_len)

        # Combined attribution (loss + incorporation)
        # per_sample_attr = (loss_attr.abs() + inc_attr.abs()).sum(dim=2)  # (batch, feature_dim)

        # per_sample_attr = inc_attr.abs().sum(dim=2)
        for i in range(per_sample_attr.size(0)):
            all_attrs.append(per_sample_attr[i].cpu().numpy())
            all_clusters.append(cluster[i])
            all_peptides.append(peptides[i])   # ※ ここが「Peptide」列になりますが、
                                              #    ユーザー環境では「Peptide に Protein の配列が入っている」とのこと。
            all_proteins.append(proteins[i])
            
    return pd.DataFrame({
        "Peptide": all_peptides,
        "Protein": all_proteins,
        "Cluster": all_clusters,
        "Attribution": all_attrs,
    })


    model.eval()
    fusion_model = FusionWrapper(model).to(device)

    def stack_inputs(loss_t, inc_t):
        return torch.stack([loss_t, inc_t], dim=1)  # (batch, 2, steps, dim)

    # --- 背景データ収集 ---
    bg_loss, bg_inc = [], []
    for batch in dataloader:
        bg_loss.append(batch["features_loss"].to(device))
        bg_inc.append(batch["features_incorporation"].to(device))
        if len(torch.cat(bg_loss)) >= n_background:
            break
    background = stack_inputs(torch.cat(bg_loss)[:n_background],
                              torch.cat(bg_inc)[:n_background])

    explainer = shap.DeepExplainer(fusion_model, background)

    # --- SHAP 実行 ---
    all_rows = []
    for batch in dataloader:
        loss = batch["features_loss"].to(device)
        inc  = batch["features_incorporation"].to(device)
        stacked = stack_inputs(loss, inc)  # (batch, 2, steps, dim)

        shap_vals = explainer.shap_values(stacked)[0]  # same shape
        per_attr  = np.abs(shap_vals).sum(axis=(1, 3))  # (batch, steps)

        for i in range(per_attr.shape[0]):
            all_rows.append({
                "Peptide":     batch["peptide"][i],
                "Protein":     batch["proteins"][i],
                "Cluster":     batch["cluster"][i].item(),
                "Attribution": per_attr[i].tolist()
            })

    return pd.DataFrame(all_rows)

# =========================================
# ドメインとモチーフの寄与度を比較
def all_fold_results(result_base_dir='data/results/03_AASeq_Transformer_based_sampling', n_folds=10):
     # 1-2) 各 fold_i の結果をまとめて RMSE 計算
    all_fold_results = []
    for i in range(n_folds):
        path_i = os.path.join(result_base_dir, f'fold_{i}', 'non_recursive.csv')
        if not os.path.exists(path_i):
            raise FileNotFoundError(f"Fold{i} の結果ファイルが見つかりません: {path_i}")
        df_i = pd.read_csv(path_i)
        # Predictions_Turnover が文字列 "[0.123]" のような形式になっている想定 → 数値に変換
        df_i['Predictions_Turnover'] = df_i['Predictions_Turnover'].apply(lambda x: float(x.strip('[]')))
        # ペプチドID列は 'Peptide_ID' というカラム名で揃えておく（ファイル内で違う場合は修正）
        if 'Peptide_ID' not in df_i.columns and 'Cleaned_Peptidoform' in df_i.columns:
            df_i.rename(columns={'Cleaned_Peptidoform': 'Peptide_ID'}, inplace=True)
        # 二乗誤差を計算
        df_i['Squared_Error'] = (df_i['True_Values_Turnover'] - df_i['Predictions_Turnover']) ** 2
        all_fold_results.append(df_i)

    # 結合してペプチドごとに平均 RMSE を算出
    df_concat = pd.concat(all_fold_results, axis=0, ignore_index=True)
    per_peptide_rmse = (
        df_concat
        .groupby('Peptide_ID')['Squared_Error']
        .mean()
        .reset_index()
        .rename(columns={'Squared_Error': 'RMSE'})
    )
    return per_peptide_rmse


def all_fold_r2(result_base_dir='data/results/03_AASeq_Transformer_based_sampling', n_folds=10):
    dfs = []
    for i in range(n_folds):
        csv_path = os.path.join(result_base_dir, f'fold_{i}', 'non_recursive.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Fold {i}: {csv_path} が見つかりません')
        df_i = pd.read_csv(csv_path)

        # 必要に応じて Peptide_ID 列名を統一
        if 'Peptide_ID' not in df_i.columns and 'Cleaned_Peptidoform' in df_i.columns:
            df_i.rename(columns={'Cleaned_Peptidoform': 'Peptide_ID'}, inplace=True)

        # Predictions_Turnover: 1値リスト "[0.123]" → float
        df_i['Predictions_Turnover'] = df_i['Predictions_Turnover'].apply(
            lambda x: float(x.strip('[]')) if isinstance(x, str) else x
        )

        dfs.append(df_i)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['Peptide'] = df_all['Peptide_ID']

    # groupby してリスト化
    grouped = df_all.groupby('Peptide').agg({
        'True_Values_Turnover': lambda x: list(x),
        'Predictions_Turnover': lambda x: list(x)
    }).reset_index()

    # R² を計算
    def compute_r2(row):
        true = row['True_Values_Turnover']
        pred = row['Predictions_Turnover']
        if len(true) < 2 or len(pred) < 2:
            return np.nan
        try:
            return r2_score(true, pred)
        except Exception:
            return np.nan

    grouped['R2'] = grouped.apply(compute_r2, axis=1)
    return grouped[['Peptide', 'R2']]


def extract_and_sort_motif_peptides(attr_df, motif, top_n=None):
    """
    attr_df: Peptide, Attribution カラムを持つ DataFrame
    motif: PATTERNS のキー
    top_n: 上位何件だけ返すか（None なら全件）
    戻り値: motif スコア順にソートされた DataFrame (Peptide, score)
    """
    pattern, aa_list, *rest = PATTERNS[motif]
    records = []
    for _, row in attr_df.iterrows():
        pep = row["Peptide"]
        # if len(pep) > 26:
        #     pep = pep[:26]
        attr = np.array(row["Attribution"], dtype=float)
        matches = list(pattern.finditer(pep))
        # モチーフが複数回マッチする場合はすべて集計
        for m in matches:
            idxs = list(range(m.start(), m.end()))
            # 属性ベクトルの長さよりはみ出したらスキップ
            if max(idxs) >= len(attr): 
                continue
            if motif == 'd_box':
                idxs = [m.start(), m.start()+3, m.start()+8]
            score = attr[idxs].mean()  # 平均値
            records.append({
                "Peptide": pep,
                "motif": motif,
                "start": m.start(),
                "end": m.end(),
                "score": score
            })

    df_scores = pd.DataFrame(records)
    df_pep_mean = (
        df_scores.groupby("Peptide")["score"].mean().reset_index(name="score_mean")
    )
    if df_pep_mean.empty:
        print(f"[X] モチーフ '{motif}' を含むペプチドが見つかりませんでした")
        return df_scores

    df_pep_mean = df_pep_mean.sort_values("score_mean", ascending=False).reset_index(drop=True)
    if top_n:
        return df_pep_mean.head(top_n)
    return df_pep_mean


def mask_motif_in_batch(batch, motif_pattern):
    """
    batch: Dataloader から出てくる dict
    motif_pattern: re.compile オブジェクト
    特定ペプチド中のマッチ領域の time-step を
    ゼロではなくペプチド全体の特徴ベクトル平均でマスクする
    """
    peptides = batch["peptide"]
    loss = batch["features_loss"]           # shape: (batch, seq_len, dim)
    inc  = batch["features_incorporation"]  # shape: (batch, seq_len, dim)

    # 各サンプルごとの平均ベクトルを計算 (seq_len 次元で平均)
    # 形状を揃えておくため keepdim=True
    mean_loss = loss.mean(dim=1, keepdim=True)  # (batch, 1, dim)
    mean_inc  = inc.mean(dim=1, keepdim=True)   # (batch, 1, dim)

    for i, pep in enumerate(peptides):
        for m in motif_pattern.finditer(pep):
            m_span = list(range(m.start(), m.end()))
            other_span = [i for i in range(len(pep)) if i not in m_span]
            for pos in m_span:
                if pos < loss.size(1):
                    # マスク領域に「そのサンプルの平均ベクトル」を代入
                    loss[i, pos, :] = mean_loss[i, 0, :]
                    inc[i, pos, :]  = mean_inc[i,  0, :]
                    # loss[i, pos, :] = 0
                    # inc[i, pos, :]  = 0

    batch["features_loss"] = loss
    batch["features_incorporation"] = inc
    return batch

def evaluate_with_mask(model, df, motif_regex, device, batch_size=8):
    """
    ① df: テストデータフレーム（Peptide_ID, True_Values_Turnover, etc. を含む）
    ② motif_regex: マスク対象の re.compile
    → マスクあり予測の per-peptide R² を返す DataFrame
    """
    ds = Datasets_AASeq(df, input_cls=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    preds_dict = defaultdict(list)
    true_dict  = defaultdict(list)
    
    model.eval()
    for batch in loader:
        batch = mask_motif_in_batch(batch, motif_regex)
        loss_feat = batch["features_loss"].to(device)
        inc_feat  = batch["features_incorporation"].to(device)
        out_loss, out_inc, _, _, _, _ = model(loss_feat, inc_feat)
        y_hat = (out_inc.squeeze() / (out_loss.squeeze() + out_inc.squeeze())).detach().cpu().numpy()
        y_true = batch["target_turnover"].numpy()  # DataLoader 側で True_Values_Turnover を "true_values" にセットしておく
        peptides = batch["peptide"]
        # 各ペプチドに対応づけ
        for pep, yt, yp in zip(peptides, y_true, y_hat):
            true_dict[pep].append(float(yt))
            preds_dict[pep].append(float(yp))
    
    
    
    # ペプチドごとに R2 を計算
    rows = []
    for pep in true_dict:
        t = true_dict[pep]
        p = preds_dict[pep]
        if len(t)>=2:
            r2 = r2_score(t, p)
            rows.append({"Peptide": pep, "R2_masked": r2})
    return pd.DataFrame(rows)


# =========================================
def main():
    # 1) 寄与度データを読み込む（compute=Trueなら再計算）
    
    if COMPUTE:
        # 10 フォールド分のテスト CSV を読み込み
        df_list = []
        for i in range(0,10):
            test_csv = f"data/dataset/sampling/fold_{i}/test_fold.csv"
            df = pd.read_csv(test_csv)
            df_list.append(df)

        df = pd.concat(df_list)
        df = df[df["Cluster"] == TARGET_CLUSTER]

        batch_size = 8
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ベースライン計算
        mean_loss, mean_inc = compute_dataset_baseline(df, device, mode="cluster", target_cluster=TARGET_CLUSTER)

        baseline = (
            mean_loss.unsqueeze(0).repeat(batch_size, 1, 1).to(device),
            mean_inc.unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        )

        # DataLoader 作成
        dataset = Datasets_AASeq(df, input_cls=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # モデルロード
        model = load_model(
            MODEL_PATH, device,
            input_dim=1287, embed_dim=512,
            n_heads=4, num_layers=2,
            dropout=0.2, activation_func="ReLU"
        )

        # 寄与度計算
        # attr_df = compute_attributions(model, loader, device, n_background=50)
        attr_df = compute_attributions(model, loader, device, baseline=baseline)

        # pickle 保存
        attr_df["Attribution"] = attr_df["Attribution"].apply(lambda arr: json.dumps(arr.tolist()))
        os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(attr_df, f)

    # compute=False の場合は pickle をロード
    with open(PICKLE_PATH, "rb") as f:
        attr_df = pickle.load(f)
    attr_df["Attribution"] = attr_df["Attribution"].apply(lambda x: json.loads(x))

    r2_df = all_fold_r2()
    attr_df = attr_df.merge(r2_df, on='Peptide', how='left')
    print(attr_df)

    r2_df['R2_rank'] = r2_df['R2'].rank(ascending=False, pct=True)
    attr_df = attr_df.merge(r2_df[['Peptide', 'R2', 'R2_rank']], on='Peptide', how='left')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device, input_dim=1287, embed_dim=512, n_heads=4, num_layers=2, dropout=0.2, activation_func="ReLU")
    # 全ペプチドの test_df を用意
    test_df = pd.concat([pd.read_csv(f"data/dataset/sampling/fold_{i}/test_fold.csv") for i in range(10)])
    test_df = test_df[test_df["Cluster"]==TARGET_CLUSTER]
    
    for motif, (regex, *_) in PATTERNS.items():
        print(motif)
        df_pep_mean = extract_and_sort_motif_peptides(attr_df, motif=motif) # ["Peptide", "score_mean"]
        df_merge = df_pep_mean.merge(r2_df, on="Peptide", how="inner")
        rho, pval = spearmanr(df_merge["score_mean"], df_merge["R2"], nan_policy="omit")
        print(f"Spearman ρ = {rho:.3f}, p = {pval:.3e}")

        # ===============================================================
        # ① motif スコアと元の R² の関係を図示（scatter＋回帰線＋相関係数）
        # ---------------------------------------------------------------
        df_scatter = df_merge.dropna(subset=["score_mean", "R2"])
        rho, pval = spearmanr(df_scatter["score_mean"], df_scatter["R2"])

        import matplotlib as mpl
        mpl.rcParams.update({
            "pdf.fonttype": 42,  # フォント埋め込みをTrueTypeに
            "ps.fonttype": 42,
            "axes.linewidth": 1.2,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        })

        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        sns.regplot(
            x="score_mean", y="R2", data=df_scatter,
            ax=ax, scatter_kws=dict(s=40, alpha=.7, edgecolor="black"),
            line_kws=dict(color="royalblue", linewidth=2)
        )
        ax.set_xlabel("Motif attribution score (mean)")
        
        ax.set_ylabel(r"$R^{2}$")
        ax.set_title(f"{motif.upper()} motif contribution to R^2$", 
                     fontweight="bold", pad=10)
        ax.text(
            0.05, 0.92,
            fr"$\rho={rho:.2f},\;p={pval:.1e}$",
            transform=ax.transAxes,
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray")
        )
        plt.tight_layout()

        out_path = f"data/results/attribution_mask/{motif}_score_vs_R2.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[Saved] {out_path}")
        # ===============================================================




        print('='*100)
        input()
        print(f"--- Evaluate masking for motif: {motif} ---")

        attr_df = attr_df[attr_df['R2_rank'] < MIDDLE_PCT]
        df_pep_mean = extract_and_sort_motif_peptides(attr_df, motif=motif)
        peptides_to_keep = set(df_pep_mean["Peptide"])
        test_df_sub = test_df[test_df["Cleaned_Peptidoform"].isin(peptides_to_keep)].reset_index(drop=True)

        # 3) マスク後の per-peptide R²
        df_masked_r2 = evaluate_with_mask(model, test_df_sub, regex, device)
        # ["Peptide", "R2_masked"]

        print(df_merge)
        print(df_masked_r2)
        input()

        # 4) まとめてマージ
        df_cmp = (
            df_merge
            .merge(df_masked_r2, on="Peptide", how="inner")
        )

        # 5) ΔR2 を計算
        df_cmp["delta_R2"] = df_cmp["R2"] - df_cmp["R2_masked"]

        # 6) 結果表示
        print(f"平均 ΔR2 (orig - masked): {df_cmp['delta_R2'].mean():.4f}")
        rho, p = spearmanr(df_cmp["score_mean"], df_cmp["delta_R2"], nan_policy="omit")
        print(f"ΔR2 とモチーフスコアの相関 ρ = {rho:.3f}, p = {p:.3e}")

        # ===============================================================
        # ② Motif attribution と ΔR² の相関を図示
        # ---------------------------------------------------------------
        df_corr = df_cmp.dropna(subset=["score_mean", "delta_R2"])
        rho2, p2 = spearmanr(df_corr["score_mean"], df_corr["delta_R2"])

        mpl.rcParams.update({
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.2,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        })

        df_corr["perc_drop"] = df_corr["delta_R2"] / df_corr["R2"] * 100

        fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=300)
        sns.regplot(
            x="score_mean",
            y="perc_drop",
            data=df_corr,
            ax=ax2,                                   # ← ax ではなく ax2
            scatter_kws=dict(s=40, alpha=.7, edgecolor="black"),
            line_kws=dict(color="royalblue", linewidth=2)
        )

        ax2.set_xlabel("Motif attribution score (mean)")
        ax2.set_ylabel("Relative drop in $R^{2}$ (%)")
        ax2.set_title(f"Impact of {motif.upper()} motif masking on model performance", 
                      fontweight="bold", pad=10)

        rho2, p2 = spearmanr(df_corr["score_mean"], df_corr["perc_drop"])
        ax2.text(
            0.05, 0.92,
            fr"$\rho={rho2:.2f},\;p={p2:.1e}$",
            transform=ax2.transAxes,
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray")
        )

        plt.tight_layout()
        out_path2 = f"data/results/attribution_mask/{motif}_deltaR2_vs_score.pdf"
        fig2.savefig(out_path2, bbox_inches="tight")
        print(f"[Saved] {out_path2}")

        # ===============================================================

        # 7) グループ分け＆ボックスプロット
        df_cmp["group"] = pd.qcut(df_cmp["score_mean"], [0, .3, .7, 1.0], labels=["Low","Mid","High"])
        sns.boxplot(x="group", y="delta_R2", data=df_cmp, order=["Low","Mid","High"])
        plt.title(f"ΔR2 by motif score group ({motif})")
        plt.ylabel("ΔR2 (orig - masked)")
        plt.savefig(f"data/results/attribution_mask/{motif}.png")



if __name__ == "__main__":
    main()
