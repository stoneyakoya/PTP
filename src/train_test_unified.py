import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.custom_dataset_loader import Datasets_AASeqTimeSeq, collate_fn
from models.unified import STCrossPredictor  # ※ 下記モデル定義に合わせてください


def run_model(
    model,
    train_ds,
    val_ds,
    test_ds,
    device,
    fold_idx: int,
    batch_size=16,
    epochs=20,
    lr=1e-4,
    weight_decay=1e-2,
    patience=3,
    lr_scheduler=True,
):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True,
            threshold=0.001,
        )
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Train {epoch}/{epochs}]"):
            emb = batch["embedding"].to(device)
            mask = batch["mask"].to(device)
            t_idx = batch["t_idx"].to(device)
            y_loss = batch["y_loss"].to(device)
            y_inc = batch["y_inc"].to(device)

            optimizer.zero_grad()
            loss_pred, inc_pred, attn_w = model(emb, mask, t_idx)
            l1 = criterion(loss_pred, y_loss)
            l2 = criterion(inc_pred, y_inc)
            loss = l1 + l2
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * emb.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc=f"[Val {epoch}/{epochs}]"):
            emb = batch["embedding"].to(device)
            mask = batch["mask"].to(device)
            t_idx = batch["t_idx"].to(device)
            y_loss = batch["y_loss"].to(device)
            y_inc = batch["y_inc"].to(device)

            with torch.no_grad():
                loss_pred, inc_pred, attn_w = model(emb, mask, t_idx)
                val_loss += (criterion(loss_pred, y_loss) + criterion(inc_pred, y_inc)).item() * emb.size(0)

        val_loss /= len(val_ds)
        if lr_scheduler:
            scheduler.step(val_loss)
        print(f"Epoch {epoch}: lr={scheduler.get_last_lr()[0]},train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- test eval with turnover calculation ---
    # --- ２．テスト評価部 ---
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_trues = []
    all_clusters = []
    all_peptides = []
    all_tsteps = []

    # clusterごとのr2をfold単位でためる
    cluster_r2 = {}

    with torch.no_grad():
        for batch in tqdm(DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn), desc="[Test]"):
            emb = batch["embedding"].to(device)
            mask = batch["mask"].to(device)
            t_idx = batch["t_idx"].to(device)
            y_loss = batch["y_loss"].to(device)
            y_inc = batch["y_inc"].to(device)
            clusters = batch["cluster"][:, 0].cpu().numpy()  # (B,)
            peptides = batch["peptide"]  # list of length B

            loss_pred, inc_pred, _ = model(emb, mask, t_idx)
            # turnover
            turn_pred = (inc_pred / (loss_pred + inc_pred)).cpu().numpy()  # (B, T)
            turn_true = (y_inc / (y_loss + y_inc)).cpu().numpy()  # (B, T)

            B, T = turn_true.shape
            # flatten and record
            for i in range(B):
                for t in range(T):
                    all_preds.append(turn_pred[i, t])
                    all_trues.append(turn_true[i, t])
                    all_clusters.append(int(clusters[i]))
                    all_peptides.append(peptides[i])
                    all_tsteps.append(int(t_idx[i, t].item()))

    # overall metrics
    preds = np.array(all_preds)
    trues = np.array(all_trues)
    clusters = np.array(all_clusters)

    overall_r2 = r2_score(trues, preds)

    # clusterごとのR2をfoldごとに計算
    for c in np.unique(clusters):
        mask = clusters == c
        cluster_r2[c] = r2_score(trues[mask], preds[mask])

    # foldごとのテーブルを返す
    # plus per-peptide/time CSV
    df_pred = pd.DataFrame(
        {
            "fold": fold_idx,
            "peptide": all_peptides,
            "cluster": all_clusters,
            "timestep_idx": all_tsteps,
            "pred_turnover": all_preds,
            "true_turnover": all_trues,
        }
    )
    os.makedirs("predictions", exist_ok=True)
    df_pred.to_csv(f"predictions/fold_{fold_idx}_peptide_predictions.csv", index=False)

    return overall_r2, cluster_r2

    # # --- analyze attention weights ---
    # time_labels = ["1h", "3h", "6h", "10h", "16h", "24h", "34h", "48h"]
    # save_attention_by_cluster(test_loader, model, device, time_labels, max_length=30)
    # # --- 可視化フェーズ ---
    # output_dir = "data/attention_outputs"
    # time_labels = ["1h", "3h", "6h", "10h", "16h", "24h", "34h", "48h"]

    # # 1) クラスタ集約ヒートマップ
    # for path in glob.glob(f"{output_dir}/cluster_*_aggregate_attention.csv"):
    #     # ファイル名から cluster_id を抜き出す
    #     fname = os.path.basename(path)
    #     cluster_id = int(fname.split("_")[1])
    #     plot_cluster_aggregate(cluster_id, time_labels, output_dir=output_dir)

    # # 2) 各クラスタ×全ペプチドのペプチド別 Attention マップ
    # #    ※表示したいペプチド名が多すぎる場合はサンプル数を絞ってください
    # for path in glob.glob(f"{output_dir}/cluster_*_peptide_attention.csv"):
    #     fname = os.path.basename(path)
    #     cluster_id = int(fname.split("_")[1])
    #     df_detail = pd.read_csv(path)
    #     for pep in df_detail["peptide"].unique():
    #         plot_peptide_attention(cluster_id, pep, time_labels, output_dir=output_dir)

    return test_loss, overall_r2


def save_attention_by_cluster(test_loader, model, device, time_labels, max_length, output_dir="data/attention_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    records = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting attention"):
            emb = batch["embedding"].to(device)
            mask = batch["mask"].to(device)
            t_idx = batch["t_idx"].to(device)
            peptides = batch["peptide"]  # list of length B
            clusters = batch["cluster"][:, 0].cpu().numpy().astype(int)

            # モデル呼び出し
            _, _, attn_w = model(emb, mask, t_idx)  # (B, T, T*L)
            attn_np = attn_w.cpu().numpy()  # numpy に

            B, T, TL = attn_np.shape
            L = TL // T

            # causal filter: only past_time ≤ future_time
            for i in range(B):
                pep = peptides[i]
                cl = clusters[i]
                w = attn_np[i].reshape(T, T, L)  # (future, past, residue)

                for t_future in range(T):
                    for t_past in range(t_future + 1):  # ← future より大きい past はスキップ
                        for res in range(L):
                            aw = float(w[t_future, t_past, res])
                            if aw <= 0.0:  # ← 0 の重みを完全に無視したければ追加
                                continue

                            records.append(
                                {
                                    "cluster": cl,
                                    "peptide": pep,
                                    "future_time": time_labels[t_future],
                                    "past_time": time_labels[t_past],
                                    "residue_idx": res + 1,
                                    "attention_weight": aw,
                                }
                            )

    df = pd.DataFrame(records)

    # クラスタ別ペプチド詳細
    for cl, group in df.groupby("cluster"):
        group.to_csv(f"{output_dir}/cluster_{cl}_peptide_attention.csv", index=False)

    # クラスタ別アグリゲート (future_time × residue_idx)
    for cl, group in df.groupby("cluster"):
        avg = group.groupby(["future_time", "residue_idx"])["attention_weight"].mean().reset_index()
        avg.to_csv(f"{output_dir}/cluster_{cl}_aggregate_attention.csv", index=False)

    print(f"Saved attention CSVs to {output_dir}")


# Example usage:


if __name__ == "__main__":
    mses = []
    r2s = []
    # 10回の実行を行う
    for i in range(0, 10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{i}/train_fold.csv")
        val_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{i}/validation_fold.csv")
        test_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{i}/test_fold.csv")

        train_ds = Datasets_AASeqTimeSeq(train_df)
        val_ds = Datasets_AASeqTimeSeq(val_df)
        test_ds = Datasets_AASeqTimeSeq(test_df)

        cfg = {
            "d_in": 1280,
            "d_model": 256,
            "time_emb_dim": 8,
            "T": 8,
            "num_layers": 3,
            "n_head": 4,
            "dim_ff": 512,
            "dropout": 0.1,
        }
        model = STCrossPredictor(**cfg).to(device)

        test_loss, overall_r2 = run_model(
            model,
            train_ds,
            val_ds,
            test_ds,
            fold_idx=i,
            device=device,
            batch_size=8,
            epochs=10,
            lr=0.0001,
            weight_decay=1e-2,
            patience=10,
            lr_scheduler=True,
        )
        mses.append(test_loss)
        r2s.append(overall_r2)
        print(f"Fold {i}: Test MSE Loss: {test_loss:.4f}, Overall R2: {overall_r2:.4f}")
    mse_mean = np.mean(mses)
    r2_mean = np.mean(r2s)
    print(f"Average Test MSE Loss: {mse_mean:.4f}, Overall R2: {r2_mean:.4f}")
