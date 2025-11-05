import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# import data_loader
from .datasets.custom_dataset_loader import Datasets_AASeq, Datasets_TimeSeq, Datasets_TimeSeq_AAC, pad_peptide_embedding

# import model
from .models.GRU import GRU
from .models.LSTM import LSTM
from .models.Transformer_based import Transformer_AASeq, Transformer_TimeSeq

# check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    model_save_path,
    patience=4,
    threshold=0.01,
    scheduler=None,
    model_type=1,
    target_mode: str = "both",
):
    model.train()
    early_stopping = EarlyStopping(patience=patience, threshold=threshold)
    print("Training started...")
    print(device)

    best_val_loss = float("inf")
    best_model_wts = None

    for epoch in range(num_epochs):
        train_loss = 0.0
        allowed_clusters = get_allowed_clusters(epoch)
        # Trainig loop
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            peptide = batch["peptide"]
            timestep = batch["timestep"]

            if "TimeSeq" in model_type:
                features_x = batch["features_x"].to(device)
                outputs = model(features_x)
                target_loss = batch["target_loss"].to(device)
                target_incorporation = batch["target_incorporation"].to(device)
                target_turnover = batch.get("target_turnover", None)
                if target_turnover is not None:
                    target_turnover = target_turnover.to(device)

                # Early point weight
                timestep_tensor = torch.stack(timestep).float().to(target_loss.device)
                weights = torch.where(
                    timestep_tensor <= 6,
                    torch.tensor(3.0, device=timestep_tensor.device),
                    torch.tensor(1.0, device=timestep_tensor.device),
                )

                if target_mode == "turnover":
                    # direct turnover if provided by model; else derive from loss/incorp
                    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                        pred_turnover = outputs[2]
                    else:
                        eps = 1e-8
                        pred_turnover = outputs[1] / (outputs[1] + outputs[0] + eps)
                    if target_turnover is None:
                        raise ValueError("target_turnover not found in batch for turnover training")
                    loss_turnover = (criterion(pred_turnover, target_turnover.unsqueeze(-1)) * weights).mean()
                    total_loss = loss_turnover
                elif target_mode == "incorp":
                    # incorporation only
                    loss_incorporation = (criterion(outputs[1], target_incorporation.unsqueeze(-1)) * weights).mean()
                    total_loss = loss_incorporation
                else:
                    # multitask: loss + incorporation
                    loss_loss = (criterion(outputs[0], target_loss.unsqueeze(-1)) * weights).mean()
                    loss_incorporation = (criterion(outputs[1], target_incorporation.unsqueeze(-1)) * weights).mean()
                    total_loss = loss_loss + loss_incorporation

            elif "AASeq":
                cluster_ids = batch["cluster"].cpu().numpy()  # shape: [B]
                mask = np.isin(cluster_ids, allowed_clusters)

                if not np.any(mask):
                    continue  # このバッチはすべて除外

                # フィルタ後にバッチを再構築（例: maskを使って選択）
                for key in [
                    "features_loss",
                    "features_incorporation",
                    "target_loss",
                    "target_incorporation",
                    "target_turnover",
                    "timestep",
                ]:
                    if key in batch:
                        batch[key] = batch[key][mask]

                features_loss = batch["features_loss"].to(device)
                features_incorporation = batch["features_incorporation"].to(device)
                # Add noise to lag features
                features_loss, features_incorporation = lag_noise(
                    features_loss, features_incorporation, noise_level=0.02, lag_dim=7
                )
                outputs = model(features_loss, features_incorporation)
                target_loss = batch["target_loss"].to(device)
                target_incorporation = batch["target_incorporation"].to(device)
                target_turnover = batch.get("target_turnover", None)
                if target_turnover is not None:
                    target_turnover = target_turnover.to(device)

                # Early point weight
                timestep_tensor = timestep.float().to(target_loss.device)
                weights = torch.where(
                    timestep_tensor <= 6,
                    torch.tensor(2.0, device=timestep_tensor.device),
                    torch.tensor(1.0, device=timestep_tensor.device),
                )

                if target_mode == "turnover":
                    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                        pred_turnover = outputs[2]
                    else:
                        eps = torch.tensor(1e-8, device=outputs[0].device)
                        pred_turnover = outputs[1] / (outputs[1] + outputs[0] + eps)
                    if target_turnover is None:
                        raise ValueError("target_turnover not found in batch for turnover training")
                    # Ensure shapes match: both (batch, 1)
                    pred_turnover = pred_turnover.view(-1, 1)
                    target_turnover = target_turnover.view(-1, 1)
                    loss_turnover = (criterion(pred_turnover, target_turnover) * weights.view(-1, 1)).mean()
                    total_loss = loss_turnover
                elif target_mode == "incorp":
                    loss_incorporation = (criterion(outputs[1], target_incorporation.unsqueeze(-1)) * weights).mean()
                    total_loss = loss_incorporation
                else:
                    loss_loss = (criterion(outputs[0], target_loss.unsqueeze(-1)) * weights).mean()
                    loss_incorporation = (criterion(outputs[1], target_incorporation.unsqueeze(-1)) * weights).mean()
                    total_loss = loss_loss + loss_incorporation

            # debug
            if i == 0 and epoch == 0:
                print("[DEBUG] batch:", batch)
                print("[DEBUG] peptide =", peptide)
                print("[DEBUG] timestep =", timestep)
                print("-" * 10)
                if "TimeSeq" in model_type:
                    print("[DEBUG] features_x.shape =", features_x.shape)
                elif "AASeq" in model_type:
                    print("[DEBUG] features_loss.shape =", features_loss.shape)
                    print("[DEBUG] features_loss =", features_loss)
                print("-" * 10)
                if target_mode == "turnover":
                    print("[DEBUG] target_turnover.shape =", (target_turnover.shape if target_turnover is not None else None))
                    print("[DEBUG] target_turnover =", target_turnover)
                elif target_mode == "incorp":
                    print("[DEBUG] target_incorporation.shape =", target_incorporation.shape)
                    print("[DEBUG] target_incorporation =", target_incorporation)
                else:
                    print("[DEBUG] target_loss.shape =", target_loss.shape)
                    print("[DEBUG] target_loss =", target_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += total_loss.item()

        # train_loss
        train_loss /= len(train_loader)
        # evaluate_loss (respect training target_mode)
        val_loss = evaluate_model(model, val_loader, criterion, model_type, target_mode)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

        if scheduler is not None:
            scheduler.step(val_loss)

        # debug
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Current Learning Rate: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        model.train()

        # Early Stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # save model
    torch.save(best_model_wts, model_save_path)
    print(f"saved best model: {model_save_path}")


def lag_noise(features_loss, features_incorporation, noise_level=0.02, lag_dim=7):
    # separate lag features
    emb_loss, lag_loss = features_loss[:, :, :-lag_dim], features_loss[:, :, -lag_dim:]
    emb_incorp, lag_incorp = (
        features_incorporation[:, :, :-lag_dim],
        features_incorporation[:, :, -lag_dim:],
    )

    noise_loss = torch.randn_like(lag_loss) * noise_level
    noise_incorp = torch.randn_like(lag_incorp) * noise_level

    # add noise to lag features
    lag_loss_noisy = lag_loss + noise_loss
    lag_incorp_noisy = lag_incorp + noise_incorp

    features_loss = torch.cat([emb_loss, lag_loss_noisy], dim=-1)
    features_incorporation = torch.cat([emb_incorp, lag_incorp_noisy], dim=-1)

    return features_loss, features_incorporation


def get_allowed_clusters(epoch, ranked_clusters=[2, 0, 3, 1, 4]):
    if epoch <= 5:
        return ranked_clusters[:1]
    elif epoch <= 10:
        return ranked_clusters[:2]
    elif epoch <= 15:
        return ranked_clusters[:3]
    else:
        return ranked_clusters


def evaluate_model(model, loader, criterion, model_type, target_mode: str = "both"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if "TimeSeq" in model_type:
                features_x = batch["features_x"].to(device)
                outputs = model(features_x)
                target_loss = batch["target_loss"].to(device)
                target_incorp = batch["target_incorporation"].to(device)
                timestep_tensor = torch.stack(batch["timestep"]).float().to(device)
                weights = torch.where(
                    timestep_tensor <= 6,
                    torch.tensor(2.0, device=timestep_tensor.device),
                    torch.tensor(1.0, device=timestep_tensor.device),
                )
                if target_mode == "turnover":
                    target_turn = batch.get("target_turnover", None)
                    if target_turn is None:
                        raise ValueError("target_turnover not found in batch for turnover eval")
                    target_turn = target_turn.to(device)
                    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                        pred_turn = outputs[2]
                    else:
                        eps = 1e-8
                        pred_turn = outputs[1] / (outputs[1] + outputs[0] + eps)
                    pred_turn = pred_turn.view(-1, 1)
                    target_turn = target_turn.view(-1, 1)
                    loss_turn = (criterion(pred_turn, target_turn) * weights.view(-1, 1)).mean()
                    total_loss += loss_turn
                elif target_mode == "incorp":
                    loss_incorp = (criterion(outputs[1], target_incorp.unsqueeze(-1)) * weights).mean()
                    total_loss += loss_incorp
                else:
                    loss_loss = (criterion(outputs[0], target_loss.unsqueeze(-1)) * weights).mean()
                    loss_incorp = (criterion(outputs[1], target_incorp.unsqueeze(-1)) * weights).mean()
                    total_loss += loss_loss + loss_incorp
            elif "AASeq":
                features_loss = batch["features_loss"].to(device)
                features_incorporation = batch["features_incorporation"].to(device)
                outputs = model(features_loss, features_incorporation)
                target_loss = batch["target_loss"].to(device)
                target_incorp = batch["target_incorporation"].to(device)
                # Early point weight
                timestep = batch["timestep"]
                timestep_tensor = timestep.float().to(target_loss.device)
                weights = torch.where(
                    timestep_tensor <= 6,
                    torch.tensor(2.0, device=timestep_tensor.device),
                    torch.tensor(1.0, device=timestep_tensor.device),
                )
                if target_mode == "turnover":
                    target_turn = batch.get("target_turnover", None)
                    if target_turn is None:
                        raise ValueError("target_turnover not found in batch for turnover eval")
                    target_turn = target_turn.to(device)
                    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                        pred_turn = outputs[2]
                    else:
                        eps = torch.tensor(1e-8, device=outputs[0].device)
                        pred_turn = outputs[1] / (outputs[1] + outputs[0] + eps)
                    loss_turn = (criterion(pred_turn, target_turn.unsqueeze(-1)) * weights).mean()
                    total_loss += loss_turn
                elif target_mode == "incorp":
                    loss_incorp = (criterion(outputs[1], target_incorp.unsqueeze(-1)) * weights).mean()
                    total_loss += loss_incorp
                else:
                    loss_loss = (criterion(outputs[0], target_loss.unsqueeze(-1)) * weights).mean()
                    loss_incorp = (criterion(outputs[1], target_incorp.unsqueeze(-1)) * weights).mean()
                    total_loss += loss_loss + loss_incorp

    return total_loss / len(loader)


def test_model_TimeSeq(model, test_loader, result_save_dir, device=None, target_mode: str = "both"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    preds_loss, preds_incorporation, preds_turnover = [], [], []
    targets_loss, targets_incorporation, targets_turnover = [], [], []
    preds_loss_K, preds_incorp_K = [], []
    all_clusters = []
    all_peptides = []
    all_timesteps = []

    with torch.no_grad():
        for batch in test_loader:
            features_x = batch["features_x"].to(device)
            outputs = model(features_x)
            target_loss = batch["target_loss"].to(device)
            target_incorporation = batch["target_incorporation"].to(device)
            target_turnover = batch["target_turnover"].to(device)
            peptides = batch["peptide"][0]
            timestep = batch["timestep"]
            cluster_tensor = batch["cluster"]

            for b, peptide in enumerate(peptides):
                timesteps = [int(t[0].item()) for t in timestep]
                cluster_id = int(cluster_tensor[b].item())

                all_timesteps.extend(timesteps)
                all_peptides.extend([peptide] * len(timesteps))
                all_clusters.extend([cluster_id] * len(timesteps))
                if target_mode == "both":
                    preds_loss.extend(outputs[0][b, :, 0].cpu().numpy().tolist())
                    preds_incorporation.extend(outputs[1][b, :, 0].cpu().numpy().tolist())
                elif target_mode == "incorp":
                    preds_incorporation.extend(outputs[1][b, :, 0].cpu().numpy().tolist())

                # Use direct turnover head if present
                if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                    preds_turnover.extend(outputs[2][b, :, 0].cpu().numpy().tolist())
                else:
                    epsilon = 1e-8
                    turnover_values = outputs[1][b, :, 0] / (outputs[1][b, :, 0] + outputs[0][b, :, 0] + epsilon)
                    preds_turnover.extend(turnover_values.cpu().numpy().tolist())
                if target_mode == "both":
                    targets_loss.extend(target_loss[b, :].cpu().numpy().tolist())
                    targets_incorporation.extend(target_incorporation[b, :].cpu().numpy().tolist())
                elif target_mode == "incorp":
                    targets_incorporation.extend(target_incorporation[b, :].cpu().numpy().tolist())
                targets_turnover.extend(target_turnover[b, :].cpu().numpy().tolist())

        # train_df = sample_peptides_for_datasize(train_df, train_size)

    rmse_turn = np.sqrt(mean_squared_error(targets_turnover, preds_turnover))
    r2_turn = r2_score(targets_turnover, preds_turnover)
    if target_mode == "both":
        rmse_loss = np.sqrt(mean_squared_error(targets_loss, preds_loss))
        rmse_incorp = np.sqrt(mean_squared_error(targets_incorporation, preds_incorporation))
        r2_loss = r2_score(targets_loss, preds_loss)
        r2_incorp = r2_score(targets_incorporation, preds_incorporation)
        results_text = (
            f"Test RMSE (Loss): {rmse_loss:.4f}, R2 Score (Loss): {r2_loss:.4f}\n"
            f"Test RMSE (Incorporation): {rmse_incorp:.4f}, R2 Score (Incorporation): {r2_incorp:.4f}\n"
            f"Test RMSE (Turnover): {rmse_turn:.4f}, R2 Score (Turnover): {r2_turn:.4f}"
        )
    elif target_mode == "incorp":
        rmse_incorp = np.sqrt(mean_squared_error(targets_incorporation, preds_incorporation))
        r2_incorp = r2_score(targets_incorporation, preds_incorporation)
        results_text = (
            f"Test RMSE (Incorporation): {rmse_incorp:.4f}, R2 Score (Incorporation): {r2_incorp:.4f}\n"
            f"Test RMSE (Turnover): {rmse_turn:.4f}, R2 Score (Turnover): {r2_turn:.4f}"
        )
    else:
        results_text = (
            f"Test RMSE (Turnover): {rmse_turn:.4f}, R2 Score (Turnover): {r2_turn:.4f}"
        )
    print(results_text)

    print(f"Length of peptides: {len(all_peptides)}")
    print(f"Length of timesteps: {len(all_timesteps)}")
    print(f"Length of targets_loss: {len(targets_loss)}")
    print(f"Length of preds_loss: {len(preds_loss)}")
    print(f"Length of targets_incorporation: {len(targets_incorporation)}")
    print(f"Length of preds_incorporation: {len(preds_incorporation)}")
    print(f"Length of targets_turnover: {len(targets_turnover)}")
    print(f"Length of preds_turnover: {len(preds_turnover)}")

    # DataFrame (columns depend on target_mode)
    if target_mode == "incorp":
        df_dict = {
            "Peptide_ID": all_peptides,
            "Timesteps": all_timesteps,
            "Cluster": all_clusters,
            "True_Values_Incorporation": targets_incorporation,
            "Predictions_Incorporation": preds_incorporation,
        }
    elif target_mode == "both":
        df_dict = {
            "Peptide_ID": all_peptides,
            "Timesteps": all_timesteps,
            "Cluster": all_clusters,
            "True_Values_Loss": targets_loss,
            "Predictions_Loss": preds_loss,
            "True_Values_Incorporation": targets_incorporation,
            "Predictions_Incorporation": preds_incorporation,
            "True_Values_Turnover": targets_turnover,
            "Predictions_Turnover": preds_turnover,
        }
    else:  # turnover
        df_dict = {
            "Peptide_ID": all_peptides,
            "True_Values_Turnover": targets_turnover,
            "Predictions_Turnover": preds_turnover,
            "Timesteps": all_timesteps,
            "Cluster": all_clusters,
        }
    results_df = pd.DataFrame(df_dict)
    print(results_df)
    scores = {
        "RMSE_Turnover": rmse_turn,
        "R2_Turnover": r2_turn,
    }
    if target_mode == "both":
        scores.update(
            {
                "RMSE_Loss": rmse_loss,
                "R2_Loss": r2_loss,
                "RMSE_Incorporation": rmse_incorp,
                "R2_Incorporation": r2_incorp,
            }
        )
    elif target_mode == "incorp":
        scores.update(
            {
                "RMSE_Incorporation": rmse_incorp,
                "R2_Incorporation": r2_incorp,
            }
        )
    print(scores)
    os.makedirs(result_save_dir, exist_ok=True)
    results_save_path = os.path.join(result_save_dir, "test_results.csv")
    results_df.to_csv(results_save_path, index=False)

    # --- クラスタごとの精度を計算 ---
    cluster_metrics = []
    for cluster_id in sorted(results_df["Cluster"].unique()):
        df_cluster = results_df[results_df["Cluster"] == cluster_id]
        n_peptides = df_cluster["Peptide_ID"].nunique()

        entry = {"Cluster": int(cluster_id), "Peptides": int(n_peptides)}

        # Metrics depend on target_mode
        if target_mode == "incorp":
            rmse_incorp = np.sqrt(mean_squared_error(df_cluster["True_Values_Incorporation"], df_cluster["Predictions_Incorporation"]))
            r2_incorp = r2_score(df_cluster["True_Values_Incorporation"], df_cluster["Predictions_Incorporation"]) 
            entry.update({
                "RMSE_Incorporation": round(rmse_incorp, 4),
                "R2_Incorporation": round(r2_incorp, 4),
            })
        else:
            # Turnover always for turnover/both
            rmse_turn = np.sqrt(
                mean_squared_error(df_cluster.get("True_Values_Turnover", []), df_cluster.get("Predictions_Turnover", []))
            )
            r2_turn = r2_score(df_cluster.get("True_Values_Turnover", []), df_cluster.get("Predictions_Turnover", []))
            entry.update({
                "RMSE_Turnover": round(rmse_turn, 4),
                "R2_Turnover": round(r2_turn, 4),
            })

        # Loss/Incorporation if present (both-mode)
        if target_mode == "both" and "True_Values_Loss" in df_cluster.columns:
            rmse_loss = np.sqrt(mean_squared_error(df_cluster["True_Values_Loss"], df_cluster["Predictions_Loss"]))
            r2_loss = r2_score(df_cluster["True_Values_Loss"], df_cluster["Predictions_Loss"]) 
            entry.update({
                "RMSE_Loss": round(rmse_loss, 4),
                "R2_Loss": round(r2_loss, 4),
            })
        if target_mode == "both" and "True_Values_Incorporation" in df_cluster.columns:
            rmse_incorp = np.sqrt(mean_squared_error(df_cluster["True_Values_Incorporation"], df_cluster["Predictions_Incorporation"]))
            r2_incorp = r2_score(df_cluster["True_Values_Incorporation"], df_cluster["Predictions_Incorporation"]) 
            entry.update({
                "RMSE_Incorporation": round(rmse_incorp, 4),
                "R2_Incorporation": round(r2_incorp, 4),
            })

        cluster_metrics.append(entry)

    cluster_metrics_df = pd.DataFrame(cluster_metrics)
    cluster_scores_path = os.path.join(result_save_dir, "cluster_scores.csv")
    cluster_metrics_df.to_csv(cluster_scores_path, index=False)
    print("\n=== Cluster-wise Scores ===")
    print(cluster_metrics_df)
    return results_df, scores


def test_model_AASeq(model, test_loader, results_save_dir, target_mode: str = "both"):
    model.eval()
    preds_loss, preds_incorporation, preds_turnover = [], [], []
    targets_loss, targets_incorporation, targets_turnover = [], [], []
    peptides, timesteps, clusters = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            # print(batch)
            # print("batch_peptide", batch["peptide"])
            features_loss = batch["features_loss"].to(device)
            features_incorporation = batch["features_incorporation"].to(device)
            target_loss = batch["target_loss"].to(device)
            target_incorporation = batch["target_incorporation"].to(device)
            target_turnover = batch["target_turnover"]
            outputs = model(features_loss, features_incorporation)
            output_loss = outputs[0]
            output_incorporation = outputs[1]
            # Robust handling: only treat outputs[2] as turnover if it's a (B,1) head
            if (
                isinstance(outputs, (list, tuple))
                and len(outputs) >= 3
                and torch.is_tensor(outputs[2])
                and outputs[2].dim() == 2
                and outputs[2].size(1) == 1
            ):
                output_turnover = outputs[2]
            else:
                epsilon = torch.tensor(1e-8, device=output_incorporation.device)
                denominator = output_incorporation + output_loss + epsilon
                output_turnover = output_incorporation / denominator
                output_turnover[denominator == epsilon] = 0.0

            # 予測値を記録（1次元配列に揃える）
            if target_mode != "turnover":
                preds_loss.extend(output_loss.view(-1).cpu().numpy().tolist())
                preds_incorporation.extend(output_incorporation.view(-1).cpu().numpy().tolist())
            preds_turnover.extend(output_turnover.view(-1).cpu().numpy().tolist())

            # 実際のターゲット値を記録（1次元配列に揃える）
            if target_mode != "turnover":
                targets_loss.extend(target_loss.view(-1).cpu().numpy().tolist())
                targets_incorporation.extend(target_incorporation.view(-1).cpu().numpy().tolist())
            targets_turnover.extend(target_turnover.view(-1).cpu().numpy().tolist())
            peptide = batch["peptide"]
            timestep = batch["timestep"]
            cluster = batch.get("cluster", None)
            peptides.extend(peptide)
            timesteps.extend(timestep.cpu().numpy().tolist())
            if cluster is not None:
                clusters.extend(cluster.cpu().numpy().astype(int).tolist())

    # 評価指標の計算（全体）
    rmse_turnover = np.sqrt(mean_squared_error(targets_turnover, preds_turnover))
    r2_turnover = r2_score(targets_turnover, preds_turnover)

    # RMSEとR2の結果を文字列としてまとめる
    if target_mode == "both":
        rmse_loss = np.sqrt(mean_squared_error(targets_loss, preds_loss))
        rmse_incorporation = np.sqrt(mean_squared_error(targets_incorporation, preds_incorporation))
        r2_loss = r2_score(targets_loss, preds_loss)
        r2_incorporation = r2_score(targets_incorporation, preds_incorporation)
        results_text = (
            f"Test RMSE (Loss): {rmse_loss:.4f}, R2 Score (Loss): {r2_loss:.4f}\n"
            f"Test RMSE (Incorporation): {rmse_incorporation:.4f}, R2 Score (Incorporation): {r2_incorporation:.4f}\n"
            f"Test RMSE (TurnoverRate): {rmse_turnover:.4f}, R2 Score (TurnoverRate): {r2_turnover:.4f}"
        )
    elif target_mode == "incorp":
        rmse_incorporation = np.sqrt(mean_squared_error(targets_incorporation, preds_incorporation))
        r2_incorporation = r2_score(targets_incorporation, preds_incorporation)
        results_text = (
            f"Test RMSE (Incorporation): {rmse_incorporation:.4f}, R2 Score (Incorporation): {r2_incorporation:.4f}\n"
            f"Test RMSE (TurnoverRate): {rmse_turnover:.4f}, R2 Score (TurnoverRate): {r2_turnover:.4f}"
        )
    else:
        results_text = (
            f"Test RMSE (TurnoverRate): {rmse_turnover:.4f}, R2 Score (TurnoverRate): {r2_turnover:.4f}"
        )

    print(results_text)

    # RMSEとR2の結果を返す
    scores = {
        "RMSE_TurnoverRate": rmse_turnover,
        "R2_TurnoverRate": r2_turnover,
    }
    if target_mode == "both":
        scores.update(
            {
                "RMSE_Loss": rmse_loss,
                "R2_Loss": r2_loss,
                "RMSE_Incorporation": rmse_incorporation,
                "R2_Incorporation": r2_incorporation,
            }
        )
    elif target_mode == "incorp":
        scores.update(
            {
                "RMSE_Incorporation": rmse_incorporation,
                "R2_Incorporation": r2_incorporation,
            }
        )

    # 結果をデータフレームとしてまとめる
    df_dict = {
        "Peptide_ID": peptides,
        "True_Values_Turnover": targets_turnover,
        "Predictions_Turnover": preds_turnover,
        "Timesteps": timesteps,
    }
    if target_mode == "both":
        df_dict.update(
            {
                "True_Values_Loss": targets_loss,
                "Predictions_Loss": preds_loss,
                "True_Values_Incorporation": targets_incorporation,
                "Predictions_Incorporation": preds_incorporation,
            }
        )
    elif target_mode == "incorp":
        df_dict.update(
            {
                "True_Values_Incorporation": targets_incorporation,
                "Predictions_Incorporation": preds_incorporation,
            }
        )
    if len(clusters) == len(peptides):
        df_dict["Cluster"] = clusters
    results_df = pd.DataFrame(df_dict)

    os.makedirs(results_save_dir, exist_ok=True)
    # 互換用のファイルと統一フォーマットの両方を保存
    results_df.to_csv(os.path.join(results_save_dir, "non_recursive.csv"), index=False)
    results_df.to_csv(os.path.join(results_save_dir, "test_results.csv"), index=False)

    # クラスタ別メトリクス
    if "Cluster" in results_df.columns:
        cluster_metrics = []
        for cluster_id in sorted(pd.unique(results_df["Cluster"])):
            df_c = results_df[results_df["Cluster"] == cluster_id]
            entry = {"Cluster": int(cluster_id)}
            rmse_turn = np.sqrt(mean_squared_error(df_c["True_Values_Turnover"], df_c["Predictions_Turnover"]))
            r2_turn = r2_score(df_c["True_Values_Turnover"], df_c["Predictions_Turnover"]) 
            entry.update({"RMSE_Turnover": round(rmse_turn, 4), "R2_Turnover": round(r2_turn, 4)})
            if target_mode != "turnover" and "True_Values_Loss" in df_c.columns:
                rmse_loss = np.sqrt(mean_squared_error(df_c["True_Values_Loss"], df_c["Predictions_Loss"]))
                r2_loss = r2_score(df_c["True_Values_Loss"], df_c["Predictions_Loss"]) 
                rmse_incorp = np.sqrt(mean_squared_error(df_c["True_Values_Incorporation"], df_c["Predictions_Incorporation"]))
                r2_incorp = r2_score(df_c["True_Values_Incorporation"], df_c["Predictions_Incorporation"]) 
                entry.update({
                    "RMSE_Loss": round(rmse_loss, 4),
                    "R2_Loss": round(r2_loss, 4),
                    "RMSE_Incorporation": round(rmse_incorp, 4),
                    "R2_Incorporation": round(r2_incorp, 4),
                })
            cluster_metrics.append(entry)
        pd.DataFrame(cluster_metrics).to_csv(os.path.join(results_save_dir, "cluster_scores.csv"), index=False)

    return results_df, scores


def recursive_test(model, test_df: pd.DataFrame, results_save_dir, debug=False):
    model.eval()

    with torch.no_grad():
        unique_peptides = test_df["Cleaned_Peptidoform"].unique().tolist()
        timesteps = test_df["timestep"].unique().tolist()

        all_peptides, all_timesteps, all_targets_loss, all_preds_loss = [], [], [], []
        all_targets_incorporation, all_preds_incorporation = [], []
        all_targets_turnover, all_preds_turnover = [], []

        for unique_peptide in tqdm(unique_peptides, total=len(unique_peptides)):
            peptide_df = test_df[test_df["Cleaned_Peptidoform"] == unique_peptide]
            peptide_df = peptide_df.sort_values("timestep")
            # lag features
            # seed with dataset-provided lags for the first timestep if available
            def extract_lags(row, prefix):
                cols = [f"{prefix}_t-{i}" for i in range(1, 8)]
                vals = []
                for c in cols:
                    if c in row:
                        vals.append(row[c])
                if len(vals) == 7:
                    return vals
                # fallback
                return ([1] * 7) if prefix == "Label loss" else ([0] * 7)

            first_row = peptide_df.iloc[0]
            lag_features_loss = extract_lags(first_row, "Label loss")
            lag_features_incorporation = extract_lags(first_row, "Label incorporation")
            if len(peptide_df) != 8:
                continue

            for timestep in timesteps:
                peptido_embedding_path = peptide_df["Peptido_Embedding_Path"].values[0]
                embedding_data = np.load(peptido_embedding_path)
                peptido_embedding = embedding_data["peptido_embedding"]
                target_loss = peptide_df[peptide_df["timestep"] == timestep]["Target_LabelLoss"].values[0]
                target_incorporation = peptide_df[peptide_df["timestep"] == timestep]["Target_LabelIncorporation"].values[0]
                target_turnover = peptide_df[peptide_df["timestep"] == timestep]["Target_TurnoverRate"].values[0]

                # Prepare Input
                peptido_embedding = pad_peptide_embedding(peptido_embedding)
                repeated_lag_loss = np.tile(lag_features_loss, (peptido_embedding.shape[0], 1))
                repeated_lag_incorporation = np.tile(lag_features_incorporation, (peptido_embedding.shape[0], 1))
                combined_features_loss = np.concatenate((peptido_embedding, repeated_lag_loss), axis=1)
                combined_features_incorporation = np.concatenate((peptido_embedding, repeated_lag_incorporation), axis=1)
                combined_features_loss_tensor = (
                    torch.tensor(combined_features_loss, dtype=torch.float32).unsqueeze(0).to(device)
                )
                combined_features_incorporation_tensor = (
                    torch.tensor(combined_features_incorporation, dtype=torch.float32).unsqueeze(0).to(device)
                )

                # Predict
                outputs = model(
                    combined_features_loss_tensor,
                    combined_features_incorporation_tensor,
                )
                output_loss = outputs[0]
                output_incorporation = outputs[1]

                # TurnoverRate
                epsilon = 1e-8
                output_turnover = output_incorporation / (output_incorporation + output_loss + epsilon)

                # results
                all_peptides.append(unique_peptide)
                all_timesteps.append(timestep)
                # 予測結果をリストに格納
                all_preds_loss.append(output_loss.cpu().numpy()[0][0])
                all_preds_incorporation.append(output_incorporation.cpu().numpy()[0][0])
                all_preds_turnover.append(output_turnover.cpu().numpy()[0][0])
                # 実際の値を格納
                all_targets_loss.append(target_loss)
                all_targets_incorporation.append(target_incorporation)
                all_targets_turnover.append(target_turnover)

                # ラグ特徴量を更新
                lag_features_loss = [output_loss.cpu().numpy()[0][0]] + lag_features_loss[:-1]
                lag_features_incorporation = [output_incorporation.cpu().numpy()[0][0]] + lag_features_incorporation[:-1]
    # 評価指標の計算（全体）
    rmse_loss = np.sqrt(mean_squared_error(all_targets_loss, all_preds_loss))
    rmse_incorporation = np.sqrt(mean_squared_error(all_targets_incorporation, all_preds_incorporation))
    rmse_turnover = np.sqrt(mean_squared_error(all_targets_turnover, all_preds_turnover))
    r2_loss = r2_score(all_targets_loss, all_preds_loss)
    r2_incorporation = r2_score(all_targets_incorporation, all_preds_incorporation)
    r2_turnover = r2_score(all_targets_turnover, all_preds_turnover)
    results_text = (
        f"Recursive Test RMSE (Loss): {rmse_loss:.4f}, R2 Score (Loss): {r2_loss:.4f}\n"
        f"Recursive Test RMSE (Incorporation): {rmse_incorporation:.4f}, R2 Score (Incorporation): {r2_incorporation:.4f}\n"
        f"Recursive Test RMSE (TurnoverRate): {rmse_turnover:.4f}, R2 Score (TurnoverRate): {r2_turnover:.4f}"
    )
    print(results_text)
    # 結果を返す
    scores = {
        "RMSE_Loss": rmse_loss,
        "R2_Loss": r2_loss,
        "RMSE_Incorporation": rmse_incorporation,
        "R2_Incorporation": r2_incorporation,
        "RMSE_TurnoverRate": rmse_turnover,
        "R2_TurnoverRate": r2_turnover,
    }
    # 結果をデータフレームとしてまとめる
    results_df = pd.DataFrame(
        {
            "Peptide_ID": all_peptides,
            "True_Values_Loss": all_targets_loss,
            "Predictions_Loss": all_preds_loss,
            "True_Values_Incorporation": all_targets_incorporation,
            "Predictions_Incorporation": all_preds_incorporation,
            "True_Values_Turnover": all_targets_turnover,
            "Predictions_Turnover": all_preds_turnover,
            "Timesteps": all_timesteps,
        }
    )
    os.makedirs(results_save_dir, exist_ok=True)
    results_save_path = os.path.join(results_save_dir, "recursive.csv")
    results_df.to_csv(results_save_path)
    return results_df, scores


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, threshold=None):
        """
        Args:
            patience (int): 損失が改善しなくなったときに許容するエポック数
            min_delta (float): 損失が改善とみなされる最小の差
            threshold (float): 早期停止させるための閾値（例: 0.006 以下になったら停止）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.threshold = threshold  # 閾値の設定

    def __call__(self, val_loss):
        # 閾値が設定されている場合、それを「下回ったら」早期停止
        if self.threshold is not None and val_loss <= self.threshold:
            self.early_stop = True
            print(f"Val loss is below the threshold of {self.threshold}. Early stopping.")
            return

        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def sample_peptides_for_datasize(df, datasize, timesteps=8):
    """
    ペプチドごとに全タイムポイントが含まれるように、指定された datasize に近づけるためにペプチドをサンプリングする。
    Args:
        df: データフレーム
        datasize: サンプリングする全データ数（例: 10000）
        timesteps: 各ペプチドが持つタイムポイント数（デフォルトは8）
    """
    peptides_needed = datasize // timesteps  # 必要なペプチド数を計算
    unique_peptides = df["Cleaned_Peptidoform"].unique()  # 全てのユニークなペプチドを取得

    # 必要なペプチド数分をランダムにサンプリング
    if len(unique_peptides) > peptides_needed:
        sampled_peptides = pd.Series(unique_peptides).sample(n=peptides_needed, random_state=42).tolist()
    else:
        sampled_peptides = unique_peptides  # もしペプチドが足りなければ全て使用

    # サンプリングしたペプチドに対応する全タイムポイントのデータを抽出
    sampled_df = df[df["Cleaned_Peptidoform"].isin(sampled_peptides)]

    return sampled_df


def run_model(
    data_loader,
    model,
    model_type,
    train_path,
    val_path,
    test_path,
    input_cls=False,
    input_dim=1280,
    embed_dim=1280,
    n_heads=4,
    num_layers=4,
    dropout=0.3,
    activation_func="Sigmoid",
    lr=0.001,
    num_epochs=20,
    batch_size=8,
    datasize=10000,
    criterion="MSE",
    optimizer="Adam",
    early_stop_patience=5,
    early_stop_threshold=0.02,
    scheduler_patience=3,
    scheduler_threshold=0.02,
    model_save_path=None,
    result_save_dir=None,
    plt_save_dir=None,
    target_mode: str = "both",
    direct_turnover: bool = False,
    clusters_filter=None,
):
    if datasize != "all":
        datasize = int(datasize)
        train_size = int(datasize * 0.7)
        val_size = int(datasize * 0.2)
        test_size = int(datasize * 0.1)

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        train_df = sample_peptides_for_datasize(df=train_df, datasize=train_size)
        val_df = sample_peptides_for_datasize(df=val_df, datasize=val_size)
        test_df = sample_peptides_for_datasize(df=test_df, datasize=test_size)

    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

    # Optional cluster filtering on all splits
    if clusters_filter is not None and len(clusters_filter) > 0:
        train_df = train_df[train_df["Cluster"].isin(clusters_filter)].reset_index(drop=True)
        val_df = val_df[val_df["Cluster"].isin(clusters_filter)].reset_index(drop=True)
        test_df = test_df[test_df["Cluster"].isin(clusters_filter)].reset_index(drop=True)

    # oversample clusters

    # train_df = oversample_clusters(train_df, clusters_to_augment=[4], factor=3)

    train_dataset = data_loader(train_df, input_cls)
    val_dataset = data_loader(val_df, input_cls)
    test_dataset = data_loader(test_df, input_cls)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    if "Transformer" in model_type:
        model = model(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            activation_func=activation_func,
            predict_turnover=direct_turnover,
        ).to(device)
    elif "LSTM" in model_type or "GRU" in model_type:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("モデル初期化を開始...")
        model = model(
            input_dim=input_dim,
            hidden_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_func=activation_func,
            predict_turnover=direct_turnover,
        )
        print("モデル初期化完了:")
        print(model)  # モデル構造を出力
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters())}")
        print(f"モデルデバイス転送前: {next(model.parameters()).device}")

        model = model.to(device)  # デバイス転送
        print(f"モデルデバイス転送後: {next(model.parameters()).device}")
    else:
        raise ValueError("Unsupported model type")

    #
    if criterion == "RMSE":
        criterion = RMSELoss()
    elif criterion == "MSE":
        criterion = torch.nn.MSELoss()
    elif criterion == "SmoothL1Loss":
        criterion = torch.nn.SmoothL1Loss()
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
        verbose=True,
        threshold=scheduler_threshold,
    )
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=num_epochs,  # 総エポック数と一致させる
    #     eta_min=1e-6,  # 最低学習率（学習率が0に落ちないように）
    # )

    # train model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        model_save_path,
        early_stop_patience,
        early_stop_threshold,
        scheduler,
        model_type,
        target_mode,
    )

    # test model
    if "TimeSeq" in model_type:
        results_df, scores = test_model_TimeSeq(model, test_loader, result_save_dir, target_mode=target_mode)
        return scores
    elif "AASeq" in model_type:
        results_df, no_recursive_scores = test_model_AASeq(model, test_loader, result_save_dir, target_mode=target_mode)
        recursive_results_df, recursive_scores = recursive_test(model, test_df, result_save_dir)
        return no_recursive_scores, recursive_scores


def oversample_clusters(df, clusters_to_augment=[1, 5], factor=3):
    augmented = pd.concat(
        [df[df["Cluster"] == c] for c in clusters_to_augment] * (factor - 1),
        ignore_index=True,
    )
    return pd.concat([df, augmented], ignore_index=True)


def main():
    for i in range(0, 10):
        timestamp = time.strftime("%Y_%m%d_%H%M%S")

        # デフォルトの共通パラメータ
        default_model_type = "TimeSeq_LSTM"
        default_dataset_dir = f"data/dataset/sampling/fold_{i}"

        # argparseでの初期パース（model_typeのみ先に抽出）
        parser = argparse.ArgumentParser(description="Train peptide turnover prediction model")
        parser.add_argument("--model_type", type=str, default=default_model_type)
        args, _ = parser.parse_known_args()
        model_type = args.model_type

        # モデルタイプに応じたinput_dimを決定
        if "AASeq_Transformer_based" in model_type:
            input_dim = 1287

        # 各種保存ディレクトリ
        model_save_dir = "data/models"
        result_df_save_dir = "data/results"
        plt_save_dir = "data/plts"
        params_scores_save_dir = "data/params_scores"

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(result_df_save_dir, exist_ok=True)
        os.makedirs(plt_save_dir, exist_ok=True)
        os.makedirs(params_scores_save_dir, exist_ok=True)

        # パラメータの初期設定
        params = {
            # "model_type": model_type,
            "train_path": f"{default_dataset_dir}/train_fold.csv",
            "val_path": f"{default_dataset_dir}/validation_fold.csv",
            "test_path": f"{default_dataset_dir}/test_fold.csv",
            "input_cls": False,
            "datasize": "all",
            "batch_size": 8,
            "input_dim": input_dim,
            "embed_dim": 512,
            "n_heads": 4,
            "num_layers": 2,
            "dropout": 0.2,
            "activation_func": "ReLU",
            "lr": 0.001,
            "num_epochs": 50,
            "criterion": "MSE",
            "optimizer": "Adam",
            "early_stop_patience": 6,
            "early_stop_threshold": 0.01,
            "scheduler_patience": 3,
            "scheduler_threshold": 0.02,
            "model_save_path": f"{model_save_dir}/model_{timestamp}.pth",
            "result_save_dir": f"{result_df_save_dir}/03_AASeq_Transformer_based_sampling/fold_{i}",
            "plt_save_dir": f"{plt_save_dir}/{timestamp}",
        }

        # argparseに全パラメータを反映
        for key, value in params.items():
            parser.add_argument(f"--{key}", default=value)

        args = parser.parse_args()
        params = vars(args)
        model_type = params["model_type"]  # 上書きしておく

        # モデルとデータローダの選択
        if model_type == "TimeSeq_Transformer_based":
            data_loader = Datasets_TimeSeq
            model = Transformer_TimeSeq
        elif model_type == "TimeSeq_LSTM":
            data_loader = Datasets_TimeSeq
            model = LSTM
        elif model_type == "TimeSeq_LSTM_AAC":
            data_loader = Datasets_TimeSeq_AAC
            model = LSTM
        elif model_type == "TimeSeq_GRU":
            data_loader = Datasets_TimeSeq
            model = GRU
        elif model_type == "AASeq_Transformer_based":
            data_loader = Datasets_AASeq
            model = Transformer_AASeq
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 実行
        params["model_type"] = model_type
        if "TimeSeq" in model_type:
            scores = run_model(data_loader=data_loader, model=model, **params)
            output = {
                "parameters": params,
                "test_scores": scores,
            }
        elif "AASeq" in model_type:
            scores, recursive_scores = run_model(data_loader=data_loader, model=model, **params)
            output = {
                "parameters": params,
                "no_recursive_scores": scores,
                "recursive_scores": recursive_scores,
            }

        # 結果保存
        with open(
            f"{result_df_save_dir}/03_AASeq_Transformer_based_sampling/fold_{i}/{timestamp}.json",
            "w",
        ) as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
