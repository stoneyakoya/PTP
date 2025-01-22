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
from datasets.custom_dataset_loader import (
    Datasets_AASeq,
    Datasets_TimeSeq,
    pad_peptide_embedding,
)

# import model
from models.GRU import GRU
from models.LSTM import LSTM
from models.Transformer_based import Transformer_AASeq, Transformer_TimeSeq

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
):
    model.train()
    early_stopping = EarlyStopping(patience=patience, threshold=threshold)
    print("Training started...")
    print(device)

    for epoch in range(num_epochs):
        train_loss = 0.0

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
                loss_loss = criterion(outputs[0], target_loss.unsqueeze(-1))
                loss_incorporation = criterion(
                    outputs[1], target_incorporation.unsqueeze(-1)
                )
                # calculation loss
                total_loss = loss_loss + loss_incorporation
            elif "AASeq":
                features_loss = batch["features_loss"].to(device)
                features_incorporation = batch["features_incorporation"].to(device)
                outputs = model(features_loss, features_incorporation)
                target_loss = batch["target_loss"].to(device)
                target_incorporation = batch["target_incorporation"].to(device)
                loss_loss = criterion(outputs[0], target_loss.unsqueeze(-1))
                loss_incorporation = criterion(
                    outputs[1], target_incorporation.unsqueeze(-1)
                )
                # calculation loss
                total_loss = loss_loss + loss_incorporation

            # debug
            if i == 0 and epoch == 0:
                print("[DEBUG] peptide =", peptide)
                print("[DEBUG] timestep =", timestep)
                print("-" * 10)
                if "TimeSeq" in model_type:
                    print("[DEBUG] features_x.shape =", features_x.shape)
                elif "AASeq" in model_type:
                    print("[DEBUG] features_loss.shape =", features_loss.shape)
                    print("[DEBUG] features_loss =", features_loss)
                print("-" * 10)
                print("[DEBUG] target_loss.shape =", target_loss.shape)
                print("[DEBUG] target_loss =", target_loss)
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # train_loss
        train_loss /= len(train_loader)
        # evaluate_loss
        val_loss = evaluate_model(model, val_loader, criterion, model_type)

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
    torch.save(model.state_dict(), model_save_path)
    print(f"モデルを保存しました: {model_save_path}")


def evaluate_model(model, loader, criterion, model_type):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if "TimeSeq" in model_type:
                features_x = batch["features_x"].to(device)
                outputs = model(features_x)
                target_loss = batch["target_loss"].to(device)
                target_incorporation = batch["target_incorporation"].to(device)
                loss_loss = criterion(outputs[0], target_loss.unsqueeze(-1))
                loss_incorporation = criterion(
                    outputs[1], target_incorporation.unsqueeze(-1)
                )
                total_loss += loss_loss + loss_incorporation.item()

            elif "AASeq":
                features_loss = batch["features_loss"].to(device)
                features_incorporation = batch["features_incorporation"].to(device)
                outputs = model(features_loss, features_incorporation)
                target_loss = batch["target_loss"].to(device)
                target_incorporation = batch["target_incorporation"].to(device)
                loss_loss = criterion(outputs[0], target_loss.unsqueeze(-1))
                loss_incorporation = criterion(
                    outputs[1], target_incorporation.unsqueeze(-1)
                )
                total_loss += loss_loss + loss_incorporation.item()

    return total_loss / len(loader)


def test_model_TimeSeq(model, test_loader, result_save_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    preds_loss, preds_incorporation, preds_turnover = [], [], []
    targets_loss, targets_incorporation, targets_turnover = [], [], []
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
            for b, peptide in enumerate(peptides):
                timesteps = [int(t[0].item()) for t in timestep]
                all_timesteps.extend(timesteps)
                all_peptides.extend([peptide] * len(timesteps))
                preds_loss.extend(outputs[0][b, :, 0].cpu().numpy().tolist())
                preds_incorporation.extend(outputs[1][b, :, 0].cpu().numpy().tolist())
                epsilon = 1e-8
                turnover_values = outputs[1][b, :, 0] / (
                    outputs[1][b, :, 0] + outputs[0][b, :, 0] + epsilon
                )
                preds_turnover.extend(turnover_values.cpu().numpy().tolist())
                targets_loss.extend(target_loss[b, :].cpu().numpy().tolist())
                targets_incorporation.extend(
                    target_incorporation[b, :].cpu().numpy().tolist()
                )
                targets_turnover.extend(target_turnover[b, :].cpu().numpy().tolist())
    rmse_loss = np.sqrt(mean_squared_error(targets_loss, preds_loss))
    rmse_incorp = np.sqrt(
        mean_squared_error(targets_incorporation, preds_incorporation)
    )
    rmse_turn = np.sqrt(mean_squared_error(targets_turnover, preds_turnover))
    r2_loss = r2_score(targets_loss, preds_loss)
    r2_incorp = r2_score(targets_incorporation, preds_incorporation)
    r2_turn = r2_score(targets_turnover, preds_turnover)
    results_text = (
        f"Test RMSE (Loss): {rmse_loss:.4f}, R2 Score (Loss): {r2_loss:.4f}\n"
        f"Test RMSE (Incorporation): {rmse_incorp:.4f}, R2 Score (Incorporation): {r2_incorp:.4f}\n"
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

    # DataFrame
    results_df = pd.DataFrame(
        {
            "Peptide_ID": all_peptides,
            "True_Values_Loss": targets_loss,
            "Predictions_Loss": preds_loss,
            "True_Values_Incorporation": targets_incorporation,
            "Predictions_Incorporation": preds_incorporation,
            "True_Values_Turnover": targets_turnover,
            "Predictions_Turnover": preds_turnover,
            "Timesteps": all_timesteps,
        }
    )
    print(results_df)
    scores = {
        "RMSE_Loss": rmse_loss,
        "R2_Loss": r2_loss,
        "RMSE_Incorporation": rmse_incorp,
        "R2_Incorporation": r2_incorp,
        "RMSE_Turnover": rmse_turn,
        "R2_Turnover": r2_turn,
    }
    print(scores)
    os.makedirs(result_save_dir, exist_ok=True)
    results_save_path = os.path.join(result_save_dir, "test_results.csv")
    results_df.to_csv(results_save_path, index=False)

    return results_df, scores


def test_model_AASeq(model, test_loader, results_save_dir):
    model.eval()
    preds_loss, preds_incorporation, preds_turnover = [], [], []
    targets_loss, targets_incorporation, targets_turnover = [], [], []
    peptides, timesteps = [], []

    with torch.no_grad():
        for batch in test_loader:
            print(batch)
            print("batch_peptide", batch["peptide"])
            features_loss = batch["features_loss"].to(device)
            features_incorporation = batch["features_incorporation"].to(device)
            target_loss = batch["target_loss"].to(device)
            target_incorporation = batch["target_incorporation"].to(device)
            target_turnover = batch["target_turnover"]
            outputs = model(features_loss, features_incorporation)
            output_loss = outputs[0]
            output_incorporation = outputs[1]
            epsilon = torch.tensor(1e-8, device=output_incorporation.device)
            denominator = output_incorporation + output_loss + epsilon
            output_turnover = output_incorporation / denominator
            output_turnover[denominator == epsilon] = 0.0

            # 予測値を記録
            preds_loss.extend(output_loss.cpu().numpy().tolist())
            preds_incorporation.extend(output_incorporation.cpu().numpy().tolist())
            preds_turnover.extend(output_turnover.cpu().numpy().tolist())

            # 実際のターゲット値を記録
            targets_loss.extend(target_loss.cpu().numpy().tolist())
            targets_incorporation.extend(target_incorporation.cpu().numpy().tolist())
            targets_turnover.extend(target_turnover.cpu().numpy().tolist())
            peptide = batch["peptide"]
            timestep = batch["timestep"]
            peptides.extend(peptide)
            timesteps.extend(timestep.cpu().numpy().tolist())

            print("[DEBUG] output_incorporation:", output_incorporation)
            print("[DEBUG] output_loss:", output_loss)
            print("[DEBUG] output_turnover:", output_turnover)

    print("[DEBUG] Lengths of data arrays:")
    print(f"Peptides: {len(peptides)}")
    print(f"Targets_Loss: {len(targets_loss)}")
    print(f"Preds_Loss: {len(preds_loss)}")
    print(f"Targets_Incorporation: {len(targets_incorporation)}")
    print(f"Preds_Incorporation: {len(preds_incorporation)}")
    print(f"Targets_Turnover: {len(targets_turnover)}")
    print(f"Preds_Turnover: {len(preds_turnover)}")
    print(f"Timesteps: {len(timesteps)}")

    # 評価指標の計算
    rmse_loss = np.sqrt(mean_squared_error(targets_loss, preds_loss))
    rmse_incorporation = np.sqrt(
        mean_squared_error(targets_incorporation, preds_incorporation)
    )
    rmse_turnover = np.sqrt(mean_squared_error(targets_turnover, preds_turnover))
    r2_loss = r2_score(targets_loss, preds_loss)
    r2_incorporation = r2_score(targets_incorporation, preds_incorporation)
    r2_turnover = r2_score(targets_turnover, preds_turnover)

    # RMSEとR2の結果を文字列としてまとめる
    results_text = (
        f"Test RMSE (Loss): {rmse_loss:.4f}, R2 Score (Loss): {r2_loss:.4f}\n"
        f"Test RMSE (Incorporation): {rmse_incorporation:.4f}, R2 Score (Incorporation): {r2_incorporation:.4f}\n"
        f"Test RMSE (TurnoverRate): {rmse_turnover:.4f}, R2 Score (TurnoverRate): {r2_turnover:.4f}"
    )

    print(results_text)

    # RMSEとR2の結果を返す
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
            "Peptide_ID": peptides,
            "True_Values_Loss": targets_loss,
            "Predictions_Loss": preds_loss,
            "True_Values_Incorporation": targets_incorporation,
            "Predictions_Incorporation": preds_incorporation,
            "True_Values_Turnover": targets_turnover,
            "Predictions_Turnover": preds_turnover,
            "Timesteps": timesteps,
        }
    )

    os.makedirs(results_save_dir, exist_ok=True)
    results_save_path = os.path.join(results_save_dir, "recursive.csv")
    results_df.to_csv(results_save_path)

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
            # lag features
            lag_features_loss = [1] * 7
            lag_features_incorporation = [0] * 7
            if len(peptide_df) != 8:
                continue

            for timestep in timesteps:
                peptido_embedding_path = peptide_df["Peptido_Embedding_Path"].values[0]
                embedding_data = np.load(peptido_embedding_path)
                peptido_embedding = embedding_data["peptido_embedding"]
                target_loss = peptide_df[peptide_df["timestep"] == timestep][
                    "Target_LabelLoss"
                ].values[0]
                target_incorporation = peptide_df[peptide_df["timestep"] == timestep][
                    "Target_LabelIncorporation"
                ].values[0]
                target_turnover = peptide_df[peptide_df["timestep"] == timestep][
                    "Target_TurnoverRate"
                ].values[0]

                # Prepare Input
                peptido_embedding = pad_peptide_embedding(peptido_embedding)
                repeated_lag_loss = np.tile(
                    lag_features_loss, (peptido_embedding.shape[0], 1)
                )
                repeated_lag_incorporation = np.tile(
                    lag_features_incorporation, (peptido_embedding.shape[0], 1)
                )
                combined_features_loss = np.concatenate(
                    (peptido_embedding, repeated_lag_loss), axis=1
                )
                combined_features_incorporation = np.concatenate(
                    (peptido_embedding, repeated_lag_incorporation), axis=1
                )
                combined_features_loss_tensor = (
                    torch.tensor(combined_features_loss, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                combined_features_incorporation_tensor = (
                    torch.tensor(combined_features_incorporation, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
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
                output_turnover = output_incorporation / (
                    output_incorporation + output_loss + epsilon
                )

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
                lag_features_loss = [
                    output_loss.cpu().numpy()[0][0]
                ] + lag_features_loss[:-1]
                lag_features_incorporation = [
                    output_incorporation.cpu().numpy()[0][0]
                ] + lag_features_incorporation[:-1]
    # 評価指標の計算
    rmse_loss = np.sqrt(mean_squared_error(all_targets_loss, all_preds_loss))
    rmse_incorporation = np.sqrt(
        mean_squared_error(all_targets_incorporation, all_preds_incorporation)
    )
    rmse_turnover = np.sqrt(
        mean_squared_error(all_targets_turnover, all_preds_turnover)
    )
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
            print(
                f"Val loss is below the threshold of {self.threshold}. Early stopping."
            )
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
    unique_peptides = df[
        "Cleaned_Peptidoform"
    ].unique()  # 全てのユニークなペプチドを取得

    # 必要なペプチド数分をランダムにサンプリング
    if len(unique_peptides) > peptides_needed:
        sampled_peptides = (
            pd.Series(unique_peptides)
            .sample(n=peptides_needed, random_state=42)
            .tolist()
        )
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
    embed_dim=256,
    n_heads=4,
    num_layers=4,
    dropout=0.2,
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
):
    if datasize != "all":
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

    train_dataset = data_loader(train_df, input_cls)
    val_dataset = data_loader(val_df, input_cls)
    test_dataset = data_loader(test_df, input_cls)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    if "Transformer" in model_type:
        model = model(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            activation_func=activation_func,
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
    if criterion == "MSE":
        criterion = torch.nn.MSELoss()
    elif criterion == "SmoothL1Loss":
        criterion = torch.nn.SmoothL1Loss()
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.75,
        patience=scheduler_patience,
        verbose=True,
        threshold=scheduler_threshold,
    )

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
    )

    # test model
    if "TimeSeq" in model_type:
        results_df, scores = test_model_TimeSeq(model, test_loader, result_save_dir)
        return scores
    elif "AASeq" in model_type:
        results_df, no_recursive_scores = test_model_AASeq(
            model, test_loader, result_save_dir
        )
        recursive_results_df, recursive_scores = recursive_test(
            model, test_df, result_save_dir
        )
        return no_recursive_scores, recursive_scores


def main():
    model_type = "AASeq_Transformer_based"
    if model_type == "AASeq_Transformer_based":
        input_dim = 1287
    else:
        input_dim = 1280
    timestamp = time.strftime("%Y_%m%d_%H%M%S")
    dataset_dir = "data/dataset/normal"
    # save dir
    model_save_dir = "data/models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    result_df_save_dir = "data/results"
    if not os.path.exists(result_df_save_dir):
        os.makedirs(model_save_dir)

    plt_save_dir = "data/plts"
    if not os.path.exists(plt_save_dir):
        os.makedirs(plt_save_dir)

    params = {
        "model_type": model_type,
        "train_path": f"{dataset_dir}/train.csv",
        "val_path": f"{dataset_dir}/validation.csv",
        "test_path": f"{dataset_dir}/test.csv",
        "input_cls": False,
        "datasize": 10000,
        "batch_size": 8,
        "input_dim": input_dim,
        "embed_dim": 512,
        "n_heads": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "activation_func": "ReLU",
        "lr": 0.001,
        "num_epochs": 1,
        "criterion": "MSE",
        "optimizer": "Adam",
        "early_stop_patience": 15,
        "early_stop_threshold": 0.010,
        "scheduler_patience": 5,
        "scheduler_threshold": 0.02,
        "model_save_path": f"{model_save_dir}/model_{timestamp}.pth",
        "result_save_dir": f"{result_df_save_dir}/{timestamp}",
        "plt_save_dir": f"{plt_save_dir}/{timestamp}",
    }

    parser = argparse.ArgumentParser(
        description="Train peptide turnover prediction model"
    )
    for key, value in params.items():
        parser.add_argument(
            f"--{key}", type=type(value), default=value, help=f"Default: {value}"
        )

    args = parser.parse_args()
    params = vars(args)

    if model_type == "TimeSeq_Transformer_based":
        data_loader = Datasets_TimeSeq
        model = Transformer_TimeSeq
    elif model_type == "TimeSeq_LSTM":
        data_loader = Datasets_TimeSeq
        model = LSTM
    elif model_type == "TimeSeq_GRU":
        data_loader = Datasets_TimeSeq
        model = GRU
    elif model_type == "AASeq_Transformer_based":
        data_loader = Datasets_AASeq
        model = Transformer_AASeq

    if "TimeSeq" in model_type:
        scores = run_model(
            data_loader=data_loader,
            model=model,
            **params,
        )
        output = {
            "parameters": params,
            "test_scores": scores,
        }
    elif "AASeq" in model_type:
        scores, recursive_scores = run_model(
            data_loader=data_loader,
            model=model,
            **params,
        )
        output = {
            "parameters": params,
            "no_recursive_scores": scores,
            "recursive_scores": recursive_scores,
        }

    params_scores_save_dir = "data/params_scores"
    if not os.path.exists(params_scores_save_dir):
        os.makedirs(params_scores_save_dir)
    with open(f"{params_scores_save_dir}/{timestamp}.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
