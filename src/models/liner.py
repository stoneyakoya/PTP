import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from datasets.custom_dataset_loader import Datasets_TimeSeq_AAC

# --- モジュール ---


# --- モデル定義 ---
class TimeSeqLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8 * 20, 8)

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)  # (batch, 160)
        return self.fc(x)  # (batch, 8)


# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 50
batch_size = 32
lr = 1e-3

for i in range(0, 1):  # fold の数だけ回す
    print(f"\n=== Fold {i} ===")

    # データ読み込み
    train_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{i}/train_fold.csv")
    val_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{i}/validation_fold.csv")
    test_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{i}/test_fold.csv")

    train_ds = Datasets_TimeSeq_AAC(train_df)
    val_ds = Datasets_TimeSeq_AAC(val_df)
    test_ds = Datasets_TimeSeq_AAC(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # モデル・最適化器・損失
    model = TimeSeqLinear().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 学習ループ
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["features_x"].to(device)
            y = batch["target_turnover"].to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 検証ステップ
        model.eval()
        val_trues = []
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features_x"].to(device)
                y = batch["target_turnover"].to(device)
                y_pred = model(x)

                val_trues.append(y.cpu().numpy())
                val_preds.append(y_pred.cpu().numpy())

        # 結合して評価
        val_trues = np.vstack(val_trues)
        val_preds = np.vstack(val_preds)
        val_mse = mean_squared_error(val_trues, val_preds)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(val_trues, val_preds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}  Train MSE: {np.mean(train_losses):.4f}  Val RMSE: {val_rmse:.4f}  Val R²: {val_r2:.4f}")

    # テスト評価
    model.eval()
    test_trues = []
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features_x"].to(device)
            y = batch["target_turnover"].to(device)
            y_pred = model(x)

            test_trues.append(y.cpu().numpy())
            test_preds.append(y_pred.cpu().numpy())

    test_trues = np.vstack(test_trues)
    test_preds = np.vstack(test_preds)
    test_mse = mean_squared_error(test_trues, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_trues, test_preds)

    print(f"Fold {i}  Test RMSE: {test_rmse:.4f}  Test R²: {test_r2:.4f}")
