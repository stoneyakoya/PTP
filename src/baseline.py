import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from datasets.custom_dataset_loader import Datasets_TimeSeq, Datasets_TimeSeq_AAC


# --- PyTorch Linear Model ---
class TimeSeqLinear(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(8 * input_dim, 8)

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
        return self.fc(x)


# --- 共通設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 50
batch_size = 8
lr = 1e-3

# ESM2 embedding の統計量プーリング後の次元
input_dim = 1280 * 4  # mean/std/max/min を concat しているため

# 指標の格納用

results_df = pd.DataFrame(columns=["fold", "Model", "RMSE", "R2"])

for fold in range(0, 10):
    results = []
    print(f"\n=== Fold {fold} ===")

    # 1) データ読み込み
    train_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{fold}/train_fold.csv")
    val_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{fold}/validation_fold.csv")
    test_df = pd.read_csv(f"/home/ishino/PTP/data/dataset/sampling/fold_{fold}/test_fold.csv")

    train_ds = Datasets_TimeSeq_AAC(train_df, input_cls=False)
    val_ds = Datasets_TimeSeq_AAC(val_df, input_cls=False)
    test_ds = Datasets_TimeSeq_AAC(test_df, input_cls=False)

    train_ds = Datasets_TimeSeq(train_df, input_cls=False)
    val_ds = Datasets_TimeSeq(val_df, input_cls=False)
    test_ds = Datasets_TimeSeq(test_df, input_cls=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # # --- A) PyTorch Linear Model ---
    # model = TimeSeqLinear(input_dim).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()

    # # 学習 + 検証
    # for epoch in range(1, n_epochs + 1):
    #     model.train()
    #     train_losses = []

    #     for batch in train_loader:
    #         x = batch["features_x"].to(device)
    #         y = batch["target_turnover"].to(device)

    #         optimizer.zero_grad()
    #         pred = model(x)
    #         loss = criterion(pred, y)
    #         loss.backward()
    #         optimizer.step()
    #         train_losses.append(loss.item())

    #     # 簡易バリデーション（エポックごとに評価したい場合はここで計算）
    #     if epoch == 1 or epoch % 10 == 0:
    #         model.eval()
    #         val_true, val_pred = [], []
    #         with torch.no_grad():
    #             for batch in val_loader:
    #                 x = batch["features_x"].to(device)
    #                 y = batch["target_turnover"].to(device)
    #                 p = model(x)
    #                 val_true.append(y.cpu().numpy())
    #                 val_pred.append(p.cpu().numpy())
    #         val_true = np.vstack(val_true)
    #         val_pred = np.vstack(val_pred)
    #         print(f"  [Linear] Epoch {epoch}  Val RMSE: {np.sqrt(mean_squared_error(val_true, val_pred)):.4f}")

    # # テスト評価
    # model.eval()
    # test_true, test_pred = [], []
    # with torch.no_grad():
    #     for batch in test_loader:
    #         x = batch["features_x"].to(device)
    #         y = batch["target_turnover"].to(device)
    #         p = model(x)
    #         test_true.append(y.cpu().numpy())
    #         test_pred.append(p.cpu().numpy())
    # test_true = np.vstack(test_true)
    # test_pred = np.vstack(test_pred)
    # rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    # r2 = r2_score(test_true, test_pred)
    # print(f"  [Linear] Test  RMSE: {rmse:.4f}, R²: {r2:.4f}")
    # results.append(("Linear", rmse, r2))

    # --- B) Prepare NumPy arrays for RF / XGB ---
    def ds_to_xy(ds):
        Xs, Ys = [], []
        for item in ds:
            # flatten features_x: (8, input_dim) → (8*input_dim,)
            x = item["features_x"][0].numpy()
            # x = item["features_x"].numpy().reshape(-1)
            y = item["target_turnover"].numpy()
            # print(x.shape, y.shape)
            # input()
            Xs.append(x)
            Ys.append(y)
        return np.vstack(Xs), np.vstack(Ys)

    X_train, y_train = ds_to_xy(train_ds)
    X_val, y_val = ds_to_xy(val_ds)
    X_test, y_test = ds_to_xy(test_ds)

    # --- D) XGBoost ---
    xgb = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42, n_jobs=-1
        )
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"  [XGB]    Test  RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")
    results.append((f"{fold}", "XGBoost", rmse_xgb, r2_xgb))

    # --- C) RandomForestRegressor ---
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"  [RF]     Test  RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")
    results.append((f"{fold}", "RandomForest", rmse_rf, r2_rf))
    results_df_fold = pd.DataFrame(results, columns=["fold", "Model", "RMSE", "R2"])
    results_df = pd.concat([results_df, results_df_fold], ignore_index=True)
    results_df.to_csv("data/results/00_baseline_ESM2/results.csv", index=False)


# --- 最終結果まとめ ---
# モデルごとに RMSE と R2 の平均を計算
avg_results = results_df.groupby("Model")[["RMSE", "R2"]].mean().reset_index()

# オプションで標準偏差も出したいときは：
std_results = results_df.groupby("Model")[["RMSE", "R2"]].std().reset_index()

print("=== 各モデルのFold平均 ===")
print(avg_results)

print("\n=== 各モデルのFold標準偏差 ===")
print(std_results)

# CSV に書き出す場合
avg_results.to_csv("data/results/00_baseline_AAC/avg_results.csv", index=False)
