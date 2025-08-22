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


def baseline_model(fold, input_type: str, results: list):
    # read data
    train_df = pd.read_csv(f"data/dataset/sampling/fold_{fold}/train_fold.csv")
    val_df = pd.read_csv(f"data/dataset/sampling/fold_{fold}/validation_fold.csv")
    test_df = pd.read_csv(f"data/dataset/sampling/fold_{fold}/test_fold.csv")

    if input_type == "AAC":
        train_ds = Datasets_TimeSeq_AAC(train_df, input_cls=False)
        val_ds = Datasets_TimeSeq_AAC(val_df, input_cls=False)
        test_ds = Datasets_TimeSeq_AAC(test_df, input_cls=False)
    elif input_type == "ESM2":
        train_ds = Datasets_TimeSeq(train_df, input_cls=False)
        val_ds = Datasets_TimeSeq(val_df, input_cls=False)
        test_ds = Datasets_TimeSeq(test_df, input_cls=False)

    # --- common ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 50
    batch_size = 8
    lr = 1e-3

    # ESM2 embedding の統計量プーリング後の次元
    input_dim = 1280 * 4  # mean/std/max/min を concat しているため

    results_df = pd.DataFrame(columns=["fold", "Model", "RMSE", "R2"])

    print(f"\n=== Fold {fold} ===")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- Prepare NumPy arrays for RF / XGB ---
    def ds_to_xy(ds):
        Xs, Ys = [], []
        for item in ds:
            # flatten features_x: (8, input_dim) → (input_dim,)
            x = item["features_x"][0].numpy()
            y = item["target_turnover"].numpy()
            Xs.append(x)
            Ys.append(y)
        return np.vstack(Xs), np.vstack(Ys)

    X_train, y_train = ds_to_xy(train_ds)
    X_val, y_val = ds_to_xy(val_ds)
    X_test, y_test = ds_to_xy(test_ds)
    # --- XGBoost ---
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
    # --- RandomForestRegressor ---
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"  [RF]     Test  RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")
    # --- results ---
    results.append((f"{fold}", "RandomForest", rmse_rf, r2_rf))
    results_df_fold = pd.DataFrame(results, columns=["fold", "Model", "RMSE", "R2"])
    results_df = pd.concat([results_df, results_df_fold], ignore_index=True)
    results_df.to_csv("data/results/00_baseline_ESM2/results.csv", index=False)

    # --- all folds results ---
    # per model analysis
    avg_results = results_df.groupby("Model")[["RMSE", "R2"]].mean().reset_index()
    std_results = results_df.groupby("Model")[["RMSE", "R2"]].std().reset_index()

    print("=== avg ===")
    print(avg_results)

    print("\n=== std ===")
    print(std_results)

    # CSV output
    avg_results.to_csv("data/results/00_baseline_AAC/avg_results.csv", index=False)


def main():
    input_type = "AAC"  # or "ESM2"
    retults = []
    for fold in range(10):
        baseline_model(fold, input_type, retults)


if __name__ == "__main__":
    main()
