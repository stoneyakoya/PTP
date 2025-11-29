import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from datasets.custom_dataset_loader import Datasets_TimeSeq, Datasets_TimeSeq_AAC, Datasets_AASeq, Datasets_AASeq_AAC




def baseline_model(fold, input_type: str, results: list):
    DEBUG = True  # set False to disable debug prints
    # read data
    train_df = pd.read_csv(f"data/dataset/sampling/fold_{fold}/train_fold.csv")
    val_df = pd.read_csv(f"data/dataset/sampling/fold_{fold}/validation_fold.csv")
    test_df = pd.read_csv(f"data/dataset/sampling/fold_{fold}/test_fold.csv")

    if input_type == "AAC":
        train_ds = Datasets_TimeSeq_AAC(train_df, input_cls=False)
        val_ds = Datasets_TimeSeq_AAC(val_df, input_cls=False)
        test_ds = Datasets_TimeSeq_AAC(test_df, input_cls=False)
        # train_ds = Datasets_AASeq_AAC(train_df, input_cls=False)
        # val_ds = Datasets_AASeq_AAC(val_df, input_cls=False)
        # test_ds = Datasets_AASeq_AAC(test_df, input_cls=False)
    elif input_type == "ESM2":
        train_ds = Datasets_TimeSeq(train_df, input_cls=False)
        val_ds = Datasets_TimeSeq(val_df, input_cls=False)
        test_ds = Datasets_TimeSeq(test_df, input_cls=False)
        # train_ds = Datasets_AASeq(train_df, input_cls=False)
        # val_ds = Datasets_AASeq(val_df, input_cls=False)
        # test_ds = Datasets_AASeq(test_df, input_cls=False)

    # print(train_ds[0])
    # print(val_ds[0])
    # print(test_ds[0])
    # input()

    # --- common ---
    batch_size = 8

    print(f"\n=== Fold {fold} ===")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- Prepare NumPy arrays for RF / XGB ---
    def ds_to_xy_multioutput(ds):
        Xs, Ys = [], []
        for item in ds:
            # Prefer features_x; fallback to features_loss
            if "features_x" in item:
                fx = item["features_x"]
                if isinstance(fx, torch.Tensor):
                    x = fx[0].numpy() if fx.dim() == 2 else fx.numpy()
                else:
                    x = np.asarray(fx)
            elif "features_loss" in item:
                fx = item["features_loss"]
                if isinstance(fx, torch.Tensor):
                    x = fx.mean(dim=0).numpy() if fx.dim() == 2 else fx.numpy()
                else:
                    x = np.asarray(fx)
            else:
                raise KeyError("Item has neither 'features_x' nor 'features_loss'")

            if DEBUG and len(Xs) < 3:
                print("[DEBUG multioutput] sample=", len(Xs))
                print("  x.shape:", x.shape, "dtype:", getattr(x, "dtype", type(x)))
                try:
                    print("  x[:10]:", x[:10])
                except Exception:
                    pass

            y_t = item["target_loss"]
            y = y_t.numpy() if isinstance(y_t, torch.Tensor) else np.asarray(y_t)
            y = np.atleast_1d(y)

            if DEBUG and len(Xs) < 3:
                # Show peptide id (first one if list), timesteps, and target preview
                try:
                    pep = item.get("peptide", None)
                    if pep is not None:
                        pep_id = pep[0] if isinstance(pep, (list, tuple)) and len(pep) > 0 else pep
                        print("  peptide:", pep_id)
                except Exception:
                    pass
                try:
                    ts = item.get("timestep", None)
                    if ts is not None:
                        print("  timesteps:", ts)
                except Exception:
                    pass
                print("  y.shape:", y.shape)
                try:
                    print("  y[:3]:", y[:3])
                except Exception:
                    pass
                if y.shape[-1] != 8:
                    print("  [WARN] target length != 8:", y.shape)

            Xs.append(x)
            Ys.append(y)
        return np.vstack(Xs), np.vstack(Ys)

    # TimeSeq-specific: per-timestep samples with time one-hot appended
    def ds_to_xy_timeseq(ds):
        X_rows, y_rows = [], []
        for item in ds:
            fx = item["features_x"] if "features_x" in item else item["features_loss"]
            # fx: (8, D)
            fx_np = fx.numpy() if isinstance(fx, torch.Tensor) else np.asarray(fx)
            # y: (8,)
            y_t = item["target_loss"]
            y_np = y_t.numpy() if isinstance(y_t, torch.Tensor) else np.asarray(y_t)
            # time index as a single scalar feature (0..7)
            for t in range(8):
                time_scalar = np.array([float(t)], dtype=np.float32)
                combined = np.concatenate([fx_np[t], time_scalar], axis=0)
                X_rows.append(combined)
                y_rows.append(y_np[t])

                if DEBUG and len(X_rows) <= 5:
                    print(f"[DEBUG timeseq] row={len(X_rows)-1} t={t}")
                    print("  fx[t].shape:", fx_np[t].shape, " + time_scalar:(1,) -> combined:", combined.shape)
                    try:
                        print("  combined[:12]:", combined[:12])
                    except Exception:
                        pass
        return np.vstack(X_rows), np.asarray(y_rows)

    # Always use multi-output regression (8 outputs) so that one sample -> 8 timepoints
    # For TimeSeq datasets, per-peptide feature is taken from the first timestep row (AAC/ESM2 are time-invariant here)
    X_train, y_train = ds_to_xy_multioutput(train_ds)
    X_val, y_val = ds_to_xy_multioutput(val_ds)
    X_test, y_test = ds_to_xy_multioutput(test_ds)

    xgb = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42, n_jobs=-1
        )
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"  [XGB-MultiOut] Test  RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")
    results.append((f"{fold}", "XGB-MultiOut", rmse_xgb, r2_xgb))

    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"  [RF-MultiOut]  Test  RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")
    results.append((f"{fold}", "RF-MultiOut", rmse_rf, r2_rf))


def main():
    input_type = "AAC"  # or "ESM2"
    results = []
    for fold in range(10):
        baseline_model(fold, input_type, results)

    # Aggregate across all folds
    results_df = pd.DataFrame(results, columns=["fold", "Model", "RMSE", "R2"])

    # Ensure output directory based on input_type
    out_dir = os.path.join("data", "results", f"00_baseline_{input_type}")
    os.makedirs(out_dir, exist_ok=True)

    # Save per-fold results
    results_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    # Compute mean and std per model across folds
    avg_results = results_df.groupby("Model")[
        ["RMSE", "R2"]
    ].mean().reset_index()
    std_results = results_df.groupby("Model")[
        ["RMSE", "R2"]
    ].std().reset_index()

    # Print
    print("\n=== Overall (mean across folds) ===")
    print(avg_results)
    print("\n=== Overall (std across folds) ===")
    print(std_results)

    # Save aggregates
    avg_results.to_csv(os.path.join(out_dir, "avg_results.csv"), index=False)
    std_results.to_csv(os.path.join(out_dir, "std_results.csv"), index=False)


if __name__ == "__main__":
    main()
