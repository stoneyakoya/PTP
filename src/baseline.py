import numpy as np
import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from .datasets.custom_dataset_loader import Datasets_TimeSeq, Datasets_TimeSeq_AAC, Datasets_AASeq, Datasets_AASeq_AAC




def baseline_model(
    fold,
    input_type: str,
    results: list,
    target: str = "loss",  # one of {loss, incorporation, turnover}
    variant: str = "full",  # one of {full, late}
    late_hour: int | None = None,  # e.g., 48; if None, use the last index
    debug: bool = False,
):
    DEBUG = bool(debug and fold == 0)  # print debug info only for the first fold
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

    # --- helper: target key ---
    if target != "both":
        target_key = {
            "loss": "target_loss",
            "incorporation": "target_incorporation",
            "turnover": "target_turnover",
        }[target]
    else:
        target_key = None  # handled by dedicated "both" branches

    # --- Prepare NumPy arrays for RF / XGB ---
    def _extract_feature_vector(item):
        # Prefer features_x; fallback to features_loss
        if "features_x" in item:
            fx = item["features_x"]
            if isinstance(fx, torch.Tensor):
                # Take the first timestep feature (AAC/ESM2 are time-invariant per peptide)
                return fx[0].numpy() if fx.dim() == 2 else fx.numpy()
            return np.asarray(fx)
        elif "features_loss" in item:
            fx = item["features_loss"]
            if isinstance(fx, torch.Tensor):
                # Aggregate across sequence/tokens if provided (mean)
                return fx.mean(dim=0).numpy() if fx.dim() == 2 else fx.numpy()
            return np.asarray(fx)
        else:
            raise KeyError("Item has neither 'features_x' nor 'features_loss'")

    def ds_to_xy_multioutput(ds):
        Xs, Ys = [], []
        for item in ds:
            x = _extract_feature_vector(item)

            if DEBUG and len(Xs) < 3:
                print("[DEBUG multioutput] sample=", len(Xs))
                print("  x.shape:", x.shape, "dtype:", getattr(x, "dtype", type(x)))
                try:
                    print("  x[:10]:", x[:10])
                except Exception:
                    pass

            y_t = item[target_key]
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

    def ds_to_xy_multioutput_both(ds):
        Xs, Ys_loss, Ys_incorp, Ys_turn = [], [], [], []
        for item in ds:
            x = _extract_feature_vector(item)
            y_loss_t = item["target_loss"]
            y_inc_t = item["target_incorporation"]
            y_turn_t = item["target_turnover"]

            y_loss = y_loss_t.numpy() if isinstance(y_loss_t, torch.Tensor) else np.asarray(y_loss_t)
            y_inc = y_inc_t.numpy() if isinstance(y_inc_t, torch.Tensor) else np.asarray(y_inc_t)
            y_turn = y_turn_t.numpy() if isinstance(y_turn_t, torch.Tensor) else np.asarray(y_turn_t)

            Xs.append(x)
            Ys_loss.append(np.atleast_1d(y_loss))
            Ys_incorp.append(np.atleast_1d(y_inc))
            Ys_turn.append(np.atleast_1d(y_turn))

        return (
            np.vstack(Xs),
            np.vstack(Ys_loss),
            np.vstack(Ys_incorp),
            np.vstack(Ys_turn),
        )

    def ds_to_xy_late(ds, hour: int | None = None):
        Xs, Ys = [], []
        for item in ds:
            x = _extract_feature_vector(item)
            y_t = item[target_key]
            y = y_t.numpy() if isinstance(y_t, torch.Tensor) else np.asarray(y_t)
            # Select specific hour if provided, else last index
            if hour is not None:
                # item may contain actual timesteps for mapping
                ts = item.get("timestep", None)
                if ts is not None:
                    try:
                        # ts could be list-like of length 8
                        idx = list(ts).index(hour)
                    except Exception:
                        idx = -1
                else:
                    idx = -1
            else:
                idx = -1
            y_scalar = float(y[idx])

            Xs.append(x)
            Ys.append(y_scalar)
        return np.vstack(Xs), np.asarray(Ys)

    def ds_to_xy_late_both(ds, hour: int | None = None):
        Xs, Ys_loss, Ys_incorp, Ys_turn = [], [], [], []
        for item in ds:
            x = _extract_feature_vector(item)
            y_loss_t = item["target_loss"]
            y_inc_t = item["target_incorporation"]
            y_turn_t = item["target_turnover"]

            y_loss = y_loss_t.numpy() if isinstance(y_loss_t, torch.Tensor) else np.asarray(y_loss_t)
            y_inc = y_inc_t.numpy() if isinstance(y_inc_t, torch.Tensor) else np.asarray(y_inc_t)
            y_turn = y_turn_t.numpy() if isinstance(y_turn_t, torch.Tensor) else np.asarray(y_turn_t)

            # Select specific hour if provided, else last index
            if hour is not None:
                ts = item.get("timestep", None)
                if ts is not None:
                    try:
                        idx = list(ts).index(hour)
                    except Exception:
                        idx = -1
                else:
                    idx = -1
            else:
                idx = -1

            Xs.append(x)
            Ys_loss.append(float(y_loss[idx]))
            Ys_incorp.append(float(y_inc[idx]))
            Ys_turn.append(float(y_turn[idx]))

        return (
            np.vstack(Xs),
            np.asarray(Ys_loss),
            np.asarray(Ys_incorp),
            np.asarray(Ys_turn),
        )

    # TimeSeq-specific: per-timestep samples with time one-hot appended
    def ds_to_xy_timeseq(ds):
        X_rows, y_rows = [], []
        for item in ds:
            fx = item["features_x"] if "features_x" in item else item["features_loss"]
            # fx: (8, D)
            fx_np = fx.numpy() if isinstance(fx, torch.Tensor) else np.asarray(fx)
            # y: (8,)
            y_t = item[target_key]
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

    # Build X/Y according to variant
    if target == "both" and variant == "full":
        # Train two multi-output models (loss and incorporation), evaluate on turnover
        X_train, yL_train, yI_train, _ = ds_to_xy_multioutput_both(train_ds)
        X_val, yL_val, yI_val, _ = ds_to_xy_multioutput_both(val_ds)
        X_test, yL_test, yI_test, yT_true = ds_to_xy_multioutput_both(test_ds)

        # XGB: separate multi-output models per target
        xgb_loss = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42, n_jobs=-1
            )
        )
        xgb_incorp = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42, n_jobs=-1
            )
        )
        xgb_loss.fit(X_train, yL_train)
        xgb_incorp.fit(X_train, yI_train)
        yL_pred_xgb = xgb_loss.predict(X_test)
        yI_pred_xgb = xgb_incorp.predict(X_test)
        eps = 1e-8
        yTurn_pred_xgb = yI_pred_xgb / (yI_pred_xgb + yL_pred_xgb + eps)
        rmse_xgb = np.sqrt(mean_squared_error(yT_true, yTurn_pred_xgb))
        r2_xgb = r2_score(yT_true, yTurn_pred_xgb)
        print(f"  [XGB-Both->Turnover] Test  RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")
        results.append((f"{fold}", "XGB-BothTurnover", rmse_xgb, r2_xgb))

        # RF: separate multi-output models per target
        rf_loss = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
        rf_incorp = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
        rf_loss.fit(X_train, yL_train)
        rf_incorp.fit(X_train, yI_train)
        yL_pred_rf = rf_loss.predict(X_test)
        yI_pred_rf = rf_incorp.predict(X_test)
        yTurn_pred_rf = yI_pred_rf / (yI_pred_rf + yL_pred_rf + eps)
        rmse_rf = np.sqrt(mean_squared_error(yT_true, yTurn_pred_rf))
        r2_rf = r2_score(yT_true, yTurn_pred_rf)
        print(f"  [RF-Both->Turnover]  Test  RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")
        results.append((f"{fold}", "RF-BothTurnover", rmse_rf, r2_rf))

    elif target == "both" and variant == "late":
        # Train two single-output models (loss and incorporation) for selected timepoint, evaluate on turnover
        X_train, yL_train, yI_train, _ = ds_to_xy_late_both(train_ds, hour=late_hour)
        X_val, yL_val, yI_val, _ = ds_to_xy_late_both(val_ds, hour=late_hour)
        X_test, yL_test, yI_test, yT_true = ds_to_xy_late_both(test_ds, hour=late_hour)

        xgb_loss = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42, n_jobs=-1
        )
        xgb_incorp = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42, n_jobs=-1
        )
        xgb_loss.fit(X_train, yL_train)
        xgb_incorp.fit(X_train, yI_train)
        yL_pred_xgb = xgb_loss.predict(X_test)
        yI_pred_xgb = xgb_incorp.predict(X_test)
        eps = 1e-8
        yTurn_pred_xgb = yI_pred_xgb / (yI_pred_xgb + yL_pred_xgb + eps)
        rmse_xgb = np.sqrt(mean_squared_error(yT_true, yTurn_pred_xgb))
        r2_xgb = r2_score(yT_true, yTurn_pred_xgb)
        print(f"  [XGB-Both->Turnover-Late] Test  RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")
        results.append((f"{fold}", "XGB-BothTurnover-Late", rmse_xgb, r2_xgb))

        rf_loss = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42, n_jobs=-1)
        rf_incorp = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42, n_jobs=-1)
        rf_loss.fit(X_train, yL_train)
        rf_incorp.fit(X_train, yI_train)
        yL_pred_rf = rf_loss.predict(X_test)
        yI_pred_rf = rf_incorp.predict(X_test)
        yTurn_pred_rf = yI_pred_rf / (yI_pred_rf + yL_pred_rf + eps)
        rmse_rf = np.sqrt(mean_squared_error(yT_true, yTurn_pred_rf))
        r2_rf = r2_score(yT_true, yTurn_pred_rf)
        print(f"  [RF-Both->Turnover-Late]  Test  RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")
        results.append((f"{fold}", "RF-BothTurnover-Late", rmse_rf, r2_rf))

    elif variant == "full":
        # Multi-output regression (8 outputs per sample)
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
    elif variant == "late":
        # Single output: only one chosen timepoint
        X_train, y_train = ds_to_xy_late(train_ds, hour=late_hour)
        X_val, y_val = ds_to_xy_late(val_ds, hour=late_hour)
        X_test, y_test = ds_to_xy_late(test_ds, hour=late_hour)

        xgb = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, objective="reg:squarederror", random_state=42, n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        r2_xgb = r2_score(y_test, y_pred_xgb)
        print(f"  [XGB-Late]     Test  RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")
        results.append((f"{fold}", "XGB-Late", rmse_xgb, r2_xgb))

        rf = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2_rf = r2_score(y_test, y_pred_rf)
        print(f"  [RF-Late]      Test  RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")
        results.append((f"{fold}", "RF-Late", rmse_rf, r2_rf))
    else:
        raise ValueError(f"Unknown variant: {variant}")



def main():
    parser = argparse.ArgumentParser(description="Baseline models for peptide time-series targets")
    parser.add_argument("--input_type", type=str, default="AAC", choices=["AAC", "ESM2"], help="Feature type")
    parser.add_argument(
        "--target",
        type=str,
        default="loss",
        choices=["loss", "incorporation", "turnover", "both"],
        help="Which target to predict (use 'both' to train loss & incorporation and evaluate turnover)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["full", "late"],
        help="Prediction variant: 8-step (full) or single late timepoint",
    )
    parser.add_argument(
        "--late_hour",
        type=int,
        default=None,
        help="Hour to select for late variant (e.g., 48). If None, use the last timepoint",
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of folds to evaluate")
    parser.add_argument("--debug", action="store_true", help="Print debug info (only first fold)")
    args = parser.parse_args()

    input_type = args.input_type
    results = []
    for fold in range(args.folds):
        baseline_model(
            fold,
            input_type,
            results,
            target=args.target,
            variant=args.variant,
            late_hour=args.late_hour,
            debug=args.debug,
        )

    # Aggregate across all folds
    results_df = pd.DataFrame(results, columns=["fold", "Model", "RMSE", "R2"])

    # Ensure output directory based on input_type, target and variant
    suffix = f"{input_type}_{args.target}_{args.variant}"
    if args.variant == "late" and args.late_hour is not None:
        suffix += f"_t{args.late_hour}"
    # Align with repository structure: generated artifacts under reports/
    out_dir = os.path.join("reports", "results", "baseline", suffix)
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

    # Merge mean and std into one summary (mean ± std)
    summary_df = avg_results.merge(
        std_results, on="Model", suffixes=("_mean", "_std")
    )
    # Reorder and round
    cols = [
        "Model",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std",
    ]
    summary_df = summary_df[cols].copy()
    summary_df = summary_df.round({"RMSE_mean": 6, "RMSE_std": 6, "R2_mean": 6, "R2_std": 6})
    summary_df.to_csv(os.path.join(out_dir, "overall_summary.csv"), index=False)

    # Also save a Markdown table for quick viewing
    md_lines = ["| Model | RMSE (mean±std) | R2 (mean±std) |", "|---|---:|---:|"]
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['Model']} | {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f} | {row['R2_mean']:.4f} ± {row['R2_std']:.4f} |"
        )
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
