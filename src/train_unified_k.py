import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.custom_dataset_loader import Datasets_AASeqTimeSeq, collate_fn
from models.unified import STCrossPredictor
from utils.k_fit import fit_k_loss, fit_k_incorp, TIME_HOURS_DEFAULT


def compute_logk_from_series(y_loss: np.ndarray, y_inc: np.ndarray) -> float:
    t = np.array(TIME_HOURS_DEFAULT, dtype=float)
    kvals = []
    r1 = fit_k_loss(y_loss, t)
    r2 = fit_k_incorp(y_inc, t)
    if r1 is not None and r1.get("success"):
        kvals.append(r1["K"]) 
    if r2 is not None and r2.get("success"):
        kvals.append(r2["K"]) 
    if len(kvals) == 0:
        return float("nan")
    k = float(np.mean(kvals))
    return float(np.log(max(k, np.finfo(float).tiny)))


@torch.no_grad()
def batch_logk_from_series(pred_loss: torch.Tensor, pred_inc: torch.Tensor) -> np.ndarray:
    """
    pred_loss, pred_inc: (B, T) tensors on any device.
    Returns np.ndarray of shape (B,) with logK estimates (may contain NaN).
    """
    loss_np = pred_loss.detach().cpu().numpy()
    inc_np = pred_inc.detach().cpu().numpy()
    out = []
    for i in range(loss_np.shape[0]):
        out.append(compute_logk_from_series(loss_np[i], inc_np[i]))
    return np.array(out, dtype=float)


def train_and_eval(
    train_df_path: str,
    val_df_path: str,
    test_df_path: str,
    device: torch.device,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    lambda_k: float = 0.2,
    patience: int = 5,
):
    import pandas as pd

    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)
    test_df = pd.read_csv(test_df_path)

    train_ds = Datasets_AASeqTimeSeq(train_df)
    val_ds = Datasets_AASeqTimeSeq(val_df)
    test_ds = Datasets_AASeqTimeSeq(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    mae = nn.L1Loss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True, threshold=0.001
    )

    best_val_k_mae = float("inf")
    no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            emb = batch["embedding"].to(device)
            mask = batch["mask"].to(device)
            t_idx = batch["t_idx"].to(device)
            y_loss = batch["y_loss"].to(device)
            y_inc = batch["y_inc"].to(device)

            optimizer.zero_grad()
            loss_pred, inc_pred, logk_pred, _ = model(emb, mask, t_idx)
            l_series = criterion(loss_pred, y_loss) + criterion(inc_pred, y_inc)

            # compute logK_true from ground-truth series per sample (CPU)
            gt_logk_np = batch_logk_from_series(y_loss, y_inc)  # (B,)
            valid = ~np.isnan(gt_logk_np)
            if valid.any():
                gt_logk = torch.from_numpy(gt_logk_np[valid]).to(logk_pred.device)
                l_k = mae(logk_pred[torch.as_tensor(valid, device=logk_pred.device)], gt_logk)
            else:
                l_k = torch.tensor(0.0, device=logk_pred.device)

            loss = l_series + lambda_k * l_k
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * emb.size(0)

        train_loss /= len(train_ds)

        # validation: early stop by Khat MAE from predicted series
        model.eval()
        with torch.no_grad():
            khat_list = []
            ktrue_list = []
            val_series_loss = 0.0
            for batch in val_loader:
                emb = batch["embedding"].to(device)
                mask = batch["mask"].to(device)
                t_idx = batch["t_idx"].to(device)
                y_loss = batch["y_loss"].to(device)
                y_inc = batch["y_inc"].to(device)

                loss_pred, inc_pred, logk_pred, _ = model(emb, mask, t_idx)
                val_series_loss += (criterion(loss_pred, y_loss) + criterion(inc_pred, y_inc)).item() * emb.size(0)

                ktrue_np = batch_logk_from_series(y_loss, y_inc)  # (B,)
                khat_np = batch_logk_from_series(loss_pred, inc_pred)  # (B,)
                m = ~np.isnan(ktrue_np) & ~np.isnan(khat_np)
                if m.any():
                    khat_list.append(khat_np[m])
                    ktrue_list.append(ktrue_np[m])

            val_series_loss /= len(val_ds)
            if khat_list:
                khat = np.concatenate(khat_list)
                ktrue = np.concatenate(ktrue_list)
                val_k_mae = float(np.mean(np.abs(khat - ktrue)))
            else:
                val_k_mae = float("inf")

        scheduler.step(val_k_mae)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: lr={lr_now:.2e}, train_loss={train_loss:.4f}, val_series={val_series_loss:.4f}, val_Kmae={val_k_mae:.4f}")

        if val_k_mae < best_val_k_mae:
            best_val_k_mae = val_k_mae
            no_improve = 0
            torch.save(model.state_dict(), os.path.join("checkpoints", "unified_k_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # test evaluation by K-MAE
    model.load_state_dict(torch.load(os.path.join("checkpoints", "unified_k_best.pth"), map_location=device))
    model.eval()
    with torch.no_grad():
        khat_list = []
        ktrue_list = []
        for batch in test_loader:
            emb = batch["embedding"].to(device)
            mask = batch["mask"].to(device)
            t_idx = batch["t_idx"].to(device)
            y_loss = batch["y_loss"].to(device)
            y_inc = batch["y_inc"].to(device)
            loss_pred, inc_pred, _, _ = model(emb, mask, t_idx)
            ktrue_np = batch_logk_from_series(y_loss, y_inc)
            khat_np = batch_logk_from_series(loss_pred, inc_pred)
            m = ~np.isnan(ktrue_np) & ~np.isnan(khat_np)
            if m.any():
                khat_list.append(khat_np[m])
                ktrue_list.append(ktrue_np[m])
        if khat_list:
            khat = np.concatenate(khat_list)
            ktrue = np.concatenate(ktrue_list)
            test_k_mae = float(np.mean(np.abs(khat - ktrue)))
        else:
            test_k_mae = float("nan")
    print(f"Test K-MAE (logK): {test_k_mae:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--lambda_k", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_and_eval(
        train_df_path=args.train_csv,
        val_df_path=args.val_csv,
        test_df_path=args.test_csv,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_k=args.lambda_k,
        patience=args.patience,
    )


