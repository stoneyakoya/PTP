#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K evaluation for pSILAC-TMT evidence-level time series.

Overview
--------
- Fit K per evidence with robust nonlinear least squares (soft_l1)
- Aggregate median K per (Peptidoform, TurnoverType)
- Compare to comb.K truth (groups passing comb.Rsq >= 0.8)
- Metrics: MAE/RMSE on logK, Spearman/Pearson on logK, AUROC/AUPRC for fast/slow
- Plots: scatter (log-log) with 45° line, MAE(logK) boxplot, R² violin, ROC
- Save: results/k_eval.csv and figures under results/

Models
------
- TurnoverType = Label loss:          y = A * exp(-K * t) + B
- TurnoverType = Label incorporation: y = A * (1 - exp(-K * t)) + B

Time points
-----------
t = [0, 1, 3, 6, 10, 16, 24, 34, 48] hours. Column names are corrRatio_{th}.
'infin.h' is excluded from fitting.

Usage
-----
python k_eval.py --csv data/raw/pSILAC_TMT.csv --outdir results

Dependencies
------------
pandas, numpy, scipy, scikit-learn, matplotlib

Synthetic check
---------------
Simple unit-test-style synthetic curves (K=0.05) are fit and reported.
"""

import argparse
import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    r2_score,
)


# -----------------------------
# Configuration
# -----------------------------

TIME_HOURS = [0, 1, 3, 6, 10, 16, 24, 34, 48]
TIME_COLS = [f"corrRatio_{h}h" for h in TIME_HOURS]

# Initial parameter defaults
DEFAULT_K0 = np.log(2) / np.median([h for h in TIME_HOURS if h > 0])  # ~ half-life at median t>0


# -----------------------------
# Utilities
# -----------------------------

def _safe_log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log(np.clip(x, a_min=np.finfo(float).tiny, a_max=None))


def _compute_initial_params(y: np.ndarray) -> Tuple[float, float, float]:
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    a0 = max(y_max - y_min, 0.0)
    b0 = y_min
    return a0, b0, float(DEFAULT_K0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return np.nan
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


# -----------------------------
# Model definitions
# -----------------------------

def _model_loss(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, k = params
    return a * np.exp(-k * t) + b


def _model_incorp(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, k = params
    return a * (1.0 - np.exp(-k * t)) + b


def _residuals(model_fn, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return model_fn(params, t) - y


def fit_k_loss(y: np.ndarray, t: np.ndarray) -> Optional[Dict[str, float]]:
    """Robustly fit y = A*exp(-K*t) + B.

    Returns dict with keys: A, B, K, success, y_pred, r2
    or None if not enough points.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = ~np.isnan(y)
    y_used = y[mask]
    t_used = t[mask]
    if y_used.size < 3:
        return None

    a0, b0, k0 = _compute_initial_params(y_used)
    x0 = np.array([a0, b0, k0], dtype=float)
    # Bounds: A >= 0, K >= 0, B free
    lb = np.array([0.0, -np.inf, 0.0], dtype=float)
    ub = np.array([np.inf, np.inf, np.inf], dtype=float)

    res = least_squares(
        fun=lambda p: _residuals(_model_loss, p, t_used, y_used),
        x0=x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=5000,
    )

    a_hat, b_hat, k_hat = res.x
    y_pred = _model_loss(res.x, t_used)
    r2 = _r2(y_used, y_pred)
    return {
        "A": float(a_hat),
        "B": float(b_hat),
        "K": float(max(k_hat, 0.0)),
        "success": bool(res.success),
        "y_pred": y_pred.astype(float),
        "r2": float(r2),
    }


def fit_k_incorp(y: np.ndarray, t: np.ndarray) -> Optional[Dict[str, float]]:
    """Robustly fit y = A*(1 - exp(-K*t)) + B.

    Returns dict with keys: A, B, K, success, y_pred, r2
    or None if not enough points.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = ~np.isnan(y)
    y_used = y[mask]
    t_used = t[mask]
    if y_used.size < 3:
        return None

    a0, b0, k0 = _compute_initial_params(y_used)
    x0 = np.array([a0, b0, k0], dtype=float)
    # Bounds: A >= 0, K >= 0, B free
    lb = np.array([0.0, -np.inf, 0.0], dtype=float)
    ub = np.array([np.inf, np.inf, np.inf], dtype=float)

    res = least_squares(
        fun=lambda p: _residuals(_model_incorp, p, t_used, y_used),
        x0=x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=5000,
    )

    a_hat, b_hat, k_hat = res.x
    y_pred = _model_incorp(res.x, t_used)
    r2 = _r2(y_used, y_pred)
    return {
        "A": float(a_hat),
        "B": float(b_hat),
        "K": float(max(k_hat, 0.0)),
        "success": bool(res.success),
        "y_pred": y_pred.astype(float),
        "r2": float(r2),
    }


# -----------------------------
# Synthetic quick test
# -----------------------------

def run_synthetic_check(random_seed: int = 42) -> None:
    rng = np.random.default_rng(random_seed)
    K_true = 0.05
    A_true = 0.8
    B_true = 0.1
    t = np.asarray(TIME_HOURS, dtype=float)
    noise_sd = 0.01

    y_loss = A_true * np.exp(-K_true * t) + B_true + rng.normal(0, noise_sd, size=t.size)
    y_incorp = A_true * (1.0 - np.exp(-K_true * t)) + B_true + rng.normal(0, noise_sd, size=t.size)

    out_loss = fit_k_loss(y_loss, t)
    out_incorp = fit_k_incorp(y_incorp, t)

    if out_loss is not None and out_incorp is not None:
        print(f"[Synthetic] True K={K_true:.5f} | Fitted loss K={out_loss['K']:.5f}, incorp K={out_incorp['K']:.5f}")
        # Basic sanity tolerance: within 30% relative error
        rel_err_loss = abs(out_loss["K"] - K_true) / K_true
        rel_err_incorp = abs(out_incorp["K"] - K_true) / K_true
        if rel_err_loss > 0.3 or rel_err_incorp > 0.3:
            warnings.warn("Synthetic check: fitted K deviates >30% from true.")
    else:
        warnings.warn("Synthetic check skipped due to insufficient points.")


# -----------------------------
# Main evaluation
# -----------------------------

def _normalize_turnover_type(x: str) -> Optional[str]:
    if not isinstance(x, str):
        return None
    x_lower = x.strip().lower()
    if "incorp" in x_lower:
        return "incorporation"
    if "loss" in x_lower:
        return "loss"
    return None


def evaluate(csv_path: str, outdir: str) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(outdir, exist_ok=True)

    # Load data
    usecols = [
        "Peptidoform",
        "Evidence ID",
        "Replicate",
        "TurnoverType",
        *TIME_COLS,
        "infin.h",
        "comb.K",
        "comb.Rsq",
    ]
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols, low_memory=False)

    # Normalize TurnoverType
    df["TurnoverTypeNorm"] = df["TurnoverType"].apply(_normalize_turnover_type)
    df = df.dropna(subset=["TurnoverTypeNorm"]).copy()

    t_array = np.asarray(TIME_HOURS, dtype=float)

    # Fit per evidence
    records = []
    for row in df.itertuples(index=False, name=None):
        # Build a reliable dict mapping original column names to values
        row_dict = {col: val for col, val in zip(df.columns, row)}
        turn = row_dict.get("TurnoverTypeNorm", None)
        if turn not in ("loss", "incorporation"):
            continue

        y_vals = np.array([row_dict.get(col, np.nan) for col in TIME_COLS], dtype=float)
        # Exclude NaNs automatically in fit functions
        if np.sum(~np.isnan(y_vals)) < 3:
            continue

        if turn == "loss":
            res = fit_k_loss(y_vals, t_array)
        else:
            res = fit_k_incorp(y_vals, t_array)

        if res is None:
            continue

        records.append({
            "Peptidoform": row_dict.get("Peptidoform"),
            "TurnoverType": row_dict.get("TurnoverType"),
            "TurnoverTypeNorm": turn,
            "Evidence ID": row_dict.get("Evidence ID"),
            "Replicate": row_dict.get("Replicate"),
            "K_pred": res["K"],
            "Rsq_pred": res["r2"],
        })

    ev_df = pd.DataFrame.from_records(records)
    if ev_df.empty:
        print("No evidence-level fits produced. Exiting.")
        return

    # Aggregate K by (Peptidoform, TurnoverTypeNorm)
    agg_df = (
        ev_df.groupby(["Peptidoform", "TurnoverTypeNorm"], as_index=False)
        .agg(K_pred_group=("K_pred", "median"))
    )

    # Truth: comb.K per group, only from rows with comb.Rsq >= 0.8
    good_df = df[df["comb.Rsq"] >= 0.8].copy()
    truth_df = (
        good_df.groupby(["Peptidoform", "TurnoverTypeNorm"], as_index=False)
        .agg(comb_K=("comb.K", "median"), comb_Rsq_group=("comb.Rsq", "median"))
    )

    eval_df = pd.merge(
        agg_df,
        truth_df,
        on=["Peptidoform", "TurnoverTypeNorm"],
        how="inner",
    )
    if eval_df.empty:
        print("No evaluable groups (comb.Rsq >= 0.8). Exiting.")
        return

    # Errors on logK
    eval_df["logK_pred"] = _safe_log(eval_df["K_pred_group"].values)
    eval_df["logK_true"] = _safe_log(eval_df["comb_K"].values)
    eval_df["logK_err_abs"] = np.abs(eval_df["logK_pred"] - eval_df["logK_true"])

    # Delta logK within the same Peptidoform across loss/incorporation (predicted)
    delta_pred = (
        eval_df.pivot(index="Peptidoform", columns="TurnoverTypeNorm", values="logK_pred")
        .assign(Delta_logK=lambda d: (d.get("loss") - d.get("incorporation")).abs())
        .reset_index()[["Peptidoform", "Delta_logK"]]
    )
    eval_df = pd.merge(eval_df, delta_pred, on="Peptidoform", how="left")

    # Also compute Delta_logK on truth for reference
    delta_true = (
        eval_df.pivot(index="Peptidoform", columns="TurnoverTypeNorm", values="logK_true")
        .assign(Delta_logK_true=lambda d: (d.get("loss") - d.get("incorporation")).abs())
        .reset_index()[["Peptidoform", "Delta_logK_true"]]
    )
    eval_df = pd.merge(eval_df, delta_true, on="Peptidoform", how="left")

    # Metrics
    y_true_log = eval_df["logK_true"].values
    y_pred_log = eval_df["logK_pred"].values

    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    sp_r, sp_p = spearmanr(y_true_log, y_pred_log, nan_policy="omit")
    pe_r, pe_p = pearsonr(y_true_log, y_pred_log)

    # Classification threshold on comb.K median
    k_true = eval_df["comb_K"].values
    k_pred = eval_df["K_pred_group"].values
    thresh = float(np.median(k_true))
    y_cls_true = (k_true > thresh).astype(int)
    y_scores = k_pred.astype(float)

    try:
        auroc = float(roc_auc_score(y_cls_true, y_scores))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_cls_true, y_scores))
    except Exception:
        auprc = float("nan")

    # Console reporting
    print("=== Metrics (logK) ===")
    print(f"MAE(logK):  {mae_log:.4f}")
    print(f"RMSE(logK): {rmse_log:.4f}")
    print(f"Spearman r: {sp_r:.4f} (p={sp_p:.2e})")
    print(f"Pearson  r: {pe_r:.4f} (p={pe_p:.2e})")
    print("=== Classification (fast/slow by comb.K median) ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # Save aggregated evaluation table
    out_csv = os.path.join(outdir, "k_eval.csv")
    # Harmonize column names per spec
    out_tbl = eval_df.rename(columns={
        "TurnoverTypeNorm": "TurnoverType",
        "comb_K": "comb.K",
        "comb_Rsq_group": "comb.Rsq_group",
    })[
        [
            "Peptidoform",
            "TurnoverType",
            "K_pred_group",
            "comb.K",
            "logK_pred",
            "logK_true",
            "logK_err_abs",
            "Delta_logK",
            "Delta_logK_true",
        ]
    ].copy()
    out_tbl.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # -----------------------------
    # Plots
    # -----------------------------
    # 1) Scatter K_pred_group vs comb.K (log-log) with 45° line
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(eval_df["comb_K"], eval_df["K_pred_group"], s=10, alpha=0.6)
    mn = float(np.nanmin([eval_df["comb_K"].min(), eval_df["K_pred_group"].min()]))
    mx = float(np.nanmax([eval_df["comb_K"].max(), eval_df["K_pred_group"].max()]))
    mn = max(mn, np.finfo(float).tiny)
    ax.plot([mn, mx], [mn, mx], color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("comb.K (truth)")
    ax.set_ylabel("K_pred_group (median)")
    ax.set_title("K_pred_group vs comb.K (log-log)")
    fig.tight_layout()
    fig_path = os.path.join(outdir, "scatter_Kpred_vs_combK_loglog.png")
    fig.savefig(fig_path)
    plt.close(fig)

    # 2) MAE(logK) boxplot (by TurnoverType)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    box_data = []
    labels = []
    for typ in ("loss", "incorporation"):
        vals = eval_df.loc[eval_df["TurnoverTypeNorm"] == typ, "logK_err_abs"].dropna().values
        if vals.size > 0:
            box_data.append(vals)
            labels.append(typ)
    if box_data:
        ax.boxplot(box_data, labels=labels, showfliers=False)
        ax.set_ylabel("|logK_pred - logK_true|")
        ax.set_title("MAE(logK) by TurnoverType (per-group errors)")
        fig.tight_layout()
        fig_path = os.path.join(outdir, "boxplot_MAE_logK.png")
        fig.savefig(fig_path)
    plt.close(fig)

    # 3) R² violin (evidence-level, by TurnoverType)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    vio_data = []
    vio_labels = []
    for typ in ("loss", "incorporation"):
        vals = ev_df.loc[ev_df["TurnoverTypeNorm"] == typ, "Rsq_pred"].dropna().values
        if vals.size > 0:
            vio_data.append(vals)
            vio_labels.append(typ)
    if vio_data:
        parts = ax.violinplot(vio_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(vio_labels) + 1))
        ax.set_xticklabels(vio_labels)
        ax.set_ylabel("R² (evidence-level)")
        ax.set_title("R² distribution by TurnoverType")
        fig.tight_layout()
        fig_path = os.path.join(outdir, "violin_R2_evidence.png")
        fig.savefig(fig_path)
    plt.close(fig)

    # 4) ROC curve for fast/slow classification
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    try:
        fpr, tpr, _ = roc_curve(y_cls_true, y_scores)
        ax.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC: fast/slow (threshold = median comb.K)")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(outdir, "roc_fast_slow.png")
        fig.savefig(fig_path)
    except Exception:
        pass
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate K from pSILAC-TMT CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("data", "raw", "pSILAC_TMT.csv"),
        help="Path to input CSV",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("results"),
        help="Directory to save results",
    )
    return parser.parse_args()


def main() -> None:
    run_synthetic_check()
    args = parse_args()
    evaluate(csv_path=args.csv, outdir=args.outdir)


if __name__ == "__main__":
    main()


