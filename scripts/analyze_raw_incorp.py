#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_time_columns(columns: List[str]) -> List[str]:
    time_cols = [c for c in columns if isinstance(c, str) and c.startswith("corrRatio_") and c.endswith("h")]
    return time_cols


def find_comb_k_column(columns: List[str]) -> Tuple[str, List[str]]:
    # pandas will rename duplicate headers as 'comb.K', 'comb.K.1', 'comb.K.2', ...
    comb_k_cols = [c for c in columns if c == "comb.K" or c.startswith("comb.K.")]
    if not comb_k_cols:
        raise ValueError("No comb.K-like column found. Expected peptidoform-level comb.K in the CSV.")
    # Order in source header: evidence A/B/K/Rsq, then peptidoform comb.A/B/K/Rsq, then protein comb.A/B/K/Rsq
    # So take the first 'comb.K' occurrence as peptidoform-level
    peptidoform_comb_k = comb_k_cols[0]
    return peptidoform_comb_k, comb_k_cols


def select_base_key(columns: List[str]) -> str:
    # Prefer explicit sequence column if present
    for cand in ["Sequence", "Peptide", "Peptidoform"]:
        if cand in columns:
            return cand
    raise ValueError("No suitable peptide key found (looked for Sequence/Peptide/Peptidoform)")


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def aggregate_by_peptidoform(df: pd.DataFrame, time_cols: List[str], comb_k_col: str) -> pd.DataFrame:
    df = df.copy()
    # PTM flags are given in columns: 'ac-Nterm', 'ac-K', 'ph-STY'
    for col in ["ac-Nterm", "ac-K", "ph-STY"]:
        if col not in df.columns:
            df[col] = 0
    # Coerce PTM indicators to numeric and form has_ptm
    for col in ["ac-Nterm", "ac-K", "ph-STY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["has_ptm"] = (df[["ac-Nterm", "ac-K", "ph-STY"]].sum(axis=1) > 0).astype(int)

    base_key = select_base_key(df.columns.tolist())
    df["BaseKey"] = df[base_key]
    group_cols = ["Peptidoform", "BaseKey", "ac-Nterm", "ac-K", "ph-STY", "has_ptm"]
    agg_cols = time_cols + [comb_k_col]

    agg_df = (
        df[group_cols + agg_cols]
        .groupby(group_cols, as_index=False)
        .agg({**{c: "mean" for c in agg_cols}})
    )
    agg_df.rename(columns={comb_k_col: "comb.K"}, inplace=True)
    # Keep track of counts per peptidoform+ptm-state
    counts = df[group_cols].value_counts().reset_index(name="n_entries")
    agg_df = agg_df.merge(counts, on=group_cols, how="left")
    return agg_df


def plot_k_distribution(agg_df: pd.DataFrame, out_dir: str) -> None:
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 5))
    sns.histplot(agg_df["comb.K"].dropna(), bins=60, kde=True, color="#4472C4")
    plt.xlabel("comb.K (peptidoform level)")
    plt.ylabel("Count")
    plt.title("Distribution of comb.K across peptidoforms")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "k_distribution.png"), dpi=200)
    plt.close()

    # Facet by PTM state
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=agg_df, x="comb.K", hue="has_ptm", fill=True, common_norm=False, palette=["#2E7D32", "#C62828"])
    plt.xlabel("comb.K (peptidoform level)")
    plt.title("K density by PTM state (0 = no PTM, 1 = any PTM)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "k_density_by_ptm.png"), dpi=200)
    plt.close()


def ptm_vs_nonptm_comparisons(agg_df: pd.DataFrame, time_cols: List[str], out_dir_fig: str, out_path_table: str, out_summary_table: str) -> pd.DataFrame:
    # Collapse to per (Peptidoform, has_ptm) by averaging in case of multiple PTM combinations
    collapsed = (
        agg_df[["BaseKey", "Peptidoform", "has_ptm", "comb.K"] + time_cols]
        .groupby(["BaseKey", "Peptidoform", "has_ptm"], as_index=False)
        .mean()
    )

    ptm = collapsed[collapsed["has_ptm"] == 1]
    non = collapsed[collapsed["has_ptm"] == 0]
    # Pair within the same BaseKey (same underlying peptide) even if Peptidoform strings differ
    merged = ptm.merge(non, on="BaseKey", suffixes=("_ptm", "_no"))

    # Compute metrics
    results = []
    for _, row in merged.iterrows():
        k_ptm = row["comb.K_ptm"]
        k_no = row["comb.K_no"]
        delta_k = k_ptm - k_no
        ts_ptm = row[[f"{c}_ptm" for c in time_cols]].to_numpy(dtype=float)
        ts_no = row[[f"{c}_no" for c in time_cols]].to_numpy(dtype=float)
        # Handle NaNs safely
        valid = ~(np.isnan(ts_ptm) | np.isnan(ts_no))
        if valid.sum() < 2:
            corr = np.nan
            l2 = np.nan
        else:
            corr = np.corrcoef(ts_ptm[valid], ts_no[valid])[0, 1]
            l2 = float(np.linalg.norm(ts_ptm[valid] - ts_no[valid]))
        # Prefer to report the non-PTM Peptidoform as key when available
        key_name = row.get("Peptidoform_no", row.get("BaseKey"))
        results.append((key_name, k_no, k_ptm, delta_k, corr, l2))

    res_df = pd.DataFrame(results, columns=["Peptidoform", "K_no_ptm", "K_with_ptm", "delta_K", "corr", "l2_distance"])
    res_df.sort_values(by="delta_K", key=lambda s: s.abs(), ascending=False, inplace=True)
    res_df.to_csv(out_path_table, index=False)

    # Also write a simple group-level K summary by PTM state
    k_summary = (
        agg_df[["has_ptm", "comb.K"]]
        .dropna()
        .groupby("has_ptm")
        .agg(n=("comb.K", "size"), mean_K=("comb.K", "mean"), median_K=("comb.K", "median"), std_K=("comb.K", "std"))
        .reset_index()
    )
    k_summary.to_csv(out_summary_table, index=False)

    # Plot histogram of delta_K
    plt.figure(figsize=(8, 5))
    sns.histplot(res_df["delta_K"].dropna(), bins=60, kde=True, color="#6A5ACD")
    plt.xlabel("Delta K (with PTM - no PTM)")
    plt.ylabel("Count")
    plt.title("Distribution of K differences between PTM vs non-PTM")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_fig, "delta_k_distribution.png"), dpi=200)
    plt.close()

    # Line overlays for top-N absolute delta_K
    top = res_df.head(12)["Peptidoform"].tolist()
    n = len(top)
    if n:
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharex=True)
        axes = np.array(axes).reshape(nrows, ncols)
        x = list(range(len(time_cols)))
        x_labels = [c.replace("corrRatio_", "").replace("h", "h") for c in time_cols]
        for idx, pep in enumerate(top):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            # select within same BaseKey; use first matching peptidoforms for display
            base = pep
            row_no = non[non["BaseKey"] == base].head(1)
            row_ptm = ptm[ptm["BaseKey"] == base].head(1)
            if not row_ptm.empty and not row_no.empty:
                y_ptm = row_ptm[time_cols].iloc[0].to_numpy(dtype=float)
                y_no = row_no[time_cols].iloc[0].to_numpy(dtype=float)
                ax.plot(x, y_no, label="no PTM", color="#2E7D32")
                ax.plot(x, y_ptm, label="with PTM", color="#C62828")
                ax.set_title(str(pep))
        for ax in axes.ravel():
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout(rect=[0, 0, 0.95, 1])
        fig.savefig(os.path.join(out_dir_fig, "ptm_vs_nonptm_top_deltaK.png"), dpi=200)
        plt.close(fig)

    return res_df


def main():
    parser = argparse.ArgumentParser(description="Analyze raw pSILAC TMT incorporation time series by peptidoform and PTM state.")
    parser.add_argument("--raw_csv", type=str, default="/home/ishino/PTP/data/raw/pSILAC_TMT.csv", help="Path to raw CSV")
    parser.add_argument("--out_interim", type=str, default="/home/ishino/PTP/data/interim/incorp_by_peptidoform.csv", help="Output CSV for aggregated peptidoforms")
    parser.add_argument("--fig_dir", type=str, default="/home/ishino/PTP/reports/figures/raw_incorp", help="Directory to save figures")
    parser.add_argument("--table_dir", type=str, default="/home/ishino/PTP/reports/tables/raw_incorp", help="Directory to save comparison tables")
    args = parser.parse_args()

    ensure_dirs([os.path.dirname(args.out_interim), args.fig_dir, args.table_dir])

    # Robust header handling: sometimes the first row is a description, with real header on row 2
    df = pd.read_csv(args.raw_csv, low_memory=False)
    time_cols = find_time_columns(df.columns.tolist())
    if not time_cols:
        # Retry assuming header is on the second row (0-based header=1)
        df = pd.read_csv(args.raw_csv, low_memory=False, header=1)
        time_cols = find_time_columns(df.columns.tolist())
    if not time_cols:
        raise ValueError("No corrRatio_*h columns found after trying multiple header rows; cannot proceed.")

    # Coerce time series and K columns to numeric
    comb_k_col, _ = find_comb_k_column(df.columns.tolist())
    for c in time_cols + [comb_k_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_df = aggregate_by_peptidoform(df, time_cols, comb_k_col)
    agg_df.to_csv(args.out_interim, index=False)

    plot_k_distribution(agg_df, args.fig_dir)

    comp_path = os.path.join(args.table_dir, "ptm_vs_nonptm_stats.csv")
    comp_summary_path = os.path.join(args.table_dir, "k_summary_by_ptm.csv")
    res_df = ptm_vs_nonptm_comparisons(agg_df, time_cols, args.fig_dir, comp_path, comp_summary_path)
    print(f"Matched PTM/non-PTM pairs: {len(res_df)} (see {comp_path})")


if __name__ == "__main__":
    main()


