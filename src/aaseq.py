import argparse
import time
import os
import pandas as pd

from .datasets.custom_dataset_loader import Datasets_AASeq
from .models.Transformer_based import Transformer_AASeq
from .train_test import run_model


def main():
    parser = argparse.ArgumentParser(description="AASeq training/evaluation with baseline-like outputs")
    parser.add_argument("--input_type", type=str, default="ESM2", choices=["ESM2"], help="Feature type")
    parser.add_argument(
        "--arch",
        type=str,
        default="Transformer",
        choices=["Transformer"],
        help="Model architecture (Transformer only supported)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="both",
        choices=["both", "incorp", "turnover"],
        help="Train for loss+incorp multitask, or direct turnover",
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of folds to evaluate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument(
        "--clusters",
        type=str,
        default="",
        help="Comma-separated cluster IDs to include (e.g., '0,2,4'). Empty for all",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sampling",
        choices=["sampling", "normal", "customized"],
        help="Which prepared dataset split to use",
    )
    parser.add_argument(
        "--dataset_base",
        type=str,
        default="data/dataset",
        help="Base directory for datasets (contains sampling/normal/customized)",
    )
    args = parser.parse_args()

    input_type = args.input_type
    arch = args.arch
    target_mode = args.target
    direct_turnover = target_mode == "turnover"

    # Dataset and input_dim
    data_loader = Datasets_AASeq
    input_dim = 1280 + 7  # residue emb (1280) + 7 lag features

    # Model class and type
    model_cls = Transformer_AASeq
    model_type = "AASeq_Transformer_based"

    run_id = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    clusters_filter = [int(x) for x in args.clusters.split(",") if x.strip() != ""] if args.clusters else None

    for fold in range(args.folds):
        base_dir = os.path.join(args.dataset_base, args.dataset, f"fold_{fold}")
        train_path = os.path.join(base_dir, "train_fold.csv")
        val_path = os.path.join(base_dir, "validation_fold.csv")
        test_path = os.path.join(base_dir, "test_fold.csv")

        suffix = f"{input_type}_{target_mode}_{arch}_{args.dataset}"
        out_dir = os.path.join("reports", "results", "aaseq", suffix, "runs", run_id, f"fold_{fold}")

        result = run_model(
            data_loader=data_loader,
            model=model_cls,
            model_type=model_type,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            input_cls=False,
            input_dim=input_dim,
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation_func=args.activation,
            lr=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            datasize="all",
            criterion="MSE",
            optimizer="Adam",
            early_stop_patience=6,
            early_stop_threshold=0.01,
            scheduler_patience=3,
            scheduler_threshold=0.02,
            model_save_path=os.path.join("data", "models", f"aaseq_{suffix}_fold{fold}.pth"),
            result_save_dir=out_dir,
            plt_save_dir=None,
            target_mode=target_mode,
            direct_turnover=direct_turnover,
            clusters_filter=clusters_filter,
        )

        # Dump run parameters per fold
        run_params = {
            "input_type": input_type,
            "arch": arch,
            "target": target_mode,
            "fold": fold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "activation": args.activation,
            "dataset": args.dataset,
            "dataset_base": args.dataset_base,
            "clusters_filter": clusters_filter,
        }
        with open(os.path.join(out_dir, "run_params.json"), "w") as f:
            import json
            json.dump(run_params, f, indent=2)

        if isinstance(result, tuple):
            no_recursive_scores, recursive_scores = result
            scores = {**no_recursive_scores, **{f"Recursive_{k}": v for k, v in recursive_scores.items()}}
        else:
            scores = result

        rows.append({"fold": fold, "Model": arch, **scores})

    results_df = pd.DataFrame(rows)
    suffix = f"{input_type}_{target_mode}_{arch}_{args.dataset}"
    out_root = os.path.join("reports", "results", "aaseq", suffix, "runs", run_id)
    os.makedirs(out_root, exist_ok=True)
    results_df.to_csv(os.path.join(out_root, "results.csv"), index=False)

    base_metric_cols = [
        "RMSE_Loss",
        "R2_Loss",
        "RMSE_Incorporation",
        "R2_Incorporation",
        "RMSE_Turnover",
        "R2_Turnover",
    ]
    extra_cols = [c for c in results_df.columns if c.startswith("Recursive_")]
    metric_cols = [c for c in base_metric_cols if c in results_df.columns] + extra_cols

    avg = results_df[metric_cols].mean().round(6)
    std = results_df[metric_cols].std().round(6)
    avg.to_frame("mean").to_csv(os.path.join(out_root, "avg_results.csv"))
    std.to_frame("std").to_csv(os.path.join(out_root, "std_results.csv"))

    summary = pd.DataFrame({"metric": metric_cols, "mean": [avg[m] for m in metric_cols], "std": [std[m] for m in metric_cols]})
    summary.to_csv(os.path.join(out_root, "overall_summary.csv"), index=False)

    md_lines = ["| Metric | mean ± std |", "|---|---:|"]
    for _, row in summary.iterrows():
        md_lines.append(f"| {row['metric']} | {row['mean']:.4f} ± {row['std']:.4f} |")
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()


