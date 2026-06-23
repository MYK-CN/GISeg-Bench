"""
GISeg-Bench  Unified Entry Point
=================================
**System Orchestrator** — the single command that drives the entire
benchmark lifecycle:

    config → train (24 models × 4 datasets) → inference → metrics → report

Usage::

    # Full benchmark (all 24 models × all 4 datasets)
    python main.py

    # Quick test — 1 model, 1 dataset, few epochs
    python main.py --quick

    # Run specific models on specific datasets
    python main.py --models unet,pranet,swin_unet --datasets kvasir,cvc

    # Skip training (inference + metrics only, using existing checkpoints)
    python main.py --skip_train

    # Skip inference (metrics only, using existing predictions)
    python main.py --skip_train --skip_infer

Output structure::

    outputs/
    └── unet/
        ├── kvasir/
        │   ├── best.pth
        │   ├── final.pth
        │   ├── metrics.json
        │   └── log.txt
        ├── cvc/
        │   └── ...
        └── ...

Design rules:
    - ❌ Never modifies existing modules — only calls them
    - ✔ Pure orchestrator — schedule, run, collect, report
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
#  Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

# ===================================================================
#  Imports from existing modules (read-only)
# ===================================================================
from configs.config_loader import load_config
from configs.model_config import MODEL_REGISTRY, list_models
from configs.dataset_config import DATASET_REGISTRY, list_datasets

from datasets.dataset_zoo import get_dataset
from datasets.transforms import SegTransform

from inference.loader import build_model, load_checkpoint
from inference.predictor import run_inference
from inference.utils import get_device

from metrics.evaluator import evaluate
from metrics.aggregator import Aggregator

from utils import seed_all, ensure_dir, setup_logger, get_logger


# ===================================================================
#  Constants
# ===================================================================
ALL_MODELS   = list_models()          # 24 models
ALL_DATASETS = list_datasets()        # 4 datasets


# ===================================================================
#  Step 1: Build model × dataset grid
# ===================================================================

def build_task_grid(models=None, datasets=None):
    """Return a list of (model_name, dataset_name) tuples to run.

    Args:
        models:   list or None (all 24).
        datasets: list or None (all 4).

    Returns:
        list of (str, str).
    """
    models   = models or ALL_MODELS
    datasets = datasets or ALL_DATASETS

    # Validate
    for m in models:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{m}'. Available: {ALL_MODELS}")
    for d in datasets:
        if d not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{d}'. Available: {ALL_DATASETS}")

    grid = []
    for model in models:
        for ds in datasets:
            grid.append((model, ds))
    return grid


# ===================================================================
#  Step 2: Train one model on one dataset
# ===================================================================

def run_train(model_name, dataset_name, cfg, output_dir):
    """Train *model_name* on *dataset_name*.

    Args:
        model_name:   e.g. "unet".
        dataset_name: e.g. "kvasir".
        cfg:          Config object from config_loader.
        output_dir:   directory for checkpoints + logs.

    Returns:
        Path to ``best.pth``, or None if training was skipped.
    """
    best_path = os.path.join(output_dir, "best.pth")

    # ---- skip if already trained ----
    if os.path.isfile(best_path):
        get_logger().info(f"[{model_name}/{dataset_name}] best.pth exists — skip training")
        return best_path

    ensure_dir(output_dir)
    logger = setup_logger(
        name=f"train_{model_name}_{dataset_name}",
        log_dir=output_dir,
        level="info",
        console=False,           # file only — tqdm handles console
        file_prefix="train",
    )
    logger.info(f"Training {model_name} on {dataset_name}")
    logger.info(f"Config: epochs={cfg.epochs} lr={cfg.lr} bs={cfg.batch_size}")

    # ---- build model ----
    model = build_model(
        model_name=model_name,
        n_classes=cfg.n_classes,
        image_size=cfg.image_size,
    )

    # ---- data ----
    tf = SegTransform(size=cfg.image_size, normalise="imagenet")
    train_ds = get_dataset(
        dataset_name, root=cfg.data_root,
        split="train", split_file=cfg.get("split_file"),
        transform=tf,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.get("num_workers", 0),
    )

    # Try val split
    val_loader = None
    try:
        val_ds = get_dataset(
            dataset_name, root=cfg.data_root,
            split="val", split_file=cfg.get("split_file"),
            transform=tf,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.get("num_workers", 0),
        )
    except Exception:
        logger.info("No val split — validation skipped")

    # ---- trainer ----
    from trainer.trainer_core import Trainer
    from trainer.callbacks import (
        ConsoleReporter, BestModelCheckpoint, EarlyStopping,
    )

    device = get_device(prefer_gpu=True)
    model = model.to(device)

    # Convert Config to plain dict for Trainer compatibility
    cfg_dict = cfg.to_dict()
    cfg_dict["output_dir"] = output_dir

    trainer = Trainer(model, train_loader, cfg_dict, val_loader)
    trainer.set_callbacks([
        ConsoleReporter(),
        BestModelCheckpoint(output_dir, monitor="train_dice"),
        EarlyStopping(monitor="train_dice", patience=cfg.patience),
    ])
    trainer.run()

    logger.info(f"Training complete — best model at {best_path}")
    return best_path


# ===================================================================
#  Step 3: Inference on test set
# ===================================================================

def run_infer(model_name, dataset_name, cfg, checkpoint_path, output_dir):
    """Run inference on the test split.

    Args:
        model_name:      e.g. "unet".
        dataset_name:    e.g. "kvasir".
        cfg:             Config object.
        checkpoint_path: path to best.pth.
        output_dir:      where to save inference summary.

    Returns:
        Inference results dict (in-memory).
    """
    ensure_dir(output_dir)
    device = get_device(prefer_gpu=True)

    get_logger().info(f"[{model_name}/{dataset_name}] Inference on test set")

    # ---- build & load model ----
    model = build_model(
        model_name=model_name,
        n_classes=cfg.n_classes,
        image_size=cfg.image_size,
    )
    model = load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)

    # ---- test data ----
    tf = SegTransform(size=cfg.image_size, normalise="imagenet")
    test_ds = get_dataset(
        dataset_name, root=cfg.data_root,
        split="test", split_file=cfg.get("split_file"),
        transform=tf,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("batch_size", 8),
        shuffle=False, num_workers=0,
    )
    get_logger().info(f"[{model_name}/{dataset_name}] Test samples: {len(test_ds)}")

    # ---- run ----
    results = run_inference(
        model=model,
        test_loader=test_loader,
        device=str(device),
        threshold=cfg.get("threshold", 0.5),
        return_logits=True,
    )

    return results


# ===================================================================
#  Step 4: Evaluate metrics
# ===================================================================

def run_metrics(model_name, dataset_name, inference_results, output_dir):
    """Compute metrics from inference results and save metrics.json.

    Args:
        model_name:        e.g. "unet".
        dataset_name:      e.g. "kvasir".
        inference_results: dict from run_inference().
        output_dir:        where to save metrics.json.

    Returns:
        Flat metrics dict.
    """
    ensure_dir(output_dir)

    get_logger().info(f"[{model_name}/{dataset_name}] Computing metrics")

    # ---- compute ----
    report = evaluate(
        inference_results,
        model=model_name,
        dataset=dataset_name,
    )
    flat = {
        "model":    report["model"],
        "dataset":  report["dataset"],
        "n_samples": report["n_samples"],
        "Dice_mean":   report["Dice"]["mean"],
        "Dice_std":    report["Dice"]["std"],
        "IoU_mean":    report["IoU"]["mean"],
        "IoU_std":     report["IoU"]["std"],
        "HD95_mean":   report["HD95"]["mean"],
        "HD95_std":    report["HD95"]["std"],
        "Precision_mean": report["Precision"]["mean"],
        "Precision_std":  report["Precision"]["std"],
        "Recall_mean":    report["Recall"]["mean"],
        "Recall_std":     report["Recall"]["std"],
        "timestamp":      datetime.now().isoformat(),
    }

    # ---- save ----
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(flat, f, indent=2)

    get_logger().info(f"[{model_name}/{dataset_name}] Metrics saved → {metrics_path}")
    return flat


# ===================================================================
#  Step 5: Save inference summary
# ===================================================================

def save_inference_summary(model_name, dataset_name, inference_results, output_dir):
    """Save a lightweight inference summary (no prediction images!)."""
    summary = {
        "model":      model_name,
        "dataset":    dataset_name,
        "n_predictions": len(inference_results["predictions"]),
        "prediction_shapes": [
            list(p.shape) for p in inference_results["predictions"]
        ][:5],   # first 5 only
        "sample_names": inference_results["names"][:10],
        "timestamp": datetime.now().isoformat(),
    }
    path = os.path.join(output_dir, "inference_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# ===================================================================
#  Step 6: Summarize all results
# ===================================================================

def summarize_results(all_metrics):
    """Print model rankings and per-dataset best models.

    Args:
        all_metrics: list of flat metrics dicts from run_metrics().
    """
    if not all_metrics:
        print("\n[Summary] No results to summarize.\n")
        return

    # ---- Rank models by average Dice across datasets ----
    from collections import defaultdict
    model_scores = defaultdict(list)
    for m in all_metrics:
        model_scores[m["model"]].append(m["Dice_mean"])

    model_avg = {
        model: sum(scores) / len(scores)
        for model, scores in model_scores.items()
    }
    ranking = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)

    print()
    print("=" * 60)
    print("  GISeg-Bench  —  Final Results")
    print("=" * 60)

    # ---- Model Ranking ----
    print()
    print("  Model Ranking (by average Dice):")
    print("  " + "-" * 40)
    for rank, (model, dice) in enumerate(ranking, 1):
        datasets_count = len(model_scores[model])
        print(f"  {rank:>2}. {model:<25s}  Dice = {dice:.4f}  ({datasets_count} datasets)")

    # ---- Per-dataset best ----
    print()
    print("  Best Model per Dataset:")
    print("  " + "-" * 40)
    dataset_best = {}
    for m in all_metrics:
        ds = m["dataset"]
        if ds not in dataset_best or m["Dice_mean"] > dataset_best[ds]["Dice_mean"]:
            dataset_best[ds] = m

    for ds in sorted(dataset_best.keys()):
        m = dataset_best[ds]
        print(f"  {ds:<15s} → {m['model']:<15s}  Dice = {m['Dice_mean']:.4f}")

    # ---- Full table ----
    print()
    print("  Full Results Table:")
    print("  " + "-" * 80)
    header = f"  {'Model':<22s} {'Dataset':<12s} {'Dice':>8s} {'IoU':>8s} {'HD95':>8s} {'Prec':>8s} {'Rec':>8s}"
    print(header)
    print("  " + "-" * 80)

    for m in sorted(all_metrics, key=lambda x: (-x["Dice_mean"], x["model"], x["dataset"])):
        line = (f"  {m['model']:<22s} {m['dataset']:<12s} "
                f"{m['Dice_mean']:>8.4f} {m['IoU_mean']:>8.4f} "
                f"{m['HD95_mean']:>8.2f} {m['Precision_mean']:>8.4f} "
                f"{m['Recall_mean']:>8.4f}")
        print(line)
    print("  " + "-" * 80)

    # ---- Save summary JSON ----
    summary_path = os.path.join(_PROJECT_ROOT, "outputs", "benchmark_summary.json")
    ensure_dir(os.path.dirname(summary_path))
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model_ranking": [
                {"rank": i+1, "model": m, "dice_avg": d}
                for i, (m, d) in enumerate(ranking)
            ],
            "per_dataset_best": {
                ds: {"model": v["model"], "dice": v["Dice_mean"]}
                for ds, v in dataset_best.items()
            },
            "all_results": all_metrics,
        }, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")
    print("=" * 60)


# ===================================================================
#  Orchestrator
# ===================================================================

def run_benchmark(models=None, datasets=None, skip_train=False,
                  skip_infer=False, epochs=None, quick=False):
    """Run the full benchmark pipeline.

    Args:
        models:      list of model names (None = all 24).
        datasets:    list of dataset names (None = all 4).
        skip_train:  if True, skip training and use existing checkpoints.
        skip_infer:  if True, skip inference (metrics-only from cached).
        epochs:      override epochs (for quick test).
        quick:       shortcut: 1 model × 1 dataset, 2 epochs.
    """
    # ---- Quick mode ----
    if quick:
        models = models or ["unet"]
        datasets = datasets or ["kvasir"]
        if epochs is None:
            epochs = 2
        skip_train = False
        skip_infer = False

    # ---- Build grid ----
    grid = build_task_grid(models, datasets)
    print(f"[Benchmark] Models: {models or ALL_MODELS}")
    print(f"[Benchmark] Datasets: {datasets or ALL_DATASETS}")
    print(f"[Benchmark] Tasks: {len(grid)} total\n")

    all_metrics = []

    for idx, (model_name, dataset_name) in enumerate(grid, 1):
        print(f"\n{'#'*60}")
        print(f"#  [{idx}/{len(grid)}]  {model_name}  /  {dataset_name}")
        print(f"{'#'*60}")

        # ---- config ----
        cfg = load_config(
            model=model_name, dataset=dataset_name, mode="train",
        )
        if epochs is not None:
            cfg.epochs = epochs

        # ---- output directory ----
        output_dir = os.path.join(
            _PROJECT_ROOT, "outputs", model_name, dataset_name
        )
        ensure_dir(output_dir)

        # ---- log file ----
        setup_logger(
            name=f"bench_{model_name}_{dataset_name}",
            log_dir=output_dir,
            level="info",
            console=True,
            file_prefix="bench",
        )

        checkpoint_path = os.path.join(output_dir, "best.pth")

        # ============================================================
        #  TRAIN
        # ============================================================
        if not skip_train:
            try:
                checkpoint_path = run_train(
                    model_name, dataset_name, cfg, output_dir
                )
            except Exception as e:
                get_logger().error(
                    f"[{model_name}/{dataset_name}] TRAIN FAILED: {e}"
                )
                import traceback
                traceback.print_exc()
                continue
        else:
            # Find existing checkpoint
            if not os.path.isfile(checkpoint_path):
                # fallback to final.pth
                final_path = os.path.join(output_dir, "final.pth")
                if os.path.isfile(final_path):
                    checkpoint_path = final_path
                else:
                    # search outputs/ subdirs
                    from utils.file_utils import find_checkpoint
                    ckpt = find_checkpoint(output_dir, "best.pth")
                    if ckpt:
                        checkpoint_path = ckpt
                    else:
                        ckpt = find_checkpoint(output_dir, "final.pth")
                        if ckpt:
                            checkpoint_path = ckpt
                        else:
                            get_logger().warn(
                                f"[{model_name}/{dataset_name}] No checkpoint found — skip"
                            )
                            continue

        if not os.path.isfile(checkpoint_path):
            get_logger().error(
                f"[{model_name}/{dataset_name}] Checkpoint not found: {checkpoint_path}"
            )
            continue

        # ============================================================
        #  INFERENCE
        # ============================================================
        if not skip_infer:
            # Switch to infer mode config
            cfg_infer = load_config(
                model=model_name, dataset=dataset_name, mode="infer",
            )
            try:
                infer_results = run_infer(
                    model_name, dataset_name, cfg_infer,
                    checkpoint_path, output_dir,
                )
                save_inference_summary(
                    model_name, dataset_name, infer_results, output_dir,
                )
            except Exception as e:
                get_logger().error(
                    f"[{model_name}/{dataset_name}] INFERENCE FAILED: {e}"
                )
                import traceback
                traceback.print_exc()
                continue
        else:
            get_logger().info(
                f"[{model_name}/{dataset_name}] Skipping inference (--skip_infer)"
            )
            # Need inference results for metrics — can't skip both
            get_logger().warn("Cannot compute metrics without inference — skipping metrics")
            continue

        # ============================================================
        #  METRICS
        # ============================================================
        try:
            flat_metrics = run_metrics(
                model_name, dataset_name, infer_results, output_dir,
            )
            all_metrics.append(flat_metrics)
        except Exception as e:
            get_logger().error(
                f"[{model_name}/{dataset_name}] METRICS FAILED: {e}"
            )
            import traceback
            traceback.print_exc()

        # ---- cleanup GPU memory ----
        torch.cuda.empty_cache()

    # ================================================================
    #  FINAL SUMMARY
    # ================================================================
    summarize_results(all_metrics)

    print(f"\n[Benchmark] Complete — {len(all_metrics)}/{len(grid)} tasks succeeded.")
    return all_metrics


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GISeg-Bench — Unified Benchmark Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Full benchmark (all models × all datasets)
  python main.py --quick                      # Quick test: 1 model × 1 dataset, 2 epochs
  python main.py --models unet,pranet         # Run specific models on all datasets
  python main.py --datasets kvasir,cvc         # Run all models on specific datasets
  python main.py --skip_train                  # Inference + metrics only (existing checkpoints)
  python main.py --epochs 50                   # Override number of epochs
        """,
    )
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all 24)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all 4)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training; run inference + metrics only")
    parser.add_argument("--skip_infer", action="store_true",
                        help="Skip inference (requires --skip_train; metrics only)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 model, 1 dataset, 2 epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # ---- Parse comma-separated lists ----
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # ---- Seed ----
    seed_all(args.seed)

    # ---- Run ----
    start_time = datetime.now()
    print(f"[Benchmark] Started at {start_time.isoformat()}")
    print(f"[Benchmark] Seed: {args.seed}")

    run_benchmark(
        models=models,
        datasets=datasets,
        skip_train=args.skip_train,
        skip_infer=args.skip_infer,
        epochs=args.epochs,
        quick=args.quick,
    )

    elapsed = datetime.now() - start_time
    print(f"[Benchmark] Finished at {datetime.now().isoformat()}")
    print(f"[Benchmark] Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
