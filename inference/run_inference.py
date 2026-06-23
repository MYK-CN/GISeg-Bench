"""
GISeg-Bench  Inference Entry Point
===================================
Standardised test-set inference + metrics pipeline.

Usage::

    # Basic usage
    python inference/run_inference.py --model unet --dataset kvasir \\
        --data_root data/Kvasir-SEG --checkpoint outputs/trainer/kvasir_unet/best.pth

    # Full options
    python inference/run_inference.py --model pranet --dataset cvc \\
        --data_root data/CVC-ClinicDB --checkpoint outputs/test_gui/kvasir_pranet/best.pth \\
        --image_size 352 --n_classes 1 --batch_size 4

Flow::

    config → model → checkpoint → test dataset → predictor → metrics → report
"""

import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
#  Project root on path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.dataset_zoo import get_dataset
from datasets.transforms import SegTransform

from .loader import build_model, load_checkpoint, list_available_models
from .predictor import run_inference
from .utils import get_device


# ===================================================================
#  Metric computation (in-memory, no file I/O)
# ===================================================================

def _compute_hd95_single(pred, target):
    """95th-percentile Hausdorff distance for a single sample.

    Returns 0.0 if scipy is unavailable, or either prediction/target
    is empty.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return 0.0

    p = pred.cpu().numpy().astype(bool)
    t = target.cpu().numpy().astype(bool)

    if not p.any() or not t.any():
        return 0.0

    dt_p = distance_transform_edt(~p)
    dt_t = distance_transform_edt(~t)

    try:
        return float(max(
            np.percentile(dt_p[t], 95),
            np.percentile(dt_t[p], 95),
        ))
    except (IndexError, ValueError):
        return 0.0


def _compute_metrics_from_results(results, eps=1e-6):
    """Compute segmentation metrics from predictor results dict.

    Operates purely in-memory on the inference output lists.

    Args:
        results: dict with keys ``"predictions"``, ``"targets"``, ``"names"``.
        eps:     numerical stabiliser.

    Returns:
        dict with keys:  dice, iou, precision, recall, hd95, n_samples
    """
    predictions = results["predictions"]   # list of [H, W] long
    targets     = results["targets"]       # list of [H, W] long
    n = len(predictions)

    if n == 0:
        return {}

    dice_vals = []
    iou_vals  = []
    prec_vals = []
    rec_vals  = []
    hd95_vals = []

    for pred, tgt in zip(predictions, targets):
        pred_fg = (pred == 1)
        tgt_fg  = (tgt == 1)

        inter = (pred_fg & tgt_fg).sum().float()
        p_sum = pred_fg.sum().float()
        t_sum = tgt_fg.sum().float()

        # -- Dice --
        denom = p_sum + t_sum
        dice_vals.append(
            ((2.0 * inter + eps) / (denom + eps)).item()
            if denom > 0 else 1.0
        )

        # -- IoU --
        union = p_sum + t_sum - inter
        iou_vals.append(
            ((inter + eps) / (union + eps)).item()
            if union > 0 else 1.0
        )

        # -- Precision --
        prec_vals.append(
            ((inter + eps) / (p_sum + eps)).item()
            if p_sum > 0 else 0.0
        )

        # -- Recall --
        rec_vals.append(
            ((inter + eps) / (t_sum + eps)).item()
            if t_sum > 0 else 0.0
        )

        # -- HD95 --
        hd95_vals.append(_compute_hd95_single(pred, tgt))

    metrics = {
        "dice":      float(np.mean(dice_vals)),
        "iou":       float(np.mean(iou_vals)),
        "precision": float(np.mean(prec_vals)),
        "recall":    float(np.mean(rec_vals)),
        "hd95":      float(np.mean(hd95_vals)),
        "n_samples": n,
    }
    return metrics


# ===================================================================
#  Report formatting
# ===================================================================

def _format_report(metrics, model_name, dataset_name):
    """Pretty-print the metrics table (ASCII-safe)."""
    sep = "=" * 52
    print(sep)
    print("  GISeg-Bench  Inference Report")
    print(sep)
    print(f"  Model   : {model_name}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Samples : {metrics.get('n_samples', 0)}")
    print(sep)
    print(f"  {'Metric':<12s} | {'Value':>10s}")
    print("-" * 52)

    for key, label in [("dice", "Dice"), ("iou", "IoU"),
                        ("precision", "Precision"), ("recall", "Recall"),
                        ("hd95", "HD95")]:
        val = metrics.get(key, 0.0)
        print(f"  {label:<12s} | {val:>10.4f}")

    print(sep)
    print()


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GISeg-Bench — Standardised Test-Set Inference"
    )

    # ---- model ----
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name. Available: {list_available_models()}")

    # ---- data ----
    parser.add_argument("--dataset", type=str, default="kvasir",
                        help="Dataset name: cvc | kvasir | wce | edd")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root folder")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Path to split JSON (auto-detected if omitted)")

    # ---- checkpoint ----
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint (best.pth / final.pth)")

    # ---- inference params ----
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size (default 256)")
    parser.add_argument("--n_classes", type=int, default=1,
                        help="Number of output classes (1=binary)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary segmentation")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference (default: auto GPU)")

    args = parser.parse_args()

    # ---- device ----
    device = torch.device("cpu") if args.cpu else get_device(prefer_gpu=True)
    print(f"[Inference] Device: {device}")

    # ---- build model ----
    print(f"[Inference] Building model: {args.model}")
    model = build_model(
        model_name=args.model,
        n_classes=args.n_classes,
        image_size=args.image_size,
        pretrain=None,  # checkpoint handles weights
    )

    # ---- load checkpoint ----
    model = load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)
    model.eval()

    # ---- test dataset ----
    print(f"[Inference] Loading test dataset: {args.dataset}  (split=test)")
    tf = SegTransform(size=args.image_size, normalise="imagenet")
    test_ds = get_dataset(
        name=args.dataset,
        root=args.data_root,
        split="test",
        split_file=args.split_file,
        transform=tf,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"[Inference] Test samples: {len(test_ds)}")

    # ---- run inference ----
    results = run_inference(
        model=model,
        test_loader=test_loader,
        device=str(device),
        threshold=args.threshold,
        return_logits=True,
    )

    # ---- compute metrics ----
    print("[Inference] Computing metrics ...")
    metrics = _compute_metrics_from_results(results)

    # ---- report ----
    _format_report(metrics, args.model, args.dataset)

    return metrics


if __name__ == "__main__":
    main()
