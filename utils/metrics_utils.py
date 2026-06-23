"""
GISeg-Bench  Metrics Utility Helpers
=====================================
Reusable metric primitives shared across the project:

    - Dice / IoU coefficient calculation (numpy & torch)
    - Batch-level statistics (mean, std, median)
    - Confusion-matrix building
    - Tensor flatten for metric computation

These are **standalone** helpers — they complement but do NOT replace
the specialised metric modules in ``metrics/``.
"""

import numpy as np
import torch


# ===================================================================
#  Dice / IoU (numpy — lightweight, no-grad assumed)
# ===================================================================

def dice_coef(pred, target, eps=1e-6):
    """Dice similarity coefficient (foreground class only).

    Args:
        pred:   numpy bool/int array, shape [H, W].
        target: numpy bool/int array, shape [H, W].
        eps:    numerical stabiliser.

    Returns:
        float in [0, 1].
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    inter = (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def iou_coef(pred, target, eps=1e-6):
    """Intersection-over-Union (Jaccard) for foreground class.

    Args:
        pred:   numpy bool/int array, shape [H, W].
        target: numpy bool/int array, shape [H, W].
        eps:    numerical stabiliser.

    Returns:
        float in [0, 1].
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    inter = (pred & target).sum()
    union = pred.sum() + target.sum() - inter
    if union == 0:
        return 1.0
    return (inter + eps) / (union + eps)


# ===================================================================
#  Confusion matrix
# ===================================================================

def confusion_matrix(pred, target, n_classes=2):
    """Compute per-class TP / FP / FN counts.

    Args:
        pred:      numpy array (int), shape [H, W], values 0…C-1.
        target:    numpy array (int), shape [H, W], values 0…C-1.
        n_classes: total classes.

    Returns:
        dict: class_idx → {"tp": int, "fp": int, "fn": int, "tn": int}
    """
    pred = np.asarray(pred).astype(np.int64)
    target = np.asarray(target).astype(np.int64)

    result = {}
    for c in range(n_classes):
        tp = int(((pred == c) & (target == c)).sum())
        fp = int(((pred == c) & (target != c)).sum())
        fn = int(((pred != c) & (target == c)).sum())
        tn = int(((pred != c) & (target != c)).sum())
        result[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return result


# ===================================================================
#  Batch statistics
# ===================================================================

def batch_mean(values):
    """Arithmetic mean of a list of scalars."""
    if not values:
        return 0.0
    return float(np.mean(values))


def batch_std(values, ddof=0):
    """Standard deviation of a list of scalars."""
    if not values:
        return 0.0
    return float(np.std(values, ddof=ddof))


def batch_median(values):
    """Median of a list of scalars."""
    if not values:
        return 0.0
    return float(np.median(values))


def batch_mean_std(values):
    """Return (mean, std) for a list of scalars."""
    return batch_mean(values), batch_std(values)


def batch_summary(values):
    """Return a compact stats dict: mean, std, median, min, max."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0, "n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "n":      len(arr),
    }
