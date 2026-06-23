"""
GISeg-Bench  Metrics Utilities
===============================
Shared helpers used by all metric modules:

    - Tensor flatten / type coercion
    - Confusion-matrix primitives (TP / FP / FN / TN)
    - Safe division
    - Batch-level statistics (mean, std, median)

All functions are pure — no file I/O, no global state.
"""

import torch
import numpy as np
from collections import defaultdict


# ===================================================================
#  Tensor helpers
# ===================================================================

def to_numpy(tensor):
    """Safely convert a torch tensor or numpy array to a numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def ensure_binary(tensor):
    """Coerce a tensor to a binary (0/1) long tensor [H, W].

    Handles:
        - float tensors (apply > 0.5 threshold)
        - multi-class tensors (foreground = class > 0)
    """
    if tensor.dtype != torch.long:
        tensor = (tensor > 0.5).long() if tensor.is_floating_point() else tensor.long()
    # Collapse multi-class → binary foreground
    if tensor.max() > 1:
        tensor = (tensor > 0).long()
    return tensor


def flatten_tensor(t):
    """Return a 1-D view of *t* (on the same device)."""
    return t.reshape(-1)


# ===================================================================
#  Confusion-matrix primitives
# ===================================================================

def compute_confusion(pred, target, n_classes=2):
    """Compute per-class TP, FP, FN counts.

    Args:
        pred:   [H, W] long tensor, class indices (0=bg, 1…C-1=fg classes).
        target: [H, W] long tensor, class indices.
        n_classes: total number of classes (including background).

    Returns:
        dict mapping class index → ``{"tp": int, "fp": int, "fn": int}``.
    """
    result = {}
    for c in range(n_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        result[c] = {"tp": tp, "fp": fp, "fn": fn}
    return result


def confusion_for_foreground(pred, target):
    """Return (tp, fp, fn) counts for **foreground class only** (class=1).

    This is the common case for binary medical segmentation.
    """
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target != 1)).sum().item()
    fn = ((pred != 1) & (target == 1)).sum().item()
    tn = ((pred != 1) & (target != 1)).sum().item()
    return tp, fp, fn, tn


# ===================================================================
#  Numerical
# ===================================================================

def safe_divide(num, den, default=0.0):
    """Element-wise division that returns *default* where *den* is zero.

    Works on scalars, tensors, and numpy arrays.
    """
    if isinstance(num, (int, float)):
        return num / den if den != 0 else default
    # tensor / numpy path
    den_safe = den.clone() if hasattr(den, "clone") else den.copy()
    if hasattr(den_safe, "masked_fill_"):
        den_safe = den_safe.masked_fill_(den_safe == 0, 1)
    else:
        den_safe[den_safe == 0] = 1
    return num / den_safe


# ===================================================================
#  Statistics over per-sample metric lists
# ===================================================================

def mean_std(values):
    """Return (mean, std) from a list of scalars.

    Returns (0.0, 0.0) for an empty list.
    """
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def mean_std_median(values):
    """Return (mean, std, median) from a list of scalars.

    NaN values (e.g. from undefined HD95) are excluded from the
    statistics so one bad sample doesn't silently corrupt the aggregate.
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    # Use nan-aware functions so HD95 returning NaN for empty masks
    # does not poison the overall mean/std/median.
    mean_v = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
    std_v = float(np.nanstd(arr, ddof=0)) if np.any(np.isfinite(arr)) else 0.0
    med_v = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else 0.0
    return mean_v, std_v, med_v


def percentiles(values, ps=(25, 50, 75)):
    """Return a dict of percentile values."""
    if not values:
        return {p: 0.0 for p in ps}
    arr = np.array(values, dtype=np.float64)
    return {p: float(np.percentile(arr, p)) for p in ps}


# ===================================================================
#  Batch reduction
# ===================================================================

def reduce_per_sample(predictions, targets, metric_fn):
    """Apply *metric_fn(pred, tgt)* to every (pred, tgt) pair.

    Args:
        predictions: list of [H, W] long tensors.
        targets:     list of [H, W] long tensors.
        metric_fn:   callable(pred_tensor, tgt_tensor) → float.

    Returns:
        list of float scores, one per sample.
    """
    return [metric_fn(p, t) for p, t in zip(predictions, targets)]
