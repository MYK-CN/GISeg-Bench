"""
GISeg-Bench  Dice Coefficient
===============================
Dice / F1 score for binary and multi-class segmentation.

Mathematical definition (foreground class, per-sample):

    Dice = 2 * |P ∩ T| / (|P| + |T|)

where P = predicted foreground mask, T = target foreground mask.

Supports:
    - Single-sample
    - Batch (list of [H, W] tensors)
    - Multi-class (per-class Dice)
"""

import torch
import numpy as np

from .utils import confusion_for_foreground, ensure_binary, reduce_per_sample


# ===================================================================
#  Single-sample
# ===================================================================

def dice_score(pred, target, eps=1e-6):
    """Dice similarity coefficient for ONE sample.

    Args:
        pred:   [H, W] long tensor (binary 0/1).
        target: [H, W] long tensor (binary 0/1).
        eps:    numerical stabiliser.

    Returns:
        float in [0, 1].
    """
    pred = ensure_binary(pred)
    target = ensure_binary(target)

    tp, fp, fn, _ = confusion_for_foreground(pred, target)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0  # both empty → perfect match by convention
    return (2.0 * tp) / (denom + eps)


# ===================================================================
#  Batch
# ===================================================================

def batch_dice(predictions, targets, eps=1e-6):
    """Compute per-sample Dice scores for a list of predictions.

    Args:
        predictions: list of [H, W] long tensors.
        targets:     list of [H, W] long tensors.
        eps:         numerical stabiliser.

    Returns:
        list of float, same length as input.
    """
    return reduce_per_sample(predictions, targets,
                             lambda p, t: dice_score(p, t, eps))


# ===================================================================
#  Multi-class
# ===================================================================

def multiclass_dice(pred, target, n_classes, eps=1e-6):
    """Per-class Dice scores for a single multi-class sample.

    Args:
        pred:      [H, W] long tensor, class indices 0…C-1.
        target:    [H, W] long tensor, class indices 0…C-1.
        n_classes: total number of classes (including background).
        eps:       numerical stabiliser.

    Returns:
        dict mapping class index → float Dice score.
    """
    from .utils import compute_confusion

    cf = compute_confusion(pred, target, n_classes)
    scores = {}
    for c in range(n_classes):
        stats = cf[c]
        denom = 2 * stats["tp"] + stats["fp"] + stats["fn"]
        scores[c] = (2.0 * stats["tp"]) / (denom + eps) if denom > 0 else 1.0
    return scores


def multiclass_dice_mean(pred, target, n_classes, eps=1e-6, include_bg=False):
    """Mean Dice across classes for one sample.

    Args:
        include_bg: if True, include class 0 (background) in the average.
    """
    per_class = multiclass_dice(pred, target, n_classes, eps)
    vals = [per_class[c] for c in range(n_classes) if include_bg or c > 0]
    return float(np.mean(vals)) if vals else 0.0
