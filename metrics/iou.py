"""
GISeg-Bench  IoU (Jaccard) Score
==================================
Intersection-over-Union for binary and multi-class segmentation.

Mathematical definition (foreground class, per-sample):

    IoU = |P ∩ T| / |P ∪ T| = TP / (TP + FP + FN)

Supports:
    - Single-sample
    - Batch (list of [H, W] tensors)
    - Multi-class (per-class IoU)
"""

import numpy as np

from .utils import confusion_for_foreground, ensure_binary, reduce_per_sample


# ===================================================================
#  Single-sample
# ===================================================================

def iou_score(pred, target, eps=1e-6):
    """Intersection-over-Union (Jaccard) for ONE sample.

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
    denom = tp + fp + fn
    if denom == 0:
        return 1.0  # both empty → perfect match
    return tp / (denom + eps)


# ===================================================================
#  Batch
# ===================================================================

def batch_iou(predictions, targets, eps=1e-6):
    """Compute per-sample IoU scores for a list of predictions.

    Args:
        predictions: list of [H, W] long tensors.
        targets:     list of [H, W] long tensors.
        eps:         numerical stabiliser.

    Returns:
        list of float, same length as input.
    """
    return reduce_per_sample(predictions, targets,
                             lambda p, t: iou_score(p, t, eps))


# ===================================================================
#  Multi-class
# ===================================================================

def multiclass_iou(pred, target, n_classes, eps=1e-6):
    """Per-class IoU scores for a single multi-class sample.

    Args:
        pred:      [H, W] long tensor, class indices 0…C-1.
        target:    [H, W] long tensor, class indices 0…C-1.
        n_classes: total number of classes (including background).
        eps:       numerical stabiliser.

    Returns:
        dict mapping class index → float IoU score.
    """
    from .utils import compute_confusion

    cf = compute_confusion(pred, target, n_classes)
    scores = {}
    for c in range(n_classes):
        stats = cf[c]
        denom = stats["tp"] + stats["fp"] + stats["fn"]
        scores[c] = stats["tp"] / (denom + eps) if denom > 0 else 1.0
    return scores


def multiclass_iou_mean(pred, target, n_classes, eps=1e-6, include_bg=False):
    """Mean IoU across classes for one sample (mIoU).

    Args:
        include_bg: if True, include class 0 (background) in the average.
    """
    per_class = multiclass_iou(pred, target, n_classes, eps)
    vals = [per_class[c] for c in range(n_classes) if include_bg or c > 0]
    return float(np.mean(vals)) if vals else 0.0
