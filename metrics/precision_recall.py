"""
GISeg-Bench  Precision & Recall
=================================
Pixel-wise Precision and Recall for binary segmentation.

Definitions (foreground class):

    Precision = TP / (TP + FP)    — fraction of predicted fg that is truly fg
    Recall    = TP / (TP + FN)    — fraction of true fg that is predicted

Supports:
    - Single-sample
    - Batch (list of [H, W] tensors)
"""

from .utils import confusion_for_foreground, ensure_binary, reduce_per_sample


# ===================================================================
#  Precision
# ===================================================================

def precision(pred, target, eps=1e-6):
    """Pixel-wise precision for ONE sample.

    Args:
        pred:   [H, W] long tensor (binary 0/1).
        target: [H, W] long tensor (binary 0/1).
        eps:    numerical stabiliser.

    Returns:
        float in [0, 1].
    """
    pred = ensure_binary(pred)
    target = ensure_binary(target)

    tp, fp, _, _ = confusion_for_foreground(pred, target)
    denom = tp + fp
    if denom == 0:
        return 0.0  # predicted no foreground → precision is undefined, return 0
    return tp / (denom + eps)


def batch_precision(predictions, targets, eps=1e-6):
    """Compute per-sample precision for a list of predictions.

    Returns list of float.
    """
    return reduce_per_sample(predictions, targets,
                             lambda p, t: precision(p, t, eps))


# ===================================================================
#  Recall
# ===================================================================

def recall(pred, target, eps=1e-6):
    """Pixel-wise recall (sensitivity) for ONE sample.

    Args:
        pred:   [H, W] long tensor (binary 0/1).
        target: [H, W] long tensor (binary 0/1).
        eps:    numerical stabiliser.

    Returns:
        float in [0, 1].
    """
    pred = ensure_binary(pred)
    target = ensure_binary(target)

    tp, _, fn, _ = confusion_for_foreground(pred, target)
    denom = tp + fn
    if denom == 0:
        return 0.0  # no true foreground → recall is undefined, return 0
    return tp / (denom + eps)


def batch_recall(predictions, targets, eps=1e-6):
    """Compute per-sample recall for a list of predictions.

    Returns list of float.
    """
    return reduce_per_sample(predictions, targets,
                             lambda p, t: recall(p, t, eps))


# ===================================================================
#  Combined
# ===================================================================

def precision_recall(pred, target, eps=1e-6):
    """Return (precision, recall) for a single sample."""
    return precision(pred, target, eps), recall(pred, target, eps)


def f1_score(pred, target, eps=1e-6):
    """F1 = harmonic mean of precision & recall (= Dice for binary)."""
    p, r = precision(pred, target, eps), recall(pred, target, eps)
    denom = p + r
    if denom == 0:
        return 0.0
    return 2.0 * p * r / (denom + eps)
