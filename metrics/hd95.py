"""
GISeg-Bench  Hausdorff Distance 95 (HD95)
===========================================
95th-percentile Hausdorff distance — a critical boundary-sensitive metric
for medical image segmentation.

Definition:
    HD95(P, T) = max( P95_{p in ∂P} d(p, ∂T),  P95_{t in ∂T} d(t, ∂P) )

Where:
    - ∂P, ∂T are the boundary point sets of prediction and target masks
    - d(p, S) is the Euclidean distance from point p to set S
    - P95 is the 95th percentile

Uses ``scipy.ndimage.distance_transform_edt`` for efficient computation.
Falls back to 0.0 when scipy is unavailable.
"""

import logging

import numpy as np

from .utils import ensure_binary, reduce_per_sample

_logger = logging.getLogger(__name__)


# ===================================================================
#  Distance-transform helpers
# ===================================================================

def _surface_distance(pred_bool, target_bool):
    """Compute the two directed 95th-percentile surface distances.

    Returns (p2t_95, t2p_95).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return 0.0, 0.0

    dt_p = distance_transform_edt(~pred_bool)
    dt_t = distance_transform_edt(~target_bool)

    try:
        p2t = np.percentile(dt_p[target_bool], 95) if target_bool.any() else 0.0
        t2p = np.percentile(dt_t[pred_bool], 95) if pred_bool.any() else 0.0
    except (IndexError, ValueError):
        return 0.0, 0.0

    return float(p2t), float(t2p)


# ===================================================================
#  Single-sample
# ===================================================================

def hd95(pred, target):
    """95th-percentile Hausdorff distance for ONE sample.

    Args:
        pred:   [H, W] long tensor (binary 0/1).
        target: [H, W] long tensor (binary 0/1).

    Returns:
        float >= 0 (in pixels).  0.0 if scipy is unavailable or either
        mask is empty.
    """
    pred = ensure_binary(pred)
    target = ensure_binary(target)

    p = pred.cpu().numpy().astype(bool)
    t = target.cpu().numpy().astype(bool)

    if not p.any() or not t.any():
        # HD95 is undefined when either mask is empty.
        # Return NaN (not 0.0) so the anomaly is visible in aggregated stats
        # rather than silently dragging the mean toward zero.
        _logger.debug(
            "HD95 undefined: pred_empty=%s target_empty=%s",
            not p.any(), not t.any(),
        )
        return float("nan")

    p2t, t2p = _surface_distance(p, t)
    return max(p2t, t2p)


# ===================================================================
#  Batch
# ===================================================================

def batch_hd95(predictions, targets):
    """Compute per-sample HD95 for a list of predictions.

    Args:
        predictions: list of [H, W] long tensors.
        targets:     list of [H, W] long tensors.

    Returns:
        list of float, same length as input.
    """
    return reduce_per_sample(predictions, targets, hd95)


# ===================================================================
#  Multi-class HD95
# ===================================================================

def multiclass_hd95(pred, target, n_classes, include_bg=False):
    """Per-class HD95 (foreground classes only by default).

    Args:
        pred:      [H, W] long tensor.
        target:    [H, W] long tensor.
        n_classes: total classes (including background).
        include_bg: whether to include class 0.

    Returns:
        dict mapping class index → float HD95.
    """
    scores = {}
    for c in range(n_classes):
        if not include_bg and c == 0:
            continue
        p_c = (pred == c).long()
        t_c = (target == c).long()
        scores[c] = hd95(p_c, t_c)
    return scores
