"""
GISeg-Bench  Unified Validator
==============================
Batch-level + epoch-level metric aggregation.

All metrics are computed in a model-output-format-agnostic way:

    - Binary (1-channel logits):    sigmoid → threshold 0.5
    - Multi-class (C-channel):      argmax over dim=1

Supported metrics:  Dice, IoU, HD95, Precision, Recall
"""

import numpy as np
import torch
import torch.nn.functional as F


# ===================================================================
#  Helpers
# ===================================================================

def _to_binary_mask(pred_logits, threshold=0.5):
    """Convert raw model output to binary prediction mask [B, H, W].

    Handles both binary (1-channel) and multi-class (C-channel) logits.
    """
    if pred_logits.shape[1] == 1:
        # binary
        return (torch.sigmoid(pred_logits).squeeze(1) > threshold).long()
    else:
        # multi-class
        return torch.argmax(pred_logits, dim=1).long()


def _ensure_2d(t):
    """Squeeze channel dim if present ([B,1,H,W] → [B,H,W])."""
    if t.ndim == 4 and t.shape[1] == 1:
        t = t.squeeze(1)
    return t.long()


# ===================================================================
#  Per-batch metrics
# ===================================================================

def batch_dice(pred_logits, target, eps=1e-6):
    """Mean Dice over a batch (foreground class only)."""
    pred = _to_binary_mask(pred_logits)
    target = _ensure_2d(target)

    pred_fg = (pred == 1)
    tgt_fg = (target == 1)

    dice_vals = []
    for b in range(pred.size(0)):
        p = pred_fg[b]
        t = tgt_fg[b]
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        dice_vals.append((2.0 * inter + eps) / (denom + eps))

    return torch.stack(dice_vals).mean().item()


def batch_iou(pred_logits, target, eps=1e-6):
    """Mean IoU over a batch (foreground class only)."""
    pred = _to_binary_mask(pred_logits)
    target = _ensure_2d(target)

    pred_fg = (pred == 1)
    tgt_fg = (target == 1)

    iou_vals = []
    for b in range(pred.size(0)):
        p = pred_fg[b]
        t = tgt_fg[b]
        inter = (p & t).sum().float()
        union = p.sum().float() + t.sum().float() - inter
        iou_vals.append((inter + eps) / (union + eps))

    return torch.stack(iou_vals).mean().item()


def batch_precision(pred_logits, target, eps=1e-6):
    pred = _to_binary_mask(pred_logits)
    target = _ensure_2d(target)
    precs = []
    for b in range(pred.size(0)):
        tp = ((pred[b] == 1) & (target[b] == 1)).sum().float()
        fp = ((pred[b] == 1) & (target[b] == 0)).sum().float()
        precs.append((tp + eps) / (tp + fp + eps))
    return torch.stack(precs).mean().item()


def batch_recall(pred_logits, target, eps=1e-6):
    pred = _to_binary_mask(pred_logits)
    target = _ensure_2d(target)
    recs = []
    for b in range(pred.size(0)):
        tp = ((pred[b] == 1) & (target[b] == 1)).sum().float()
        fn = ((pred[b] == 0) & (target[b] == 1)).sum().float()
        recs.append((tp + eps) / (tp + fn + eps))
    return torch.stack(recs).mean().item()


def batch_hd95(pred_logits, target):
    """95th-percentile Hausdorff Distance (fallback: 0.0 if scipy unavailable).

    Computed on CPU numpy arrays.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return 0.0  # graceful fallback

    pred = _to_binary_mask(pred_logits).cpu().numpy()
    target = _ensure_2d(target).cpu().numpy()

    hd_vals = []
    for b in range(pred.shape[0]):
        p = pred[b].astype(bool)
        t = target[b].astype(bool)
        if not p.any() or not t.any():
            hd_vals.append(0.0)
            continue

        dt_p = distance_transform_edt(~p)
        dt_t = distance_transform_edt(~t)

        hd = max(
            np.percentile(dt_p[t], 95),
            np.percentile(dt_t[p], 95),
        )
        hd_vals.append(hd)

    return float(np.mean(hd_vals))


# ===================================================================
#  Aggregated epoch-level
# ===================================================================

_METRIC_FNS = {
    "dice":      batch_dice,
    "iou":       batch_iou,
    "precision": batch_precision,
    "recall":    batch_recall,
    "hd95":      batch_hd95,
}


def compute_all_metrics(pred_logits, target):
    """Return a dict of all per-batch metrics."""
    out = {}
    for name, fn in _METRIC_FNS.items():
        out[name] = fn(pred_logits, target)
    return out


class MetricTracker:
    """Accumulates per-batch metrics and reports epoch averages."""

    def __init__(self, metric_names=None):
        self.names = list(metric_names or _METRIC_FNS.keys())
        self.reset()

    def reset(self):
        self.sum = {n: 0.0 for n in self.names}
        self.count = 0

    def update(self, metrics_dict):
        for n in self.names:
            self.sum[n] += metrics_dict.get(n, 0.0)
        self.count += 1

    def averages(self):
        if self.count == 0:
            return {n: 0.0 for n in self.names}
        return {n: self.sum[n] / self.count for n in self.names}
