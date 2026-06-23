"""
GISeg-Bench  Post-processing
=============================
Lightweight, in-memory post-processing for segmentation model outputs.

Pipeline (caller picks which path)::

    logits ──► sigmoid/softmax ──► threshold/argmax ──► resize to orig

Every function works on ``torch.Tensor`` and returns ``torch.Tensor``.
No file I/O — results stay in memory for direct consumption by metrics.
"""

import torch
import torch.nn.functional as F


# ===================================================================
#  Activation
# ===================================================================

def apply_sigmoid(logits):
    """Apply sigmoid to raw logits.

    Args:
        logits: [B, 1, H, W] or [B, H, W]

    Returns:
        probability map with same shape as input, values in [0, 1].
    """
    return torch.sigmoid(logits)


def apply_softmax(logits, dim=1):
    """Apply channel-wise softmax.

    Args:
        logits: [B, C, H, W]
        dim:    channel dimension (default 1)

    Returns:
        class-probability tensor [B, C, H, W], sums to 1 along *dim*.
    """
    return F.softmax(logits, dim=dim)


# ===================================================================
#  Threshold / argmax → hard prediction
# ===================================================================

def threshold_predict(probs, threshold=0.5):
    """Binarise probability maps with a fixed threshold.

    Typical use::

        mask = threshold_predict(apply_sigmoid(logits))

    Args:
        probs:    [B, 1, H, W] or [B, H, W] float tensor
        threshold: decision boundary (default 0.5)

    Returns:
        long tensor of same spatial shape, values {0, 1}.
    """
    # ensure 2D spatial
    if probs.ndim == 4 and probs.shape[1] == 1:
        probs = probs.squeeze(1)
    return (probs > threshold).long()


def argmax_predict(probs):
    """Multi-class prediction via argmax.

    Args:
        probs: [B, C, H, W] softmax probabilities

    Returns:
        long tensor [B, H, W] of class indices.
    """
    return torch.argmax(probs, dim=1).long()


def logits_to_mask(logits, threshold=0.5):
    """Convenience: raw logits → hard binary mask.

    Handles both binary (1-channel) and multi-class (C-channel) logits::

        - 1 channel → sigmoid + threshold
        - C > 1     → softmax + argmax

    Args:
        logits:    [B, C, H, W] tensor
        threshold: used only for binary case

    Returns:
        long tensor [B, H, W] of class indices.
    """
    if logits.shape[1] == 1:
        return threshold_predict(apply_sigmoid(logits), threshold)
    else:
        return argmax_predict(apply_softmax(logits))


# ===================================================================
#  Resize
# ===================================================================

def resize_to_original(prediction, orig_size):
    """Resize a single prediction tensor to its original spatial dimensions.

    Args:
        prediction: [H, W] or [C, H, W] long/float tensor
                    (single sample, no batch dim).
        orig_size:  (height, width) tuple of the **original** image.

    Returns:
        Tensor with spatial size ``orig_size``, same dtype/device as input.
    """
    h, w = orig_size
    cur_h, cur_w = prediction.shape[-2:]

    if (cur_h == h) and (cur_w == w):
        return prediction

    needs_squeeze = False
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)   # [1, H, W]
        needs_squeeze = True
    elif prediction.ndim == 3 and prediction.shape[0] > 4:
        # [C, H, W] with C being class channels — keep
        pass

    # nearest-neighbour for masks to preserve label boundaries
    resized = F.interpolate(
        prediction.unsqueeze(0).float(),
        size=(h, w),
        mode="nearest",
    ).squeeze(0)

    if needs_squeeze:
        resized = resized.squeeze(0)

    return resized.to(prediction.dtype)


def resize_batch_to_original(predictions, orig_sizes):
    """Resize a batch of predictions to their respective original sizes.

    Args:
        predictions:  [B, H, W] long tensor, OR list of tensors.
        orig_sizes:   list of (h, w) tuples, one per sample.

    Returns:
        list of tensors, each at its original spatial size.
    """
    if isinstance(predictions, torch.Tensor):
        # batched — split into per-sample
        out = []
        for i, (h, w) in enumerate(orig_sizes):
            out.append(resize_to_original(predictions[i], (h, w)))
        return out
    else:
        # already a list
        return [
            resize_to_original(p, (h, w))
            for p, (h, w) in zip(predictions, orig_sizes)
        ]
