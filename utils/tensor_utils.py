"""
GISeg-Bench  Tensor Utilities
===============================
Shared tensor operations used across trainer / inference / metrics:

    - tensor ↔ numpy conversion
    - sigmoid / softmax (unified activation helpers)
    - threshold / argmax prediction
    - ImageNet denormalisation

All functions operate on ``torch.Tensor`` and return ``torch.Tensor``
or ``numpy.ndarray``.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ===================================================================
#  Conversion
# ===================================================================

def to_numpy(tensor):
    """Safely convert a torch tensor to a numpy array (detaches first)."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(array, device=None, dtype=None):
    """Convert a numpy array to a torch tensor, optionally casting."""
    t = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


# ===================================================================
#  Activation
# ===================================================================

def apply_sigmoid(logits):
    """Sigmoid activation on raw logits.

    Args:
        logits: tensor of any shape.

    Returns:
        probability tensor, same shape, values in [0, 1].
    """
    return torch.sigmoid(logits)


def apply_softmax(logits, dim=1):
    """Channel-wise softmax.

    Args:
        logits: [B, C, H, W] or [C, H, W].
        dim:    channel axis.

    Returns:
        probability tensor summing to 1 along *dim*.
    """
    return F.softmax(logits, dim=dim)


# ===================================================================
#  Threshold → hard prediction
# ===================================================================

def threshold_predict(probs, threshold=0.5):
    """Binarise probability maps.

    Args:
        probs: float tensor, values in [0, 1].  Shape [H, W] or [B, H, W].
        threshold: decision boundary.

    Returns:
        long tensor of same shape, values {0, 1}.
    """
    return (probs > threshold).long()


def threshold_batch(probs_batch, threshold=0.5):
    """Binarise a batch of probability maps.

    Args:
        probs_batch: [B, 1, H, W] or [B, H, W] float tensor.
        threshold:   decision boundary.

    Returns:
        [B, H, W] long tensor.
    """
    if probs_batch.ndim == 4 and probs_batch.shape[1] == 1:
        probs_batch = probs_batch.squeeze(1)
    return (probs_batch > threshold).long()


def argmax_predict(probs, dim=1):
    """Multi-class prediction via argmax.

    Args:
        probs: [B, C, H, W] softmax probabilities.
        dim:   class dimension.

    Returns:
        [B, H, W] long tensor of class indices.
    """
    return torch.argmax(probs, dim=dim).long()


def logits_to_mask(logits, threshold=0.5):
    """Convenience: raw logits → hard mask.

    - 1 channel  → sigmoid + threshold
    - C > 1      → softmax + argmax

    Args:
        logits: [B, C, H, W] tensor.

    Returns:
        [B, H, W] long tensor.
    """
    if logits.shape[1] == 1:
        return threshold_predict(apply_sigmoid(logits).squeeze(1), threshold)
    else:
        return argmax_predict(apply_softmax(logits))


# ===================================================================
#  Normalisation helpers
# ===================================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_imagenet(tensor_3hw):
    """Reverse ImageNet normalisation → [0, 1] approximate RGB.

    Args:
        tensor_3hw: [3, H, W] normalised tensor.

    Returns:
        [3, H, W] tensor in approx [0, 1].
    """
    mean = IMAGENET_MEAN.to(tensor_3hw.device, tensor_3hw.dtype)
    std  = IMAGENET_STD.to(tensor_3hw.device, tensor_3hw.dtype)
    return (tensor_3hw * std + mean).clamp(0.0, 1.0)


# ===================================================================
#  Mask normalisation
# ===================================================================

def normalize_mask(mask):
    """Coerce any mask tensor to [B, H, W] long.

    Handles [B, 1, H, W], [B, H, W] float, multi-class.
    """
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.dtype != torch.long:
        if mask.is_floating_point():
            mask = (mask > 0.5).long()
        else:
            mask = mask.long()
    return mask
