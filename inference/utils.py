"""
GISeg-Bench  Inference Utilities
=================================
Lightweight helpers shared across the inference pipeline:

    - Device / dtype conversions
    - Batch collation utilities
    - Image denormalisation (reverse ImageNet or per-image norms)

No file I/O — everything operates on in-memory tensors.
"""

import torch
import numpy as np


# ===================================================================
#  Device helpers
# ===================================================================

def get_device(prefer_gpu=True):
    """Return the best available torch device.

    Args:
        prefer_gpu: if True and CUDA is available, returns ``cuda``.

    Returns:
        ``torch.device``
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_to_device(batch, device):
    """Recursively move a (nested) batch structure to *device*.

    Handles:
        - ``torch.Tensor``          → ``.to(device)``
        - ``list`` / ``tuple``      → new list/tuple with moved elements
        - anything else             → returned as-is
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    return batch


# ===================================================================
#  Format conversion (in-memory only)
# ===================================================================

def to_numpy(tensor):
    """Convert a torch tensor to a numpy array (detaches first)."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(array, device=None, dtype=None):
    """Convert a numpy array to a torch tensor."""
    t = torch.from_numpy(array)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


# ===================================================================
#  Image normalisation helpers
# ===================================================================

# Standard ImageNet constants used by CNN / ViT backbones
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_imagenet(tensor_3hw):
    """Reverse ImageNet normalisation on a [3, H, W] float tensor.

    Used when visual inspection of predictions is required (rare).
    Returns a tensor in [0, 1] range (approx, clipped).
    """
    mean = IMAGENET_MEAN.to(tensor_3hw.device, tensor_3hw.dtype)
    std  = IMAGENET_STD.to(tensor_3hw.device, tensor_3hw.dtype)
    out = tensor_3hw * std + mean
    return out.clamp(0.0, 1.0)


# ===================================================================
#  Batch aggregation helpers
# ===================================================================

def stack_tensors(tensor_list):
    """Stack a list of tensors (possibly of different spatial sizes) into
    a single batch tensor.

    When sizes differ, each tensor is stored in a list (cannot stack).
    Returns either a stacked tensor or the original list.
    """
    if not tensor_list:
        return None
    if all(t.shape == tensor_list[0].shape for t in tensor_list):
        return torch.stack(tensor_list, dim=0)
    return tensor_list  # heterogeneous sizes — keep as list
