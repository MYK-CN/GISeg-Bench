"""
GISeg-Bench  Image Utilities
==============================
Standalone image-processing helpers used by datasets / inference /
visualisation.  All operate on PIL Images or numpy tensors — no
torch dependency for simple I/O transforms.

Functions:
    - Resize (bilinear for images, nearest for masks)
    - Normalisation (ImageNet & per-image)
    - PIL ↔ tensor conversion
    - Mask binarisation
"""

import numpy as np
from PIL import Image
import torch


# -------------------------------------------------------------------
#  ImageNet constants
# -------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===================================================================
#  Resize
# ===================================================================

def resize_image(pil_img, size, interpolation=Image.BILINEAR):
    """Resize a PIL RGB image to ``(size, size)``.

    Args:
        pil_img:       PIL.Image (RGB).
        size:          int → square, or (w, h) tuple.
        interpolation: PIL interpolation mode.
    """
    if isinstance(size, int):
        size = (size, size)
    return pil_img.resize(size, interpolation)


def resize_mask(pil_mask, size):
    """Resize a PIL mask with nearest-neighbour (preserves label boundaries).

    Args:
        pil_mask: PIL.Image (grayscale or palette).
        size:     int → square, or (w, h) tuple.
    """
    if isinstance(size, int):
        size = (size, size)
    return pil_mask.resize(size, Image.NEAREST)


# ===================================================================
#  Normalisation
# ===================================================================

def imagenet_normalize(array_3hw):
    """Apply ImageNet mean/std to a [3, H, W] numpy or torch tensor.

    Args:
        array_3hw: np.ndarray [3, H, W] float (range 0-1), or torch Tensor.

    Returns:
        same type as input, normalised.
    """
    if isinstance(array_3hw, torch.Tensor):
        mean = torch.tensor(IMAGENET_MEAN, dtype=array_3hw.dtype,
                            device=array_3hw.device).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD, dtype=array_3hw.dtype,
                            device=array_3hw.device).view(3, 1, 1)
        return (array_3hw - mean) / std
    else:
        # numpy path
        mean = IMAGENET_MEAN.reshape(3, 1, 1)
        std  = IMAGENET_STD.reshape(3, 1, 1)
        return (array_3hw - mean) / std


def per_image_normalize(array_3hw):
    """Z-score normalisation per image channel.

    Args:
        array_3hw: [3, H, W] numpy or torch tensor.

    Returns:
        (normalised, mean, std) tuple.
    """
    if isinstance(array_3hw, torch.Tensor):
        mean = array_3hw.mean(dim=(1, 2), keepdim=True)
        std  = array_3hw.std(dim=(1, 2), keepdim=True) + 1e-8
        return (array_3hw - mean) / std, mean, std
    else:
        mean = array_3hw.mean(axis=(1, 2), keepdims=True)
        std  = array_3hw.std(axis=(1, 2), keepdims=True) + 1e-8
        return (array_3hw - mean) / std, mean, std


# ===================================================================
#  PIL ↔ tensor
# ===================================================================

def pil_to_tensor_image(pil_img, size=None):
    """PIL RGB → float32 tensor [3, H, W], range [0, 1].

    Args:
        pil_img: PIL RGB Image.
        size:    optional square resize before conversion.
    """
    if size is not None:
        pil_img = resize_image(pil_img, size)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def pil_to_tensor_mask(pil_mask, size=None, binarise=True):
    """PIL grayscale → long tensor [H, W].

    Args:
        pil_mask: PIL grayscale Image.
        size:     optional square resize before conversion.
        binarise: threshold at 127 (only when max > 127).
    """
    if size is not None:
        pil_mask = resize_mask(pil_mask, size)
    arr = np.array(pil_mask, dtype=np.float32)
    if binarise and arr.max() > 127:
        arr = (arr > 127).astype(np.float32)
    return torch.from_numpy(arr).long()


# ===================================================================
#  Mask helpers
# ===================================================================

def binarise_mask(mask_array, threshold=127):
    """Convert a grayscale mask array to binary 0/1.

    Args:
        mask_array: numpy array (uint8 or float).
        threshold:  pixel value above which = foreground.

    Returns:
        uint8 numpy array, values 0 or 1.
    """
    mask = np.asarray(mask_array, dtype=np.float32)
    if mask.max() > 1:
        mask = (mask > threshold).astype(np.uint8)
    else:
        mask = (mask > 0.5).astype(np.uint8)
    return mask


def mask_to_rgb(mask, fg_color=(255, 0, 0), bg_color=(0, 0, 0)):
    """Map a binary mask to an RGB colour image.

    Args:
        mask:      [H, W] numpy array, 0/1.
        fg_color:  (R, G, B) for foreground.
        bg_color:  (R, G, B) for background.

    Returns:
        [H, W, 3] uint8 numpy array.
    """
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask]  = fg_color
    rgb[~mask] = bg_color
    return rgb
