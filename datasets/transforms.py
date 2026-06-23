"""
GISeg-Bench Standard Transforms
===============================
Provides reusable transform pipelines matching the preprocessing
conventions found across all training scripts in this project.

Key patterns observed across models/:
    - ImageNet normalisation (CNN / transformer backbones)
    - Per-image mean-std normalisation (SAM / MedSAM family)
    - Nearest-neighbour resize for masks, bilinear for images
"""

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
#  ImageNet constants (used by CNN backbones & ViT variants)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ===================================================================
#  Functional transforms  (operate on [H, W, C] numpy / PIL)
# ===================================================================

def resize_image(pil_img, size, interpolation=Image.BILINEAR):
    """Resize a PIL image."""
    return pil_img.resize((size, size), interpolation)


def resize_mask(pil_mask, size):
    """Resize a PIL mask with nearest-neighbour (preserves label edges)."""
    return pil_mask.resize((size, size), Image.NEAREST)


def to_tensor_image(pil_img, size=None):
    """PIL RGB -> float32 tensor [3, H, W], optionally resizing first."""
    if size is not None:
        pil_img = resize_image(pil_img, size)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def to_tensor_mask(pil_mask, size=None, binarise=True):
    """PIL grayscale -> long tensor [H, W], optionally resize + binarise."""
    if size is not None:
        pil_mask = resize_mask(pil_mask, size)
    arr = np.array(pil_mask, dtype=np.float32)
    # Only binarise if the mask appears to be a grayscale 0/255 mask
    # (max value > 127); skip for class-labeled masks (values 0..N, N small)
    if binarise and arr.max() > 127:
        arr = (arr > 127).astype(np.float32)
    return torch.from_numpy(arr).long()


def imagenet_normalize(tensor_3hw):
    """Apply ImageNet mean/std to a [3, H, W] float tensor (in-place)."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor_3hw.dtype,
                        device=tensor_3hw.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor_3hw.dtype,
                       device=tensor_3hw.device).view(3, 1, 1)
    tensor_3hw.sub_(mean).div_(std)
    return tensor_3hw


def per_image_normalize(tensor_3hw):
    """Per-image z-score normalisation (used by SAM / MedSAM family).

    Returns (normalised_tensor, mean, std) so the caller can reconstruct
    if needed.
    """
    mean = tensor_3hw.mean(dim=(1, 2), keepdim=True)
    std = tensor_3hw.std(dim=(1, 2), keepdim=True) + 1e-8
    return (tensor_3hw - mean) / std, mean, std


# ===================================================================
#  Augmentations (optional, applied on numpy arrays [H, W, C])
# ===================================================================

def random_hflip(image_np, mask_np, p=0.5):
    """Horizontal flip with probability p."""
    if np.random.random() < p:
        return np.flip(image_np, 1).copy(), np.flip(mask_np, 1).copy()
    return image_np, mask_np


def random_vflip(image_np, mask_np, p=0.5):
    """Vertical flip with probability p."""
    if np.random.random() < p:
        return np.flip(image_np, 0).copy(), np.flip(mask_np, 0).copy()
    return image_np, mask_np


def random_rotate90(image_np, mask_np):
    """Random 0/90/180/270 degree rotation (same for image & mask)."""
    k = np.random.randint(0, 4)
    if k == 0:
        return image_np, mask_np
    return np.rot90(image_np, k).copy(), np.rot90(mask_np, k).copy()


def random_crop(image_np, mask_np, crop_size):
    """Random square crop of ``crop_size`` from image and mask.

    If the image is smaller than ``crop_size``, returns the original.
    """
    h, w = image_np.shape[:2]
    if h < crop_size or w < crop_size:
        return image_np, mask_np
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    img_crop = image_np[top:top + crop_size, left:left + crop_size]
    msk_crop = mask_np[top:top + crop_size, left:left + crop_size]
    return img_crop.copy(), msk_crop.copy()


# ===================================================================
#  Compose-style pipeline (torchvision-free, lightweight)
# ===================================================================

class SegTransform:
    """Callable that bundles image + mask transforms into one step.

    Usage::

        tf = SegTransform(size=256, normalise="imagenet")
        img_t, mask_t = tf(pil_image, pil_mask)
    """

    def __init__(self, size=256, normalise="imagenet", binarise_mask=True):
        """
        Args:
            size:          target square size (int). None = no resize.
            normalise:     "imagenet" | "per_image" | "none"
            binarise_mask: if True, threshold mask values > 0
        """
        self.size = size
        self.normalise = normalise
        self.binarise_mask = binarise_mask

    def __call__(self, image_pil, mask_pil):
        # ---- image ----
        img = to_tensor_image(image_pil, self.size)
        if self.normalise == "imagenet":
            img = imagenet_normalize(img)
        elif self.normalise == "per_image":
            img, _, _ = per_image_normalize(img)

        # ---- mask ----
        mask = to_tensor_mask(mask_pil, self.size, self.binarise_mask)

        return img, mask
