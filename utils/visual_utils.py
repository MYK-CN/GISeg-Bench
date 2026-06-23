"""
GISeg-Bench  Visualisation Utilities
======================================
Lightweight, **standalone** visualisation helpers — for debugging,
paper figures, and UI overlays.  These do NOT participate in the
inference pipeline; they are tools for the developer / analyst.

Features:
    - ``overlay_mask``:   alpha-blend a binary prediction mask onto an RGB image
    - ``compare_gt_pred``: side-by-side or overlay of GT and prediction
    - ``blend_images``:   weighted blend of two RGB images
"""

import numpy as np


# ===================================================================
#  Colour constants
# ===================================================================

# Standard colour palette for segmentation overlay
COLOR_RED    = (255, 0, 0)
COLOR_GREEN  = (0, 255, 0)
COLOR_BLUE   = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_CYAN   = (0, 255, 255)

# Default: GT = green, Prediction = red
DEFAULT_GT_COLOR   = COLOR_GREEN
DEFAULT_PRED_COLOR  = COLOR_RED


# ===================================================================
#  Core overlay
# ===================================================================

def overlay_mask(image, mask, color=COLOR_RED, alpha=0.5):
    """Alpha-blend a binary mask onto an RGB image.

    Args:
        image:  [H, W, 3] uint8 or float32 numpy array (range 0-255 or 0-1).
        mask:   [H, W] binary numpy array (0/1, bool, or float).
        color:  (R, G, B) tuple for the mask overlay.
        alpha:  blend weight — 0 = image only, 1 = mask only.

    Returns:
        [H, W, 3] uint8 numpy array.
    """
    image = _to_uint8(image)
    mask = np.asarray(mask).astype(bool)

    overlay = image.copy()
    color_array = np.array(color, dtype=np.uint8).reshape(1, 1, 3)

    blended = (image * (1.0 - alpha) + color_array * alpha).astype(np.uint8)
    overlay[mask] = blended[mask]
    return overlay


# ===================================================================
#  GT vs Prediction comparison
# ===================================================================

def compare_gt_pred(image, gt_mask, pred_mask,
                    gt_color=DEFAULT_GT_COLOR,
                    pred_color=DEFAULT_PRED_COLOR,
                    alpha=0.4, layout="overlay"):
    """Generate a comparison visualisation of ground-truth and prediction.

    Args:
        image:      [H, W, 3] uint8 RGB image.
        gt_mask:    [H, W] binary ground-truth mask.
        pred_mask:  [H, W] binary prediction mask.
        gt_color:   colour for ground-truth overlay.
        pred_color: colour for prediction overlay.
        alpha:      blend strength.
        layout:     ``"overlay"`` — single image with both masks blended,
                    ``"side_by_side"`` — [left: gt_overlay, right: pred_overlay].

    Returns:
        If layout == ``"overlay"``:  [H, W, 3] uint8 numpy array.
        If layout == ``"side_by_side"``:  [H, 2*W, 3] uint8 numpy array.
    """
    image = _to_uint8(image)

    if layout == "overlay":
        # GT in green, Pred in red — both overlaid
        result = overlay_mask(image, gt_mask, color=gt_color, alpha=alpha)
        result = overlay_mask(result, pred_mask, color=pred_color, alpha=alpha)
        return result

    elif layout == "side_by_side":
        gt_overlay = overlay_mask(image, gt_mask, color=gt_color, alpha=alpha)
        pred_overlay = overlay_mask(image, pred_mask, color=pred_color, alpha=alpha)
        return np.concatenate([gt_overlay, pred_overlay], axis=1)

    else:
        raise ValueError(f"Unknown layout '{layout}'.  Use 'overlay' or 'side_by_side'.")


# ===================================================================
#  Utility blend
# ===================================================================

def blend_images(img_a, img_b, weight=0.5):
    """Weighted blend of two RGB images.

    Args:
        img_a:  [H, W, 3] uint8.
        img_b:  [H, W, 3] uint8.
        weight: blend ratio — 0 = all A, 1 = all B.

    Returns:
        [H, W, 3] uint8 numpy array.
    """
    img_a = _to_uint8(img_a)
    img_b = _to_uint8(img_b)
    return (img_a * (1.0 - weight) + img_b * weight).astype(np.uint8)


# ===================================================================
#  Internal
# ===================================================================

def _to_uint8(image):
    """Convert an image array to uint8 [0, 255]."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        image = (image * 255.0)
    return image.clip(0, 255).astype(np.uint8)
