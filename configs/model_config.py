"""
GISeg-Bench  Model Configuration
==================================
Central model registry — all 24 segmentation architectures with
metadata:

    - Family grouping (CNN / Transformer / Foundation / Hybrid)
    - Default image size
    - Default number of output channels

Usage::

    from configs.model_config import MODEL_REGISTRY, MODEL_FAMILIES
    print(MODEL_REGISTRY["unet"])       # {"family": "cnn", "image_size": 256, ...}
"""


# ===================================================================
#  Model registry  (24 architectures)
# ===================================================================
MODEL_REGISTRY = {
    # ----------------------------- CNN -----------------------------
    "unet": {
        "family":      "cnn",
        "image_size":   256,
        "n_classes":    1,
        "description": "U-Net (Ronneberger 2015)",
    },
    "pranet": {
        "family":      "cnn",
        "image_size":   352,
        "n_classes":    2,
        "description": "PraNet — Parallel Reverse Attention (Fan 2020)",
    },
    "pranet_v2": {
        "family":      "cnn",
        "image_size":   352,
        "n_classes":    2,
        "description": "PraNet v2 — improved variant",
    },
    "fcn": {
        "family":      "cnn",
        "image_size":   256,
        "n_classes":    2,
        "description": "FCN-ResNet50 (torchvision)",
    },
    "deeplabv3": {
        "family":      "cnn",
        "image_size":   256,
        "n_classes":    2,
        "description": "DeepLabV3-ResNet50 (torchvision)",
    },
    "densenet": {
        "family":      "cnn",
        "image_size":   256,
        "n_classes":    2,
        "description": "DenseNet121-FCN",
    },
    "resnet": {
        "family":      "cnn",
        "image_size":   256,
        "n_classes":    1,
        "description": "ResNet-UNet",
    },
    "ce_net": {
        "family":      "cnn",
        "image_size":   256,
        "n_classes":    1,
        "description": "CE-Net — Context Encoder Network",
    },
    "htc_net": {
        "family":      "cnn",
        "image_size":   224,
        "n_classes":    2,
        "description": "HTC-Net — Hybrid Transformer-CNN",
    },
    "viewpoint_aware_net": {
        "family":      "cnn",
        "image_size":   256,
        "n_classes":    2,
        "description": "Viewpoint-Aware Net",
    },

    # --------------------------- Transformer ----------------------
    "swin_unet": {
        "family":      "transformer",
        "image_size":   224,
        "n_classes":    1,
        "description": "Swin-UNet (Cao 2021)",
    },
    "transunet": {
        "family":      "transformer",
        "image_size":   256,
        "n_classes":    2,
        "description": "TransUNet (Chen 2021)",
    },
    "hiformer": {
        "family":      "transformer",
        "image_size":   224,
        "n_classes":    2,
        "description": "HiFormer — Hierarchical Multi-scale Transformer",
    },
    "h2former": {
        "family":      "transformer",
        "image_size":   224,
        "n_classes":    1,
        "description": "H2Former — Hybrid High-resolution Hierarchical",
    },
    "daeformer": {
        "family":      "transformer",
        "image_size":   224,
        "n_classes":    2,
        "description": "DAE-Former — Dual Attention Enhanced",
    },
    "transnuseg": {
        "family":      "transformer",
        "image_size":   512,
        "n_classes":    2,
        "description": "TransNuSeg — Transformer for Nucleus Segmentation",
    },
    "mt_unet": {
        "family":      "transformer",
        "image_size":   224,
        "n_classes":    1,
        "description": "MT-UNet — Mixed Transformer UNet",
    },

    # --------------------------- Foundation -----------------------
    "medsam": {
        "family":      "foundation",
        "image_size":   1024,
        "n_classes":    1,
        "description": "MedSAM — Medical Segment Anything Model",
    },
    "sam_med2d": {
        "family":      "foundation",
        "image_size":   256,
        "n_classes":    1,
        "description": "SAM-Med2D",
    },
    "universeg": {
        "family":      "foundation",
        "image_size":   256,
        "n_classes":    1,
        "description": "UniverSeg — Universal Segmentation",
    },
    "sam2_unet": {
        "family":      "foundation",
        "image_size":   256,
        "n_classes":    1,
        "description": "SAM2-UNet",
    },
    "scribbleprompt": {
        "family":      "foundation",
        "image_size":   256,
        "n_classes":    1,
        "description": "ScribblePrompt — Weakly-supervised",
    },
    "medical_sam_adapter": {
        "family":      "foundation",
        "image_size":   1024,
        "n_classes":    1,
        "description": "Medical SAM Adapter",
    },

    # --------------------------- Hybrid ---------------------------
    "condseg": {
        "family":      "hybrid",
        "image_size":   256,
        "n_classes":    1,
        "description": "ConDSeg — Conditional Diffusion Segmentation",
    },
}


# ===================================================================
#  Model families  (for auto-selecting optimisers / schedulers)
# ===================================================================
MODEL_FAMILIES = {
    "cnn":         ["unet", "pranet", "pranet_v2", "fcn", "deeplabv3",
                     "densenet", "resnet", "ce_net", "htc_net",
                     "viewpoint_aware_net"],
    "transformer": ["swin_unet", "transunet", "hiformer", "h2former",
                     "daeformer", "transnuseg", "mt_unet"],
    "foundation":  ["medsam", "sam_med2d", "universeg", "sam2_unet",
                     "scribbleprompt", "medical_sam_adapter"],
    "hybrid":      ["condseg"],
}


# ===================================================================
#  Helpers
# ===================================================================

def list_models(family=None):
    """Return sorted model names, optionally filtered by family.

    Args:
        family: ``"cnn"`` | ``"transformer"`` | ``"foundation"`` | ``"hybrid"``
                or None for all.

    Returns:
        list of str.
    """
    if family is not None:
        return MODEL_FAMILIES.get(family, [])
    return sorted(MODEL_REGISTRY.keys())


def get_model_info(name):
    """Return the metadata dict for a model name, or None if unknown."""
    return MODEL_REGISTRY.get(name.lower())
