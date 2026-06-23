"""
GISeg-Bench  Inference Configuration
======================================
Default settings for the inference / evaluation pipeline.

Covers:
    - Batch size for inference
    - Threshold for binary segmentation
    - Device selection
    - Metrics to run
    - Output file-handling behaviour

Usage::

    from configs.infer_config import INFER_DEFAULTS, get_infer_config
    cfg = get_infer_config(batch_size=8, threshold=0.5)
"""

from .base_config import Config


# ===================================================================
#  Global inference defaults
# ===================================================================
INFER_DEFAULTS = {
    "batch_size":     8,
    "threshold":      0.5,
    "device":        "auto",      # "auto" | "cuda" | "cpu"
    "n_classes":      1,
    "image_size":    256,
    "metrics":        ["dice", "iou", "hd95", "precision", "recall"],
    "return_logits":  True,
    "save_predictions": False,    # ❌ never auto-save to disk
    "dump_metrics_json": False,   # optional: save metrics dict as JSON
}


# ===================================================================
#  Builder
# ===================================================================

def get_infer_config(model_name=None, dataset_name=None, **overrides):
    """Build an inference Config with sensible defaults.

    Resolution order::

        INFER_DEFAULTS → model info → dataset info → overrides

    Args:
        model_name:   e.g. ``"unet"`` (for auto image_size / n_classes).
        dataset_name: e.g. ``"kvasir"`` (for auto n_classes).
        **overrides:  key-value overrides.

    Returns:
        ``Config`` object.
    """
    cfg = Config(INFER_DEFAULTS)

    # ---- model-specific ----
    if model_name is not None:
        from .model_config import MODEL_REGISTRY
        info = MODEL_REGISTRY.get(model_name.lower(), {})
        if "image_size" in info:
            cfg.image_size = info["image_size"]
        if "n_classes" in info:
            cfg.n_classes = info["n_classes"]

    # ---- dataset-specific ----
    if dataset_name is not None:
        from .dataset_config import DATASET_REGISTRY
        dsinfo = DATASET_REGISTRY.get(dataset_name.lower(), {})
        if "n_classes" in dsinfo:
            cfg.n_classes = dsinfo["n_classes"]

    # ---- overrides ----
    cfg.merge(overrides)

    return cfg
