"""
GISeg-Bench  Training Configuration
=====================================
Default hyperparameter presets distilled from 24 model training scripts.

Covers:
    - Epochs, batch size, learning rate
    - Optimiser selection (with per-family defaults)
    - Scheduler selection
    - Loss function auto-detection
    - AMP / early-stopping / patience

Usage::

    from configs.train_config import TRAIN_DEFAULTS, get_train_config
    cfg = get_train_config("unet")    # returns a Config with sensible defaults
"""

from .base_config import Config


# ===================================================================
#  Global defaults (apply to all models unless overridden)
# ===================================================================
TRAIN_DEFAULTS = {
    "epochs":       100,
    "batch_size":    4,
    "lr":            1e-4,
    "weight_decay":  0.0,
    "optimizer":    "auto",
    "scheduler":    "none",
    "loss":         "auto",
    "image_size":   256,
    "n_classes":     1,
    "use_amp":      False,
    "patience":      15,
    "output_dir":   "outputs/trainer",
    "num_workers":   0,
}


# ===================================================================
#  Per-family presets  (override TRAIN_DEFAULTS)
# ===================================================================
OPTIMIZER_PRESETS = {
    "cnn": {
        "optimizer":    "adam",
        "weight_decay":  0.0,
        "lr":            1e-4,
    },
    "transformer": {
        "optimizer":     "adamw",
        "weight_decay":   1e-5,
        "lr":             1e-4,
    },
    "foundation": {
        "optimizer":     "adamw",
        "weight_decay":   0.01,
        "lr":             1e-4,
    },
    "hybrid": {
        "optimizer":    "adam",
        "weight_decay":  0.0,
        "lr":            1e-4,
    },
}


# ===================================================================
#  Builder
# ===================================================================

def get_train_config(model_name=None, dataset_name=None, family=None, **overrides):
    """Build a training Config with sensible defaults.

    Resolution order::

        TRAIN_DEFAULTS → family presets → model info → overrides

    Args:
        model_name:   e.g. ``"unet"`` (optional — for family auto-detection).
        dataset_name: e.g. ``"kvasir"`` (optional — for n_classes).
        family:       ``"cnn"`` | ``"transformer"`` | ``"foundation"`` | ``"hybrid"``.
                      Auto-detected from *model_name* if omitted.
        **overrides:  any key-value pairs to override the final config.

    Returns:
        ``Config`` object.
    """
    cfg = Config(TRAIN_DEFAULTS)

    # ---- resolve family ----
    if family is None and model_name is not None:
        from .model_config import MODEL_REGISTRY
        info = MODEL_REGISTRY.get(model_name.lower(), {})
        family = info.get("family", "cnn")

    # ---- apply family presets ----
    if family is not None and family in OPTIMIZER_PRESETS:
        cfg.merge(OPTIMIZER_PRESETS[family])

    # ---- apply model-specific info ----
    if model_name is not None:
        from .model_config import MODEL_REGISTRY
        info = MODEL_REGISTRY.get(model_name.lower(), {})
        if "image_size" in info:
            cfg.image_size = info["image_size"]
        if "n_classes" in info:
            cfg.n_classes = info["n_classes"]

    # ---- apply dataset-specific info ----
    if dataset_name is not None:
        from .dataset_config import DATASET_REGISTRY
        dsinfo = DATASET_REGISTRY.get(dataset_name.lower(), {})
        if "n_classes" in dsinfo:
            cfg.n_classes = dsinfo["n_classes"]

    # ---- user overrides (highest priority) ----
    cfg.merge(overrides)

    return cfg
