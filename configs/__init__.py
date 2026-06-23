"""
GISeg-Bench  Configuration Module
===================================
**Experiment Control Center** — single source of truth for all
model, dataset, training, and inference parameters.

Components:
    - ``base_config``:   Unified Config class with dot-access
    - ``model_config``:  24 model name registry + family grouping
    - ``dataset_config``: 4 medical datasets with DATA_ROOT mapping
    - ``train_config``:  Training hyperparameter presets
    - ``infer_config``:  Inference-specific settings
    - ``config_loader``: Merge + load → unified cfg object

Usage::

    from configs.config_loader import load_config
    cfg = load_config(model="unet", dataset="kvasir")
    print(cfg.model)    # "unet"
    print(cfg.lr)       # 1e-4
"""

from .base_config import Config
from .model_config import MODEL_REGISTRY, MODEL_FAMILIES, list_models
from .dataset_config import DATASET_REGISTRY, get_data_root, list_datasets
from .train_config import TRAIN_DEFAULTS, OPTIMIZER_PRESETS
from .infer_config import INFER_DEFAULTS
from .config_loader import load_config, merge_configs
