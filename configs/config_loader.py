"""
GISeg-Bench  Configuration Loader (Core)
==========================================
**The single entry point** for building a unified experiment configuration.

Merges::

    base_config + model_config + dataset_config + train_config + infer_config

into one ``Config`` object that can drive **any** subsystem:

    - trainer/
    - inference/
    - metrics/

Usage::

    from configs.config_loader import load_config

    # Minimal — auto-resolves everything
    cfg = load_config(model="unet", dataset="kvasir")

    # Training mode
    cfg = load_config(model="swin_unet", dataset="cvc", mode="train",
                      epochs=200, lr=5e-5)

    # Inference mode
    cfg = load_config(model="unet", dataset="kvasir", mode="infer",
                      checkpoint="outputs/unet_kvasir/best.pth")

    # Access
    print(cfg.model)       # "unet"
    print(cfg.image_size)  # 256
    print(cfg.optimizer)   # "adam"
"""

import os
import sys

from .base_config import Config
from .train_config import TRAIN_DEFAULTS, get_train_config
from .infer_config import INFER_DEFAULTS, get_infer_config
from .model_config import MODEL_REGISTRY, get_model_info
from .dataset_config import DATASET_REGISTRY, get_data_root


# ===================================================================
#  Master loader
# ===================================================================

def load_config(model=None, dataset=None, mode="train", **overrides):
    """Build a unified experiment Config.

    Args:
        model:    model name, e.g. ``"unet"``, ``"pranet"``, …
                  Must be a key in ``MODEL_REGISTRY``.
        dataset:  dataset name, e.g. ``"kvasir"``, ``"cvc"``, …
                  Must be a key in ``DATASET_REGISTRY``.
        mode:     ``"train"`` — training hyperparameters,
                  ``"infer"`` — inference settings.
        **overrides:  any extra key-value pairs to override the defaults
                  (e.g. ``epochs=200``, ``lr=1e-3``, ``batch_size=8``).

    Returns:
        ``Config`` with all parameters resolved.

    Raises:
        ValueError if *model* or *dataset* is not recognised.
    """
    # ---- validate model ----
    if model is not None and model.lower() not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model}'.  Available: {available}"
        )

    # ---- validate dataset ----
    if dataset is not None and dataset.lower() not in DATASET_REGISTRY:
        available = sorted(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset}'.  Available: {available}"
        )

    # ---- resolve model family ----
    family = None
    if model is not None:
        family = MODEL_REGISTRY[model.lower()]["family"]

    # ---- build base config according to mode ----
    if mode == "infer":
        cfg = get_infer_config(model_name=model, dataset_name=dataset)
    else:
        cfg = get_train_config(model_name=model, dataset_name=dataset,
                               family=family)

    # ---- inject model & dataset identifiers ----
    cfg.model   = model.lower() if model else "unknown"
    cfg.dataset = dataset.lower() if dataset else "unknown"
    cfg.mode    = mode
    cfg.family  = family or "unknown"

    # ---- resolve data_root ----
    if dataset is not None:
        cfg.data_root = get_data_root(dataset)
        cfg.split_file = cfg.get("split_file", None)  # keep if set

    # ---- resolve output_dir ----
    if "output_dir" not in overrides and model and dataset:
        from utils.file_utils import output_dir_for
        cfg.output_dir = output_dir_for(model, dataset, base="outputs")

    # ---- apply user overrides (highest priority) ----
    cfg.merge(overrides)

    return cfg


# ===================================================================
#  Merge helper (standalone)
# ===================================================================

def merge_configs(*configs):
    """Merge multiple Config objects / dicts into one, left-to-right.

    Rightmost takes precedence (like ``dict.update``).
    """
    merged = Config()
    for c in configs:
        merged.merge(c)
    return merged


# ===================================================================
#  CLI convenience  (python -m configs.config_loader --model unet)
# ===================================================================

def _cli():
    """Quick introspection: ``python -m configs.config_loader --model unet``."""
    import argparse
    parser = argparse.ArgumentParser("Config Loader — quick introspection")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--dataset", type=str, default="kvasir")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "infer"])
    parser.add_argument("--key", type=str, default=None,
                        help="Print only this key (e.g. 'image_size')")
    args = parser.parse_args()

    cfg = load_config(model=args.model, dataset=args.dataset, mode=args.mode)

    if args.key:
        print(cfg.get(args.key, "<not set>"))
    else:
        print(f"=== Config: {args.model} / {args.dataset} ({args.mode}) ===")
        for k, v in sorted(cfg.items()):
            print(f"  {k:<20s} = {v!r}")


if __name__ == "__main__":
    _cli()
