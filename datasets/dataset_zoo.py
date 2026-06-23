"""
GISeg-Bench Dataset Zoo
=======================
Unified dataset registry — every model and trainer loads data through
this single entry point.

Usage::

    from datasets.dataset_zoo import get_dataset

    train_ds = get_dataset("kvasir", root="/data/Kvasir-SEG",
                           split="train", split_file="datasets/splits/kvasir.json")

    test_ds  = get_dataset("cvc", root="/data/CVC-ClinicDB",
                           split="test", split_file="datasets/splits/cvc.json")

Supported ``name`` values: ``"cvc"`` | ``"kvasir"`` | ``"wce"`` | ``"edd"``
"""

import os

# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------
_DATASET_REGISTRY = {}


def _register(name, cls):
    _DATASET_REGISTRY[name.lower()] = cls


# Populate on first import
from .cvc import CVCClinicDB
from .kvasir import KvasirSEG
from .wce import WCEBleedGen
from .edd import EDD2020

_register("cvc", CVCClinicDB)
_register("kvasir", KvasirSEG)
_register("wce", WCEBleedGen)
_register("edd", EDD2020)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def get_dataset(name, root, split="train", split_file=None, transform=None):
    """Return a segmentation Dataset instance.

    Args:
        name:       one of {"cvc", "kvasir", "wce", "edd"}
        root:       path to the dataset root folder
        split:      "train" | "val" | "test" | "all"
        split_file: optional path to a .json split file (reproducible splits)
        transform:  optional transform callable

    Returns:
        torch.utils.data.Dataset instance.

    Raises:
        ValueError if ``name`` is not recognised.
    """
    name = name.lower()
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(_DATASET_REGISTRY.keys())}"
        )

    # resolve split_file default
    if split_file is None:
        default_split = os.path.join(
            os.path.dirname(__file__), "splits", f"{name}.json"
        )
        if os.path.exists(default_split):
            split_file = default_split

    return _DATASET_REGISTRY[name](
        root=root,
        split=split,
        split_file=split_file,
        transform=transform,
    )


def list_datasets():
    """Return the names of all registered datasets."""
    return sorted(_DATASET_REGISTRY.keys())
