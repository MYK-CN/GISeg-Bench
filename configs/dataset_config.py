"""
GISeg-Bench  Dataset Configuration
====================================
Central dataset registry — 4 medical segmentation datasets with:

    - Standard names
    - Default relative DATA_ROOT paths
    - Split availability
    - Task type (binary / multi-class)

Usage::

    from configs.dataset_config import DATASET_REGISTRY, get_data_root
    root = get_data_root("kvasir")   # "<project>/data/Kvasir-SEG"
"""

import os


# ===================================================================
#  Project root
# ===================================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===================================================================
#  Dataset registry
# ===================================================================
DATASET_REGISTRY = {
    "cvc": {
        "name":        "CVC-ClinicDB",
        "data_subdir":  "cvc-clinicdb-DatasetNinja",
        "task":         "binary",
        "n_classes":    1,
        "modality":     "colonoscopy",
        "description":  "Colonoscopy polyp segmentation (CVC-ClinicDB)",
        "splits":       ["train", "val", "test"],
    },
    "kvasir": {
        "name":        "Kvasir-SEG",
        "data_subdir":  "Kvasir-SEG",
        "task":         "binary",
        "n_classes":    1,
        "modality":     "colonoscopy",
        "description":  "Gastrointestinal polyp segmentation (Kvasir-SEG)",
        "splits":       ["train", "val", "test"],
    },
    "wce": {
        "name":        "WCEBleedGen",
        "data_subdir":  "WCEBleedGen (updated)",
        "task":         "binary",
        "n_classes":    1,
        "modality":     "wireless-capsule-endoscopy",
        "description":  "WCE bleeding region segmentation",
        "splits":       ["train", "val", "test"],
    },
    "edd": {
        "name":        "EDD2020",
        "data_subdir":  "EDD2020",
        "task":         "multi-class",
        "n_classes":    5,
        "modality":     "endoscopy",
        "description":  "Endoscopic Disease Detection 2020 (multi-class)",
        "splits":       ["train", "val", "test"],
    },
}


# ===================================================================
#  DATA_ROOT auto-resolution
# ===================================================================

def get_data_root(dataset_name):
    """Return the default absolute path to a dataset's root folder.

    Resolution order:
        1. Environment variable ``GISEG_DATA_ROOT`` (if set).
        2. Default location: ``<project_root>/data/<data_subdir>``.

    Args:
        dataset_name: one of ``"cvc"`` | ``"kvasir"`` | ``"wce"`` | ``"edd"``.

    Returns:
        Absolute path string.

    Raises:
        KeyError if the dataset is unknown.
    """
    name = dataset_name.lower()
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    # Env override
    env_root = os.environ.get("GISEG_DATA_ROOT")
    if env_root:
        return os.path.join(env_root, DATASET_REGISTRY[name]["data_subdir"])

    # Default
    return os.path.join(_PROJECT_ROOT, "data", DATASET_REGISTRY[name]["data_subdir"])


def list_datasets(task=None):
    """Return sorted dataset names, optionally filtered by task type.

    Args:
        task: ``"binary"`` | ``"multi-class"`` | None.
    """
    if task is not None:
        return sorted(
            k for k, v in DATASET_REGISTRY.items()
            if v.get("task") == task
        )
    return sorted(DATASET_REGISTRY.keys())


def get_dataset_info(name):
    """Return the metadata dict for a dataset name."""
    return DATASET_REGISTRY.get(name.lower())
