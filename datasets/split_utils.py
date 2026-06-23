"""
GISeg-Bench Split Utilities
===========================
Deterministic train / val / test splitting with fixed random seed
so every model sees exactly the same data division.

Generates and loads .json split files of the form::

    {
        "train": [
            {"image": "img001.jpg", "mask": "img001.png"},
            {"image": "img002.jpg", "mask": "img002.png"}
        ],
        "val":   [ ... ],
        "test":  [ ... ]
    }

Each entry is an {image, mask} pair — the image filename and its
corresponding mask filename (may have different extensions).
"""

import os
import json
import random


SPLIT_SEED = 42  # fixed for reproducibility


def generate_split(image_dir, mask_dir, save_path,
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create a deterministic train/val/test split from image filenames.

    Images and masks are matched by stem (filename without extension),
    which allows different extensions (e.g. image .jpg, mask .png).

    Args:
        image_dir:   path to folder containing images
        mask_dir:    path to folder containing masks
        save_path:   where to write the .json split file
        train_ratio: proportion of data for training
        val_ratio:   proportion for validation
        test_ratio:  proportion for testing
    """
    # --- collect common stems with their actual filenames ---
    img_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    mask_exts = {'.png', '.jpg', '.jpeg', '.gif', '.tif', '.tiff', '.bmp'}

    img_by_stem = {}
    for f in os.listdir(image_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in img_exts:
            img_by_stem[stem] = f

    mask_by_stem = {}
    for f in os.listdir(mask_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in mask_exts:
            mask_by_stem[stem] = f

    common_stems = sorted(set(img_by_stem.keys()) & set(mask_by_stem.keys()))

    if not common_stems:
        raise RuntimeError(
            f"No common image-mask pairs found between\n"
            f"  images: {image_dir}\n  masks:  {mask_dir}"
        )

    # --- deterministic shuffle ---
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(common_stems)

    n = len(common_stems)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    def _pairs(stems):
        return [{"image": img_by_stem[s], "mask": mask_by_stem[s]} for s in stems]

    split_map = {
        "train": _pairs(common_stems[:n_train]),
        "val":   _pairs(common_stems[n_train:n_train + n_val]),
        "test":  _pairs(common_stems[n_train + n_val:]),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(split_map, f, indent=2)

    print(f"[Split] Saved {save_path}  —  "
          f"train={len(split_map['train'])} "
          f"val={len(split_map['val'])} "
          f"test={len(split_map['test'])}")

    return split_map


def load_split(split_path):
    """Load a pre-computed split file.

    Returns:
        dict: {"train": [...], "val": [...], "test": [...]}
        where each entry is {"image": "file.jpg", "mask": "file.png"}.
    """
    with open(split_path, "r") as f:
        return json.load(f)
