"""
GISeg-Bench Base Dataset
========================
Abstract base class for all medical image segmentation datasets.
All dataset-specific implementations must inherit from this class.

Unified output contract (every dataset must return):
    image: torch.FloatTensor  [3, H, W]  (RGB, normalized)
    mask:   torch.LongTensor   [H, W]     (class indices, 0=background)
"""

import os
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseSegDataset(Dataset, ABC):
    """Abstract base for all segmentation datasets in GISeg-Bench.

    Subclasses must implement:
        _load_image(path) -> PIL.Image
        _load_mask(path)   -> PIL.Image
    and set:
        self.samples: list of (img_path, mask_path, name) tuples
    """

    def __init__(self, root, split="train", split_file=None, transform=None):
        """
        Args:
            root:       path to dataset root directory
            split:      "train" | "val" | "test" | "all"
            split_file: path to a .json split file; if None, loads all samples
            transform:  optional albumentations / torchvision transform (applied to both image & mask)
        """
        self.root = root
        self.split = split
        self.split_file = split_file
        self.transform = transform
        self.samples = []  # list of (img_path, mask_path, name)

        self._build_index()

        if split_file and split != "all":
            self._apply_split(split_file)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] No samples found in "
                f"root={root}, split={split}"
            )

    # ------------------------------------------------------------------
    #  Subclass interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_index(self):
        """Scan `self.root` and populate `self.samples` with all available
        (img_path, mask_path, name) tuples.
        """
        ...

    def _apply_split(self, split_file):
        """Filter self.samples to keep only the {image, mask} pairs listed in the JSON.

        The JSON file is expected to have the structure::

            {
                "train": [{"image": "img001.jpg", "mask": "img001.png"}, ...],
                "val":   [...],
                "test":  [...]
            }
        """
        import json
        with open(split_file, "r") as f:
            split_map = json.load(f)

        subset = split_map.get(self.split, [])
        # build lookup: image filename -> mask filename
        pair_lookup = {entry["image"]: entry["mask"] for entry in subset}

        filtered = [
            (ip, mp, n) for ip, mp, n in self.samples
            if n in pair_lookup
        ]
        if filtered:
            self.samples = filtered
        else:
            print(
                f"[Warn] Split '{self.split}' matched 0 samples; "
                f"using all {len(self.samples)} samples."
            )

    # ------------------------------------------------------------------
    #  Image / mask I/O (overrideable)
    # ------------------------------------------------------------------
    def _load_image(self, path):
        """Load an image from disk and return a PIL RGB image."""
        return Image.open(path).convert("RGB")

    def _load_mask(self, path):
        """Load a mask from disk and return a PIL grayscale image."""
        return Image.open(path).convert("L")

    # ------------------------------------------------------------------
    #  PyTorch Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, name = self.samples[idx]

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # --- apply transform (e.g. resize + normalize) ---
        if self.transform is not None:
            return self.transform(image, mask)

        # --- fallback: default preprocessing when no transform ---
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)

        # binarise for binary segmentation datasets
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)

        # channel-first + tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask = torch.from_numpy(mask).long()               # [H, W]

        return image, mask
