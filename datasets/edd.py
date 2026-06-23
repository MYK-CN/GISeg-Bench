"""
EDD2020 Dataset
===============
Endoscopy Disease Detection 2020 — multi-class disease segmentation.

Expected directory layout::

    EDD2020/
    ├── images/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── masks/
        ├── 001.png
        ├── 002.png
        └── ...

Masks are multi-class (0 = background, 1…N = disease categories).
Unlike the other three datasets, this dataset returns masks with
original class labels intact (no automatic binarisation).

Override ``_load_mask`` and ``__getitem__`` if you need per-class
remapping.
"""

import os
import numpy as np
from PIL import Image
import torch
from .base_dataset import BaseSegDataset


class EDD2020(BaseSegDataset):
    """EDD2020 multi-class endoscopic disease segmentation dataset."""

    def _build_index(self):
        img_dir = os.path.join(self.root, "images")
        # Prefer merged single-file masks; fall back to original multi-TIFF masks
        mask_dir = os.path.join(self.root, "masks_merged")
        if not os.path.isdir(mask_dir):
            mask_dir = os.path.join(self.root, "masks")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"images/ not found under {self.root}")

        mask_stems = {}
        if os.path.isdir(mask_dir):
            for f in os.listdir(mask_dir):
                stem, ext = os.path.splitext(f)
                if ext.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}:
                    mask_stems[stem] = os.path.join(mask_dir, f)

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for f in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in valid_exts:
                continue
            if stem in mask_stems:
                self.samples.append((
                    os.path.join(img_dir, f),
                    mask_stems[stem],
                    f,
                ))

    # ------------------------------------------------------------------
    #  Multi-class mask loading (no automatic binarisation)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path, mask_path, name = self.samples[idx]

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # --- apply transform (e.g. resize + normalize) if provided ---
        if self.transform is not None:
            return self.transform(image, mask)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask)  # keep original class labels

        # some EDD masks encode classes as 0…N grayscale; cast to long
        mask = mask.astype(np.int64)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return image, mask


# ---- alias ----
EDD = EDD2020
