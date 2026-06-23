"""
WCEBleedGen Dataset
===================
Wireless Capsule Endoscopy bleeding segmentation.

Expected directory layout::

    WCEBleedGen/
    ├── images/
    │   ├── img-001.png
    │   ├── img-002.png
    │   └── ...
    └── masks/
        ├── img-001.png
        ├── img-002.png
        └── ...

Image and mask filenames are identical (stem-matched).
Binary segmentation: 0 = non-bleeding, 1 = bleeding region.
"""

import os
from .base_dataset import BaseSegDataset


class WCEBleedGen(BaseSegDataset):
    """WCEBleedGen bleeding segmentation dataset."""

    def _build_index(self):
        img_dir = os.path.join(self.root, "images")
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


# ---- alias ----
WCE = WCEBleedGen
