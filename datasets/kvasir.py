"""
Kvasir-SEG Dataset
==================
Gastrointestinal polyp segmentation.

Expected directory layout::

    Kvasir-SEG/
    ├── images/
    │   ├── cju0qx73q9q6s0808b3e5b2e4.jpg
    │   ├── ...
    │   └── ...
    ├── masks/
    │   ├── cju0qx73q9q6s0808b3e5b2e4.jpg
    │   ├── ...
    │   └── ...
    ├── test/          (optional – held-out test)
    │   └── ...
    └── masktest/      (optional – held-out test masks)
        └── ...

For ``split="test"`` the loader automatically looks for ``test/`` and
``masktest/`` subdirectories when they exist; otherwise it uses the main
``images/`` and ``masks/`` folders filtered by the split file.
"""

import os
from .base_dataset import BaseSegDataset


class KvasirSEG(BaseSegDataset):
    """Kvasir-SEG polyp segmentation dataset."""

    def _build_index(self):
        # --- test split may live in separate folders ---
        if self.split == "test":
            img_dir = os.path.join(self.root, "test")
            mask_dir = os.path.join(self.root, "masktest")
            if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
                self._scan(img_dir, mask_dir)
                return

        # --- default: images/ + masks/ ---
        img_dir = os.path.join(self.root, "images")
        mask_dir = os.path.join(self.root, "masks")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"images/ not found under {self.root}")
        self._scan(img_dir, mask_dir)

    # ------------------------------------------------------------------
    def _scan(self, img_dir, mask_dir):
        mask_stems = {}
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
