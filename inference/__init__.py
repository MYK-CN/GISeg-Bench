"""
GISeg-Bench  Inference Module
==============================
Standardised test-set inference — forward-only, no disk writes.

Key components:
    - ``loader``:      model building & checkpoint loading
    - ``predictor``:   batch forward inference core
    - ``postprocess``: sigmoid / softmax / threshold / resize
    - ``utils``:       tensor helpers & device utilities
    - ``run_inference``: CLI entry point + metric aggregation

Usage::

    python -m inference.run_inference --model unet --dataset kvasir \\
        --data_root data/Kvasir-SEG --checkpoint outputs/.../best.pth
"""

from .predictor import Predictor, run_inference
from .loader import build_model, load_checkpoint, list_available_models
from .postprocess import logits_to_mask, apply_sigmoid, apply_softmax
from .utils import get_device
