"""
GISeg-Bench  Global Utilities
==============================
Shared tooling layer used by all subsystems:

    - trainer/
    - datasets/
    - inference/
    - metrics/

Components:
    - ``logger``:        Unified console + file logging
    - ``metrics_utils``: Dice / IoU helpers, batch statistics
    - ``tensor_utils``:  Tensor conversion, sigmoid / softmax, threshold
    - ``image_utils``:   Image resize, normalise, mask handling
    - ``file_utils``:    Path management, auto-create output directories
    - ``seed``:          Deterministic random seeding
    - ``visual_utils``:  Mask overlay, GT-vs-prediction comparison
"""

from .logger import get_logger, setup_logger
from .seed import seed_all, seed_worker, get_generator
from .file_utils import (
    ensure_dir, output_dir_for, experiment_dir,
    find_checkpoint, latest_checkpoint,
)
from .tensor_utils import (
    to_numpy, to_tensor, apply_sigmoid, apply_softmax,
    threshold_predict, denormalize_imagenet,
)
from .image_utils import (
    resize_image, resize_mask, imagenet_normalize,
    per_image_normalize, pil_to_tensor_image, pil_to_tensor_mask,
)
from .metrics_utils import (
    dice_coef, iou_coef, batch_mean, batch_std,
    confusion_matrix,
)
from .visual_utils import overlay_mask, compare_gt_pred
