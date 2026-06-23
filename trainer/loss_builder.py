"""
GISeg-Bench  Task-Driven Loss Builder
=====================================
Dynamically assembles loss functions based on segmentation task.

Two canonical families observed across 24 models:

    Binary (n_classes=1):
        BCEWithLogitsLoss  +  DiceLoss(binary)
        Used by: UNet, SwinUNet, H2Former, MedSAM, SAM-Med2D, ConDSeg

    Multi-class (n_classes >= 2):
        CrossEntropyLoss
        Used by: PraNet, FCN, DeepLabV3, TransUNet, HiFormer, DenseNet

Boundary-aware auxiliary loss is available for models that output
edge predictions (TransNuSeg-style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
#  Dice Loss (differentiable)
# ===================================================================
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Args:
        smooth: Laplace smoothing term
        mode:   "sigmoid" — applies sigmoid to logits before dice
                "softmax" — applies softmax then takes channel-1
                "none"    — assumes input is already probabilities
    """

    def __init__(self, smooth=1.0, mode="sigmoid"):
        super().__init__()
        self.smooth = smooth
        self.mode = mode

    def forward(self, logits, targets):
        if self.mode == "sigmoid":
            probs = torch.sigmoid(logits)
        elif self.mode == "softmax":
            probs = F.softmax(logits, dim=1)[:, 1:2, ...]
        else:
            probs = logits

        # flatten spatial dims
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ===================================================================
#  Loss builder
# ===================================================================
def build_loss(cfg):
    """Return (criterion, auxiliary_losses_dict).

    ``cfg`` keys used:

        loss       : "bce_dice" | "ce" | "bce" | "dice"
        n_classes  : int (1 = binary, 2+ = multi-class)
        dice_smooth: float
        class_weights: optional list for CE
    """
    if isinstance(cfg, dict):
        d = cfg
    else:
        d = vars(cfg)

    loss_mode = d.get("loss", "auto")
    n_classes = d.get("n_classes", 1)
    dice_smooth = d.get("dice_smooth", 1.0)

    # ---- auto-detect ----
    if loss_mode == "auto":
        loss_mode = "ce" if n_classes > 1 else "bce_dice"

    criteria = {}
    # Tokenise to avoid substring false positives (e.g. "ce" in "bce_dice")
    tokens = set(loss_mode.split("_"))

    # ---- BCE ----
    if "bce" in tokens:
        criteria["bce"] = nn.BCEWithLogitsLoss()

    # ---- CE (multi-class) ----
    if "ce" in tokens:
        class_weights = d.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criteria["ce"] = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Dice ----
    if "dice" in tokens:
        if n_classes > 1:
            mode = "softmax"
        else:
            mode = "sigmoid"
        criteria["dice"] = DiceLoss(smooth=dice_smooth, mode=mode)

    return criteria
