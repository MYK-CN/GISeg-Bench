"""
GISeg-Bench  Unified Optimizer Builder
======================================
Auto-selects the optimal optimizer & hyper-parameters based on
model family, distilled from 24 model training scripts.

Patterns observed across models/:

    CNN (UNet / PraNet / FCN / DeepLabV3):
        Adam  lr=1e-4,  no weight decay

    Transformer (SwinUNet / TransUNet / HiFormer):
        AdamW  lr=1e-4,  weight_decay=1e-5

    Foundation (SAM / MedSAM):
        AdamW  lr=1e-4  (or lower),  weight_decay=0.01
        + selective parameter groups (frozen encoder)
"""

import torch.optim as optim


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------
_OPTIMIZER_MAP = {
    "adam":   optim.Adam,
    "adamw":  optim.AdamW,
    "sgd":    optim.SGD,
}


def build_optimizer(model, cfg, named_params=None):
    """Build optimizer from a config dict or namespace.

    Args:
        model:         nn.Module
        cfg:           dict or argparse.Namespace with keys:
                           optimizer  : "adam" | "adamw" | "sgd" | "auto" (default "auto")
                           lr         : float (default 1e-4)
                           weight_decay : float (default: auto-derived)
        named_params:  optional list of (name, param) tuples for
                       per-group settings (SAM-style). If None,
                       ``model.parameters()`` is used.

    Returns:
        torch.optim.Optimizer
    """
    if isinstance(cfg, dict):
        d = cfg
    else:
        d = vars(cfg)

    opt_name = d.get("optimizer", "auto")
    lr = d.get("lr", 1e-4)
    wd = d.get("weight_decay", None)

    # ---- auto-detect best defaults ----
    if opt_name == "auto":
        opt_name, wd = _auto_optimizer(model, wd)

    opt_cls = _OPTIMIZER_MAP.get(opt_name.lower(), optim.Adam)

    params = named_params if named_params is not None else model.parameters()

    return opt_cls(params, lr=lr, weight_decay=wd if wd is not None else 0.0)


# ---------------------------------------------------------------------------
#  Auto-selection logic
# ---------------------------------------------------------------------------
def _auto_optimizer(model, wd_override):
    """Guess the best (optimizer, weight_decay) from the model's name."""
    model_name = type(model).__name__.lower()

    # --- Foundation / SAM family ---
    if any(k in model_name for k in ("sam", "medsam", "universeg")):
        return "adamw", wd_override if wd_override is not None else 0.01

    # --- Transformer family ---
    if any(k in model_name for k in ("swin", "vit", "transformer",
                                      "hiformer", "h2former", "daeformer",
                                      "transnuseg", "mt_unet")):
        return "adamw", wd_override if wd_override is not None else 1e-5

    # --- CNN / hybrid ---
    return "adam", wd_override if wd_override is not None else 0.0
