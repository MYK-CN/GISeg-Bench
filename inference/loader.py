"""
GISeg-Bench  Model Loader
==========================
Responsible for:

1. **Building** a segmentation model from the project's unified registry
   (reuses the exact same builders defined in ``trainer/train.py``).
2. **Loading** a checkpoint (``best.pth`` / ``final.pth``) into the model.

No disk writes — the returned model is ready for ``.eval()`` inference.
"""

import os
import sys

import torch

# ---------------------------------------------------------------------------
#  Ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
#  Import the model registry from the unified trainer
# ---------------------------------------------------------------------------
def _get_model_registry():
    """Lazy-import the model builder dict so we don't trigger side-effects
    at module level (e.g. CUDA init, heavy torchvision imports)."""
    from trainer.train import _MODEL_BUILDERS
    return _MODEL_BUILDERS


# ===================================================================
#  Public API
# ===================================================================

def build_model(model_name, n_classes=1, image_size=256, pretrain=None):
    """Build a segmentation model using the project registry.

    This is the **same** function call that ``trainer/train.py`` uses,
    guaranteeing identical architecture.

    Args:
        model_name:  e.g. ``"unet"``, ``"pranet"``, ``"deeplabv3"``, …
        n_classes:   number of output channels (1 = binary)
        image_size:  input spatial size expected by the model
        pretrain:    optional path to pretrained backbone weights
                     (used by SAM-family models)

    Returns:
        ``nn.Module`` (on CPU).  Caller should move it to the desired device.

    Raises:
        KeyError if *model_name* is not registered.
    """
    registry = _get_model_registry()

    if model_name not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Registered models: {available}"
        )

    cfg = {
        "n_classes":  n_classes,
        "image_size": image_size,
        "pretrain":   pretrain,
    }
    return registry[model_name](cfg)


def load_checkpoint(model, checkpoint_path, device=None):
    """Load a ``.pth`` checkpoint into *model*, handling multiple save formats.

    Supported formats (auto-detected):
        - Raw state_dict:              ``OrderedDict`` at top level
        - Trainer-wrapped:             ``{"state_dict": …, "epoch": …}``
        - Wrapper-wrapped:            ``{"model": …}``

    Args:
        model:          ``nn.Module`` instance (architecture already built).
        checkpoint_path: path to the ``.pth`` file.
        device:         torch device to map weights onto (auto-detected if None).

    Returns:
        *model* with weights loaded (same object, mutated in-place).

    Raises:
        FileNotFoundError if *checkpoint_path* does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ---- unwrap common container keys ----
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    # ---- load with strict=False to tolerate minor key mismatches ----
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[Loader] WARNING: {len(missing)} missing keys (first 5): "
              f"{missing[:5]}")
    if unexpected:
        print(f"[Loader] WARNING: {len(unexpected)} unexpected keys (first 5): "
              f"{unexpected[:5]}")

    print(f"[Loader] Checkpoint loaded: {checkpoint_path}")
    return model


def list_available_models():
    """Return the sorted list of model names known to the registry."""
    return sorted(_get_model_registry().keys())
