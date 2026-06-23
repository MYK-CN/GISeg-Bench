"""
GISeg-Bench  File / Path Utilities
====================================
Centralised path management — no hard-coded paths anywhere else.

Features:
    - ``ensure_dir``:           create directory tree if missing
    - ``output_dir_for``:       canonical output path:  outputs/<experiment>/<model>_<dataset>/
    - ``experiment_dir``:       timestamped experiment root
    - ``find_checkpoint``:      locate best.pth / final.pth in an output dir
    - ``latest_checkpoint``:    pick the most recent checkpoint

Used by: trainer/, inference/, configs/
"""

import os
import glob
from datetime import datetime


# ===================================================================
#  Directory helpers
# ===================================================================

def ensure_dir(path):
    """Create a directory (and parents) if it does not exist.

    Returns *path* unchanged for chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path


def project_root():
    """Return the absolute path to the GISeg-Bench project root.

    Derives from *this file's* location:  ``<root>/utils/file_utils.py``.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===================================================================
#  Output directory conventions
# ===================================================================

def output_dir_for(model_name, dataset_name, base="outputs", tag=None):
    """Build the canonical output directory for a model-dataset pair.

    Pattern::

        <base>/<tag>/<model>_<dataset>/

    Examples::

        output_dir_for("unet", "kvasir")
        # → outputs/unet_kvasir

        output_dir_for("swin_unet", "cvc", base="outputs", tag="exp01")
        # → outputs/exp01/swin_unet_cvc

    Args:
        model_name:   e.g. ``"unet"``, ``"pranet"``.
        dataset_name: e.g. ``"kvasir"``, ``"cvc"``.
        base:         root output directory (default ``"outputs"``).
        tag:          optional sub-folder for grouping runs.

    Returns:
        Absolute path string.
    """
    root = project_root()
    parts = [root, base]
    if tag:
        parts.append(tag)
    parts.append(f"{model_name}_{dataset_name}")
    return os.path.join(*parts)


def experiment_dir(prefix="exp", base="outputs"):
    """Create a timestamped experiment directory.

    Pattern::

        outputs/<prefix>_20260101_120000/

    Returns the created absolute path.
    """
    root = project_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}"
    path = os.path.join(root, base, name)
    return ensure_dir(path)


# ===================================================================
#  Checkpoint discovery
# ===================================================================

def find_checkpoint(model_dir, filename="best.pth"):
    """Locate a specific checkpoint file inside a model directory.

    Args:
        model_dir: path to the model output directory.
        filename:  ``"best.pth"`` or ``"final.pth"``.

    Returns:
        Absolute path to the checkpoint, or **None** if not found.
    """
    path = os.path.join(model_dir, filename)
    if os.path.isfile(path):
        return os.path.abspath(path)

    # also search one level deep (some layouts nest differently)
    for root, _, files in os.walk(model_dir):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))

    return None


def latest_checkpoint(model_dir):
    """Return the path to the most recently modified ``.pth`` file.

    Falls back to ``best.pth``, then ``final.pth``, then any ``.pth``.
    """
    # prefer best.pth
    best = find_checkpoint(model_dir, "best.pth")
    if best:
        return best

    final = find_checkpoint(model_dir, "final.pth")
    if final:
        return final

    # fallback: any .pth sorted by mtime
    candidates = glob.glob(os.path.join(model_dir, "*.pth"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return os.path.abspath(candidates[0])


# ===================================================================
#  Path normalisation
# ===================================================================

def join(*parts):
    """os.path.join shorthand."""
    return os.path.join(*parts)


def resolve(path):
    """Return an absolute, normalised path."""
    return os.path.abspath(os.path.expanduser(path))
