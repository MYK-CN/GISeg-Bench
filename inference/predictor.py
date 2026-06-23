"""
GISeg-Bench  Predictor (Inference Core)
========================================
Runs model forward inference over a **test DataLoader** and returns
predictions **in-memory** — zero disk writes.

Design rules (strict):
    - model.eval()  +  torch.no_grad()
    - batch forward, collect tensors in lists
    - heterogeneous model output → normalised logit tensor
    - ground-truth masks collected alongside predictions for metrics

Returns a ``dict`` ready for ``compute_metrics()``::

    {
        "logits":       list of [C, H, W] tensors (raw model outputs),
        "predictions":  list of [H, W] long tensors  (post-processed),
        "targets":      list of [H, W] long tensors  (ground truth),
        "names":        list of str                   (sample filenames),
    }
"""

import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .utils import get_device, move_to_device
from .postprocess import logits_to_mask


# ===================================================================
#  Output extraction — same logic as trainer/engine.py
# ===================================================================

def _extract_prediction(model_output):
    """Normalise heterogeneous model outputs into a single logit tensor.

    Handles:
        - Tensor [B, C, H, W]                  → as-is
        - dict with 'out' / 'output' key        → dict['out']
        - tuple / list (multi-scale)            → first element
    """
    if isinstance(model_output, dict):
        return model_output.get("out", model_output.get("output", model_output))
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


# ===================================================================
#  Mask normalisation
# ===================================================================

def _normalize_mask(mask):
    """Ensure mask is [B, H, W] long.

    Input may be [B, 1, H, W], float, or long.
    """
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.dtype != torch.long:
        mask = (mask > 0.5).long() if mask.is_floating_point() else mask.long()
    return mask


# ===================================================================
#  Predictor
# ===================================================================

class Predictor:
    """Run model inference over a test DataLoader.

    Usage::

        predictor = Predictor(model, test_loader, device="cuda")
        results = predictor.run(threshold=0.5)
        # results → dict of in-memory tensors

    Parameters
    ----------
    model : nn.Module
        Already loaded model (on the correct device).
    test_loader : DataLoader
        DataLoader yielding ``(images, masks)`` or ``(images, masks, names)``.
    device : str or torch.device
        Device to run inference on.
    return_logits : bool
        If True, include raw logits in the results (default True — useful
        for soft metrics / calibration analysis).
    """

    def __init__(self, model, test_loader, device="cuda", return_logits=True):
        self.model = model
        self.loader = test_loader
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.return_logits = return_logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(self, threshold=0.5):
        """Execute full test-set inference.

        Args:
            threshold: decision boundary for binary segmentation (sigmoid).

        Returns:
            dict with keys:
                - ``"predictions"``: list of [H, W] long tensors
                - ``"targets"``:     list of [H, W] long tensors
                - ``"names"``:       list of str
                - ``"logits"`` (optional): list of [C, H, W] tensors
        """
        self.model.eval()

        all_predictions = []
        all_targets     = []
        all_names       = []
        all_logits      = [] if self.return_logits else None

        pbar = tqdm(self.loader, desc="Inference", unit="batch")
        for batch in pbar:
            # ---- unpack batch (handles 2-tuple or 3-tuple) ----
            images, masks = batch[0], batch[1]
            names = batch[2] if len(batch) > 2 else [f"sample_{i}" for i in range(len(masks))]

            images = move_to_device(images, self.device)

            # ---- forward ----
            outputs = _extract_prediction(self.model(images))

            # ---- move to CPU for post-processing & metric computation ----
            outputs_cpu = outputs.detach().cpu()
            masks_cpu   = _normalize_mask(masks)

            # ---- post-process to hard predictions ----
            preds = logits_to_mask(outputs_cpu, threshold=threshold)

            # ---- collect per sample ----
            for b in range(preds.shape[0]):
                all_predictions.append(preds[b].clone())
                all_targets.append(masks_cpu[b].clone())
                all_names.append(names[b] if isinstance(names, (list, tuple)) else names)

                if self.return_logits:
                    all_logits.append(outputs_cpu[b].clone())

            pbar.set_postfix(samples=len(all_predictions))

        results = {
            "predictions": all_predictions,
            "targets":     all_targets,
            "names":       all_names,
        }
        if self.return_logits:
            results["logits"] = all_logits

        print(f"[Predictor] Done — {len(all_predictions)} samples inferred.")
        return results


# ===================================================================
#  Convenience function
# ===================================================================

@torch.no_grad()
def run_inference(model, test_loader, device="cuda", threshold=0.5,
                  return_logits=True):
    """Functional shortcut for ``Predictor.run()``.

    Returns the same results dict.
    """
    pred = Predictor(model, test_loader, device=device, return_logits=return_logits)
    return pred.run(threshold=threshold)
