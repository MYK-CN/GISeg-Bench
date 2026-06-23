"""
GISeg-Bench  Unified Training Engine
====================================
The single training-loop implementation distilled from all 24
model-specific train.py files.

Handles:
    - CNN / Transformer / SAM-family forward passes
    - Mixed output formats (tensor, dict['out'], tuple)
    - Automatic mask shape normalisation
    - AMP (Automatic Mixed Precision) when requested
    - Multi-loss combination
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .validator import compute_all_metrics


# ===================================================================
#  Output extraction (handles all model families)
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
def _normalize_mask(mask, n_classes):
    """Ensure mask has the correct shape and dtype.

    Input:  [B, 1, H, W] or [B, H, W], float or long
    Output: [B, H, W] long
    """
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    if mask.dtype != torch.long:
        if n_classes == 1:
            mask = (mask > 0.5).long()
        else:
            mask = mask.long()

    return mask


# ===================================================================
#  Training step
# ===================================================================
def train_step(model, images, masks, criteria, optimizer, device,
               n_classes=1, use_amp=False, scaler=None):
    """Single forward + backward + metric step.

    Returns:
        (loss_item, metrics_dict)
    """
    images = images.to(device)
    masks = _normalize_mask(masks.to(device), n_classes)

    # ---- AMP context ----
    with torch.cuda.amp.autocast() if use_amp and device.type == "cuda" else _null_context():
        outputs = _extract_prediction(model(images))

        # Compute total loss from all active criteria
        loss = 0.0
        for name, criterion in criteria.items():
            if name == "ce":
                loss = loss + criterion(outputs, masks)
            elif name in ("bce", "dice"):
                # BCE / Dice expect [B, H, W] targets; binary output [B, 1, H, W]
                target = masks.float().unsqueeze(1)  # [B, 1, H, W]
                if outputs.shape[1] > 1:
                    # multi-class output → extract foreground channel for binary loss
                    logits = outputs[:, 1:2, ...]
                else:
                    logits = outputs
                loss = loss + criterion(logits, target)

    # ---- backward ----
    optimizer.zero_grad()
    if use_amp and device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # ---- metrics ----
    with torch.no_grad():
        metrics = compute_all_metrics(outputs.detach(), masks)

    return loss.item(), metrics


# ===================================================================
#  Validation step
# ===================================================================
@torch.no_grad()
def val_step(model, images, masks, criteria, device, n_classes=1):
    """Single validation forward + metric step."""
    images = images.to(device)
    masks = _normalize_mask(masks.to(device), n_classes)

    outputs = _extract_prediction(model(images))

    loss = 0.0
    for name, criterion in criteria.items():
        if name == "ce":
            loss = loss + criterion(outputs, masks)
        elif name in ("bce", "dice"):
            target = masks.float().unsqueeze(1)
            logits = outputs[:, 1:2, ...] if outputs.shape[1] > 1 else outputs
            loss = loss + criterion(logits, target)

    metrics = compute_all_metrics(outputs.detach(), masks)
    return loss.item(), metrics


# ===================================================================
#  Epoch runners
# ===================================================================
def train_epoch(model, loader, criteria, optimizer, device,
                n_classes=1, use_amp=False, scaler=None):
    """Run one training epoch.  Returns (avg_loss, avg_metrics_dict)."""
    model.train()

    from .validator import MetricTracker
    tracker = MetricTracker()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        images, masks = batch[:2]
        loss_val, met = train_step(
            model, images, masks, criteria, optimizer, device,
            n_classes, use_amp, scaler
        )
        total_loss += loss_val
        tracker.update(met)
        pbar.set_postfix(loss=f"{loss_val:.4f}", dice=f"{met.get('dice', 0):.4f}")

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, tracker.averages()


@torch.no_grad()
def val_epoch(model, loader, criteria, device, n_classes=1):
    """Run one validation epoch.  Returns (avg_loss, avg_metrics_dict)."""
    model.eval()

    from .validator import MetricTracker
    tracker = MetricTracker()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Val")
    for batch in pbar:
        images, masks = batch[:2]
        loss_val, met = val_step(model, images, masks, criteria, device, n_classes)
        total_loss += loss_val
        tracker.update(met)
        pbar.set_postfix(loss=f"{loss_val:.4f}", dice=f"{met.get('dice', 0):.4f}")

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, tracker.averages()


# ===================================================================
#  Utility
# ===================================================================
class _null_context:
    """Context manager that does nothing (fallback when AMP is off)."""
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
