"""
GISeg-Bench  Evaluator (Core)
===============================
**Unified evaluation engine** — the single entry point that converts
inference output into paper-grade metrics.

Design rules (strict):
    - Receives inference results **in-memory** (dict of tensors).
    - No file I/O, no visualisation, no training logic.
    - Calls all metric modules and returns a standardised results dict.

Input contract (from :class:`inference.predictor.Predictor`)::

    {
        "predictions":  list of [H, W] long tensors,   # post-processed masks
        "targets":      list of [H, W] long tensors,   # ground-truth masks
        "names":        list of str,                    # sample filenames
        "logits":       list of [C, H, W] tensors (optional),
    }

Output contract::

    {
        "model":            str,
        "dataset":          str,
        "n_samples":        int,

        # Per-sample lists (for box plots / statistical tests)
        "dice_per_sample":  [float, ...],
        "iou_per_sample":   [float, ...],
        "hd95_per_sample":  [float, ...],
        "prec_per_sample":  [float, ...],
        "rec_per_sample":   [float, ...],

        # Aggregated summary
        "Dice":             {"mean": ..., "std": ..., "median": ...},
        "IoU":              {"mean": ..., "std": ..., "median": ...},
        "HD95":             {"mean": ..., "std": ..., "median": ...},
        "Precision":        {"mean": ..., "std": ..., "median": ...},
        "Recall":           {"mean": ..., "std": ..., "median": ...},
    }

Usage::

    from metrics.evaluator import evaluate

    results = predictor.run()          # inference
    report  = evaluate(results, model="UNet", dataset="Kvasir-SEG")
    print(report["Dice"]["mean"])      # 0.912
"""

import math
import logging

from .dice import batch_dice
from .iou import batch_iou
from .hd95 import batch_hd95
from .precision_recall import batch_precision, batch_recall
from .utils import mean_std_median

_logger = logging.getLogger(__name__)


# ===================================================================
#  Core evaluation function
# ===================================================================

def evaluate(inference_results, model="unknown", dataset="unknown",
             n_classes=1, metrics_to_run=None):
    """Run full evaluation on inference output.

    Args:
        inference_results: dict from ``Predictor.run()`` with keys
                           ``"predictions"``, ``"targets"``, ``"names"``.
        model:             model identifier string (for bookkeeping).
        dataset:           dataset identifier string (for bookkeeping).
        n_classes:         number of classes (1=binary). For multi-class
                           the per-class breakdown will be computed.
        metrics_to_run:    optional subset of metric names, e.g.
                           ``["dice", "iou"]``.  If None, runs all 5.

    Returns:
        dict as described in the module docstring.
    """
    # Defensive copy: clone every tensor to prevent any downstream mutation
    # from affecting later metric computations (e.g. in-place ops inside
    # ``ensure_binary`` or distance transforms).
    predictions = [p.clone().detach() for p in inference_results["predictions"]]
    targets     = [t.clone().detach() for t in inference_results["targets"]]

    n = len(predictions)
    if n == 0:
        return _empty_report(model, dataset)

    if n != len(targets):
        _logger.warning(
            "[%s/%s] Mismatch: %d predictions vs %d targets — truncating to min.",
            model, dataset, n, len(targets),
        )
        n = min(n, len(targets))
        predictions = predictions[:n]
        targets = targets[:n]

    # Validate data types before metric computation
    _validate_input_tensors(predictions, targets, model, dataset)

    # Determine which metrics to run
    all_metrics = {"dice", "iou", "hd95", "precision", "recall"}
    if metrics_to_run is not None:
        all_metrics = all_metrics.intersection(metrics_to_run)

    # ------------------------------------------------------------------
    #  Per-sample metric computation
    # ------------------------------------------------------------------
    report = {
        "model":     model,
        "dataset":   dataset,
        "n_samples": n,
    }

    if "dice" in all_metrics:
        dice_list = batch_dice(predictions, targets)
        report["dice_per_sample"] = dice_list
        report["Dice"] = _summarise(dice_list)

    if "iou" in all_metrics:
        iou_list = batch_iou(predictions, targets)
        report["iou_per_sample"] = iou_list
        report["IoU"] = _summarise(iou_list)

    if "hd95" in all_metrics:
        hd95_list = batch_hd95(predictions, targets)
        report["hd95_per_sample"] = hd95_list
        report["HD95"] = _summarise(hd95_list)

    if "precision" in all_metrics:
        prec_list = batch_precision(predictions, targets)
        report["prec_per_sample"] = prec_list
        report["Precision"] = _summarise(prec_list)

    if "recall" in all_metrics:
        rec_list = batch_recall(predictions, targets)
        report["rec_per_sample"] = rec_list
        report["Recall"] = _summarise(rec_list)

    # ------------------------------------------------------------------
    #  Mathematical consistency check (catches anomalies like 5.2)
    # ------------------------------------------------------------------
    if "dice" in all_metrics and "precision" in all_metrics and "recall" in all_metrics:
        _validate_consistency(
            dice_list, prec_list, rec_list,
            report["Dice"]["mean"], report["Precision"]["mean"], report["Recall"]["mean"],
            model, dataset,
        )

    return report


# ===================================================================
#  Multi-class evaluation
# ===================================================================

def evaluate_multiclass(inference_results, model="unknown", dataset="unknown",
                        n_classes=2, class_names=None):
    """Evaluate with per-class breakdown.

    Args:
        inference_results: as above, but predictions/targets may have
                           class indices 0…C-1.
        n_classes:         number of classes (≥ 2).
        class_names:       optional dict mapping index → name, e.g.
                           ``{0: "BG", 1: "polyp", 2: "border"}``.

    Returns:
        dict with extra key ``"per_class"`` containing per-class stats.
    """
    from .dice import multiclass_dice, multiclass_dice_mean
    from .iou import multiclass_iou, multiclass_iou_mean

    predictions = inference_results["predictions"]
    targets     = inference_results["targets"]
    n = len(predictions)

    report = evaluate(inference_results, model=model, dataset=dataset,
                      n_classes=n_classes)

    # ---- per-class per-sample ----
    per_class = {}
    for c in range(1, n_classes):  # skip background
        dice_c = []
        iou_c  = []
        for p, t in zip(predictions, targets):
            p_bin = (p == c).long()
            t_bin = (t == c).long()
            from .dice import dice_score
            from .iou import iou_score
            dice_c.append(dice_score(p_bin, t_bin))
            iou_c.append(iou_score(p_bin, t_bin))

        label = class_names.get(c, f"class_{c}") if class_names else f"class_{c}"
        per_class[label] = {
            "Dice": _summarise(dice_c),
            "IoU":  _summarise(iou_c),
        }

    report["per_class"] = per_class
    return report


# ===================================================================
#  Internal helpers
# ===================================================================

def _validate_input_tensors(predictions, targets, model, dataset):
    """Sanity-check prediction/target tensors before metric computation.

    Detects common pathologies (wrong dtype, non-binary values in binary
    tensors, size mismatches) that could silently corrupt metric results.
    """
    import torch as _torch

    for i, (p, t) in enumerate(zip(predictions, targets)):
        if p.shape != t.shape:
            _logger.warning(
                "[%s/%s] Shape mismatch at index %d: pred %s vs target %s",
                model, dataset, i, tuple(p.shape), tuple(t.shape),
            )

        if p.dtype != _torch.long:
            _logger.debug(
                "[%s/%s] prediction[%d] dtype=%s (expected long) — metric "
                "functions will coerce automatically.",
                model, dataset, i, p.dtype,
            )
        if t.dtype != _torch.long:
            _logger.debug(
                "[%s/%s] target[%d] dtype=%s (expected long) — metric "
                "functions will coerce automatically.",
                model, dataset, i, t.dtype,
            )

        # If tensor is long and values are {0, 1}, it's already binary.
        # If values exceed 1, it is multi-class and will be collapsed by
        # ensure_binary — flag this so the user is aware.
        vals = p.unique()
        if p.dtype == _torch.long and len(vals) > 2:
            _logger.info(
                "[%s/%s] prediction[%d] has %d unique values (multi-class "
                "or post-processing issue) — binary metrics will collapse "
                "foreground to class 1.",
                model, dataset, i, len(vals),
            )


def _validate_consistency(dice_list, prec_list, rec_list,
                         dice_mean, prec_mean, rec_mean,
                         model, dataset, atol=1e-4):
    """Verify that Dice ≈ 2·Precision·Recall / (Precision + Recall).

    If a discrepancy exceeds ``atol``, emit a warning — this is the
    signature of the 5.2 WCE anomaly where per-sample Dice is sensible
    but Precision/Recall report zero due to a mutation or dtype bug.
    """
    n = len(dice_list)
    inconsistent = 0
    for i, (d, p, r) in enumerate(zip(dice_list, prec_list, rec_list)):
        denom = p + r
        expected_dice = (2.0 * p * r) / denom if denom > 0 else 0.0
        if abs(d - expected_dice) > atol:
            inconsistent += 1
            if inconsistent <= 3:  # log first 3 only
                _logger.warning(
                    "[%s/%s] Metric inconsistency at index %d: "
                    "per-sample Dice=%.6f, Precision=%.6f, Recall=%.6f — "
                    "expected Dice from P,R: %.6f (Δ=%.6f).  "
                    "This indicates a data-preprocessing or tensor-mutation bug.",
                    model, dataset, i, d, p, r, expected_dice,
                    abs(d - expected_dice),
                )

    # Check aggregated values too
    denom_agg = prec_mean + rec_mean
    expected_agg = (2.0 * prec_mean * rec_mean) / denom_agg if denom_agg > 0 else 0.0
    if abs(dice_mean - expected_agg) > atol:
        _logger.warning(
            "[%s/%s] Aggregated metric inconsistency: "
            "Dice_mean=%.6f, Precision_mean=%.6f, Recall_mean=%.6f — "
            "expected Dice from P,R: %.6f (Δ=%.6f).  "
            "%d/%d per-sample pairs are inconsistent.",
            model, dataset, dice_mean, prec_mean, rec_mean, expected_agg,
            abs(dice_mean - expected_agg), inconsistent, n,
        )


def _summarise(values):
    """Return a compact stats dict for a list of scalar values."""
    m, s, med = mean_std_median(values)
    return {"mean": m, "std": s, "median": med}


def _empty_report(model, dataset):
    return {
        "model":      model,
        "dataset":    dataset,
        "n_samples":  0,
        "Dice":       {"mean": 0.0, "std": 0.0, "median": 0.0},
        "IoU":        {"mean": 0.0, "std": 0.0, "median": 0.0},
        "HD95":       {"mean": 0.0, "std": 0.0, "median": 0.0},
        "Precision":  {"mean": 0.0, "std": 0.0, "median": 0.0},
        "Recall":     {"mean": 0.0, "std": 0.0, "median": 0.0},
    }


# ===================================================================
#  Convenience: flat dict format (for paper tables)
# ===================================================================

def to_flat_dict(report):
    """Convert the nested report to a flat dict suitable for CSV / pandas.

    Returns keys like ``"Dice_mean"``, ``"IoU_std"``, etc.
    """
    flat = {
        "model":   report["model"],
        "dataset": report["dataset"],
        "n":       report["n_samples"],
    }
    for metric in ["Dice", "IoU", "HD95", "Precision", "Recall"]:
        if metric in report:
            for stat in ["mean", "std", "median"]:
                flat[f"{metric}_{stat}"] = report[metric].get(stat, 0.0)
    return flat
