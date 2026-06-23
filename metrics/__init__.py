"""
GISeg-Bench  Metrics Module
=============================
Standardised evaluation engine — consumes in-memory inference results,
produces paper-grade metrics with zero disk I/O.

Key components:
    - ``evaluator``:        Unified evaluation entry point
    - ``dice``:             Dice coefficient
    - ``iou``:              IoU / Jaccard
    - ``hd95``:             Hausdorff Distance 95
    - ``precision_recall``: Precision & Recall
    - ``utils``:            Tensor helpers & statistics
    - ``aggregator``:       Multi-model / multi-dataset result aggregation

Usage::

    from metrics.evaluator import evaluate
    from metrics.aggregator import Aggregator

    results = predictor.run()              # inference in-memory
    report  = evaluate(results, model="UNet", dataset="Kvasir-SEG")
    print(report["Dice"]["mean"])          # 0.912
"""

from .evaluator import evaluate, evaluate_multiclass, to_flat_dict
from .aggregator import Aggregator, aggregate_reports
from .dice import dice_score, batch_dice
from .iou import iou_score, batch_iou
from .hd95 import hd95, batch_hd95
from .precision_recall import precision, recall, batch_precision, batch_recall
