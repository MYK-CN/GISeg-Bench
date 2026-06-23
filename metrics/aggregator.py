"""
GISeg-Bench  Results Aggregator
=================================
Aggregate evaluation results across multiple models and datasets into
paper-ready comparison tables.

Typical workflow::

    agg = Aggregator()
    agg.add(evaluate(..., model="UNet",     dataset="Kvasir-SEG"))
    agg.add(evaluate(..., model="SwinUNet", dataset="Kvasir-SEG"))
    agg.add(evaluate(..., model="UNet",     dataset="CVC-ClinicDB"))
    agg.add(evaluate(..., model="SwinUNet", dataset="CVC-ClinicDB"))

    # Print a markdown table grouped by dataset
    agg.print_table(by="dataset", fmt="markdown")

Output example (markdown)::

    | Model     | Dataset      | Dice ± Std  | IoU ± Std   | HD95 ± Std  |
    |-----------|-------------|-------------|-------------|-------------|
    | UNet      | Kvasir-SEG  | 0.912±0.03 | 0.842±0.04 | 3.21±1.05 |
    | SwinUNet  | Kvasir-SEG  | 0.934±0.02 | 0.871±0.03 | 2.87±0.92 |

Supports:
    - Grouping rows by dataset or model
    - Markdown, LaTeX, and plain-text table formats
    - Export to dict / list-of-dicts for programmatic use
"""

import sys
from collections import defaultdict


# ===================================================================
#  Aggregator
# ===================================================================

class Aggregator:
    """Collect ``evaluate()`` reports and produce comparison tables.

    Each entry is a dict conforming to the ``evaluator.evaluate()``
    output contract.
    """

    def __init__(self):
        self._entries = []  # list of report dicts

    # ------------------------------------------------------------------
    #  Data ingestion
    # ------------------------------------------------------------------

    def add(self, report):
        """Add one evaluation report.

        Args:
            report: dict from ``evaluator.evaluate()``.
        """
        self._entries.append(report)

    def add_batch(self, reports):
        """Add a list of reports."""
        self._entries.extend(reports)

    def clear(self):
        """Remove all entries."""
        self._entries.clear()

    def __len__(self):
        return len(self._entries)

    # ------------------------------------------------------------------
    #  Table rendering
    # ------------------------------------------------------------------

    def print_table(self, by="dataset", fmt="markdown",
                    metrics=None, file=None):
        """Print a comparison table to stdout (or *file*).

        Args:
            by:      grouping key — ``"dataset"`` or ``"model"``.
                     - ``"dataset"``: each row is one (model, dataset) pair,
                       grouped by dataset with sub-headers.
                     - ``"model"``:   each row is one (dataset, model) pair,
                       grouped by model.
            fmt:     ``"markdown"`` | ``"latex"`` | ``"plain"``
            metrics: which metrics to display. Default:
                     ``["Dice", "IoU", "HD95", "Precision", "Recall"]``
            file:    optional file-like object to write to (default sys.stdout).
        """
        if file is None:
            file = sys.stdout

        if metrics is None:
            metrics = ["Dice", "IoU", "HD95", "Precision", "Recall"]

        if fmt == "markdown":
            self._render_markdown(by, metrics, file)
        elif fmt == "latex":
            self._render_latex(by, metrics, file)
        else:
            self._render_plain(by, metrics, file)

    def to_dataframe(self):
        """Return results as a list-of-dicts (ready for pandas).

        Example::

            import pandas as pd
            df = pd.DataFrame(agg.to_dataframe())
        """
        rows = []
        for r in self._entries:
            row = {
                "model":   r.get("model", "?"),
                "dataset": r.get("dataset", "?"),
                "n":       r.get("n_samples", 0),
            }
            for metric in ["Dice", "IoU", "HD95", "Precision", "Recall"]:
                if metric in r:
                    row[f"{metric}_mean"]   = r[metric]["mean"]
                    row[f"{metric}_std"]    = r[metric]["std"]
                    row[f"{metric}_median"] = r[metric]["median"]
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    #  Internal renderers
    # ------------------------------------------------------------------

    def _render_markdown(self, by, metrics, file):
        groups = self._group(by)
        for group_key in sorted(groups.keys()):
            entries = groups[group_key]
            print(f"\n### {group_key}\n", file=file)
            self._print_md_table(entries, metrics, by, file)

    def _render_latex(self, by, metrics, file):
        """LaTeX table using ``booktabs`` style, suitable for papers."""
        groups = self._group(by)

        # Column spec
        cols = "l" * 2 + "c" * len(metrics)
        print(r"\begin{table}[htbp]", file=file)
        print(r"  \centering", file=file)
        print(f"  \\begin{{tabular}}{{{cols}}}", file=file)
        print(r"    \toprule", file=file)

        # Header
        header = "    Model & Dataset"
        for m in metrics:
            header += f" & {m}"
        header += r" \\"
        print(header, file=file)
        print(r"    \midrule", file=file)

        for group_key in sorted(groups.keys()):
            entries = groups[group_key]
            for r in entries:
                model   = r.get("model", "?")
                dataset = r.get("dataset", "?")
                line = f"    {model} & {dataset}"
                for m in metrics:
                    stats = r.get(m, {})
                    line += f" & {stats.get('mean', 0):.4f}"
                line += r" \\"
                print(line, file=file)

        print(r"    \bottomrule", file=file)
        print(r"  \end{tabular}", file=file)
        print(r"  \caption{Segmentation results.}", file=file)
        print(r"  \label{tab:seg-results}", file=file)
        print(r"\end{table}", file=file)

    def _render_plain(self, by, metrics, file):
        groups = self._group(by)
        for group_key in sorted(groups.keys()):
            entries = groups[group_key]
            print(f"\n[{group_key}]", file=file)
            self._print_plain_table(entries, metrics, by, file)

    # ------------------------------------------------------------------
    #  Table builders
    # ------------------------------------------------------------------

    def _print_md_table(self, entries, metrics, by, file):
        """Print a single markdown table block."""
        # Column widths
        col_widths = self._compute_widths(entries, metrics, by)

        # Header
        header = "| " + self._pad("Model", col_widths["model"]) + \
                 " | " + self._pad("Dataset", col_widths["dataset"])
        for m in metrics:
            header += " | " + self._pad(f"{m} ± Std", col_widths[m])
        header += " |"
        print(header, file=file)

        # Separator
        sep = "|" + "-" * (col_widths["model"] + 2) + \
              "|" + "-" * (col_widths["dataset"] + 2)
        for m in metrics:
            sep += "|" + "-" * (col_widths[m] + 2)
        sep += "|"
        print(sep, file=file)

        # Rows
        for r in entries:
            model   = r.get("model", "?")
            dataset = r.get("dataset", "?")
            row = f"| {self._pad(model, col_widths['model'])} " \
                  f"| {self._pad(dataset, col_widths['dataset'])}"
            for m in metrics:
                stats = r.get(m, {})
                cell = f"{stats.get('mean', 0):.3f}±{stats.get('std', 0):.3f}"
                row += f" | {self._pad(cell, col_widths[m])}"
            row += " |"
            print(row, file=file)

    def _print_plain_table(self, entries, metrics, by, file):
        """Print a plain-text table with fixed-width columns."""
        lines = []
        # Header
        header = f"{'Model':<12s} {'Dataset':<18s}"
        for m in metrics:
            header += f" {m + ' ± Std':>16s}"
        lines.append(header)
        lines.append("-" * len(header))

        for r in entries:
            model   = r.get("model", "?")
            dataset = r.get("dataset", "?")
            row = f"{model:<12s} {dataset:<18s}"
            for m in metrics:
                stats = r.get(m, {})
                cell = f"{stats.get('mean', 0):.4f}±{stats.get('std', 0):.4f}"
                row += f" {cell:>16s}"
            lines.append(row)

        for line in lines:
            print(line, file=file)
        print(file=file)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _group(self, by):
        """Group entries by *by* key, preserving insertion order."""
        groups = defaultdict(list)
        for r in self._entries:
            groups[r.get(by, "?")].append(r)
        return groups

    def _compute_widths(self, entries, metrics, by):
        """Compute column widths for aligned markdown."""
        widths = {"model": 7, "dataset": 9}
        for m in metrics:
            widths[m] = len(m) + 7  # " ± Std" + 4 digits
        for r in entries:
            widths["model"]   = max(widths["model"],   len(r.get("model", "?")))
            widths["dataset"] = max(widths["dataset"], len(r.get("dataset", "?")))
            for m in metrics:
                stats = r.get(m, {})
                cell = f"{stats.get('mean', 0):.3f}±{stats.get('std', 0):.3f}"
                widths[m] = max(widths[m], len(cell))
        return widths

    @staticmethod
    def _pad(text, width):
        return text.ljust(width)


# ===================================================================
#  Convenience function
# ===================================================================

def aggregate_reports(reports, by="dataset", fmt="markdown", metrics=None,
                       file=None):
    """One-shot: aggregate a list of reports and print a table.

    Args:
        reports: list of dicts from ``evaluate()``.
        by:      grouping key.
        fmt:     ``"markdown"`` | ``"latex"`` | ``"plain"``
        metrics: optional metric subset.
        file:    optional file-like object.
    """
    agg = Aggregator()
    agg.add_batch(reports)
    agg.print_table(by=by, fmt=fmt, metrics=metrics, file=file)
    return agg
