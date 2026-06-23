"""
GISeg-Bench  Training Callbacks
===============================
Lightweight callback system — no heavy framework dependency.

Built-in callbacks:
    - BestModelCheckpoint  (save best model by a monitored metric)
    - EarlyStopping        (stop when metric plateaus)
    - TensorBoardLogger    (scalar logging, optional)
    - ConsoleReporter       (print epoch summaries)
"""

import os
import torch


# ===================================================================
#  Callback base
# ===================================================================
class Callback:
    """Minimal callback interface.  Override any method you need."""

    def on_train_begin(self, trainer):    pass
    def on_epoch_begin(self, trainer):    pass
    def on_epoch_end(self, trainer):      pass
    def on_train_end(self, trainer):      pass


# ===================================================================
#  Built-in callbacks
# ===================================================================

class BestModelCheckpoint(Callback):
    """Save the model whenever a monitored metric improves."""

    def __init__(self, save_dir, monitor="dice", mode="max", filename="best.pth"):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best = float("-inf") if mode == "max" else float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer):
        val = trainer.metrics.get(self.monitor, 0.0)
        improved = (
            (self.mode == "max" and val > self.best) or
            (self.mode == "min" and val < self.best)
        )
        if improved:
            self.best = val
            path = os.path.join(self.save_dir, self.filename)
            torch.save(trainer.model.state_dict(), path)
            trainer.log(f"[Checkpoint] Best {self.monitor}={val:.4f} → {path}")


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor="dice", patience=10, mode="max", min_delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("-inf") if mode == "max" else float("inf")
        self.wait = 0

    def on_epoch_end(self, trainer):
        val = trainer.metrics.get(self.monitor, 0.0)
        improved = (
            (self.mode == "max" and val > self.best + self.min_delta) or
            (self.mode == "min" and val < self.best - self.min_delta)
        )
        if improved:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.log(
                    f"[EarlyStopping] No improvement for {self.patience} epochs; "
                    f"stopping."
                )
                trainer.stop()


class ConsoleReporter(Callback):
    """Print a one-line summary after each epoch."""

    def on_epoch_end(self, trainer):
        parts = [f"Epoch {trainer.current_epoch}/{trainer.epochs}"]
        for k, v in trainer.metrics.items():
            parts.append(f"{k}={v:.4f}")
        trainer.log(" | ".join(parts))


class TensorBoardLogger(Callback):
    """Log scalars to TensorBoard (requires ``pip install tensorboard``)."""

    def __init__(self, log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            self.writer = None
            print("[Warn] tensorboard not installed — logging disabled.")

    def on_epoch_end(self, trainer):
        if self.writer is None:
            return
        step = trainer.current_epoch
        for k, v in trainer.metrics.items():
            self.writer.add_scalar(f"epoch/{k}", v, step)
        self.writer.add_scalar("epoch/loss", trainer.epoch_loss, step)

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()
