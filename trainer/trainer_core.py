"""
GISeg-Bench  Trainer Core
=========================
Unified Training Paradigm distilled from 24 model training scripts.

Single ``Trainer`` class drives all models (CNN / Transformer / Foundation).
Exposes a clean programmatic API so the GUI can drive training without
subprocess hacks.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from .engine import train_epoch, val_epoch
from .optimizer_builder import build_optimizer
from .scheduler_builder import build_scheduler
from .loss_builder import build_loss
from .callbacks import ConsoleReporter, BestModelCheckpoint, EarlyStopping


class Trainer:
    """Unified trainer for all segmentation models.

    Usage::

        trainer = Trainer(model, train_loader, cfg)
        trainer.set_callbacks([
            ConsoleReporter(),
            BestModelCheckpoint("outputs/"),
            EarlyStopping(patience=10),
        ])
        trainer.run()
    """

    def __init__(self, model, train_loader, cfg, val_loader=None):
        """
        Args:
            model:        nn.Module (already on device)
            train_loader: DataLoader
            cfg:          dict or argparse.Namespace with training hyperparams
            val_loader:   optional DataLoader for validation
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg if isinstance(cfg, dict) else vars(cfg).copy()
        if not isinstance(cfg, dict):
            self.cfg.update(vars(cfg))

        # ---- resolve device ----
        self.device = next(model.parameters()).device

        # ---- resolve config values ----
        self.epochs = self.cfg.get("epochs", 20)
        self.n_classes = self.cfg.get("n_classes", 1)
        self.use_amp = self.cfg.get("use_amp", False)

        # ---- build components ----
        self.criteria = build_loss(self.cfg)
        self.optimizer = build_optimizer(model, self.cfg)
        steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else None
        self.scheduler = build_scheduler(self.optimizer, self.cfg, steps_per_epoch)

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # ---- state ----
        self.current_epoch = 0
        self.epoch_loss = 0.0
        self.metrics = {}
        self._callbacks = []
        self._stop_flag = False

        # ---- output dir ----
        self.output_dir = self.cfg.get("output_dir", "outputs/trainer")
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------
    def set_callbacks(self, callbacks):
        self._callbacks = list(callbacks)

    def add_callback(self, cb):
        self._callbacks.append(cb)

    def stop(self):
        self._stop_flag = True

    def log(self, msg):
        print(msg, flush=True)

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def run(self):
        self._fire("on_train_begin")

        for epoch in range(1, self.epochs + 1):
            if self._stop_flag:
                break

            self.current_epoch = epoch
            self._fire("on_epoch_begin")

            # ---- train ----
            tr_loss, tr_metrics = train_epoch(
                self.model, self.train_loader, self.criteria,
                self.optimizer, self.device,
                n_classes=self.n_classes,
                use_amp=self.use_amp,
                scaler=self.scaler,
            )
            self.epoch_loss = tr_loss

            # ---- val (if loader provided) ----
            if self.val_loader is not None:
                va_loss, va_metrics = val_epoch(
                    self.model, self.val_loader, self.criteria,
                    self.device, n_classes=self.n_classes,
                )
                # merge with prefix
                self.metrics = {"train_loss": tr_loss}
                self.metrics.update({f"train_{k}": v for k, v in tr_metrics.items()})
                self.metrics.update({"val_loss": va_loss})
                self.metrics.update({f"val_{k}": v for k, v in va_metrics.items()})
            else:
                self.metrics = {"loss": tr_loss}
                self.metrics.update(tr_metrics)

            # ---- lr step ----
            if self.scheduler is not None:
                self.scheduler.step()

            self._fire("on_epoch_end")

        self._fire("on_train_end")

        # always save final
        final_path = os.path.join(self.output_dir, "final.pth")
        torch.save(self.model.state_dict(), final_path)
        self.log(f"[Trainer] Final model saved → {final_path}")

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------
    def _fire(self, hook):
        for cb in self._callbacks:
            getattr(cb, hook, lambda t: None)(self)
