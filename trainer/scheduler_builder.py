"""
GISeg-Bench  Unified Scheduler Builder
======================================
Provides cosine / step / poly LR schedules, auto-adapting
to each model family's training stability requirements.

Patterns observed across models/:
    - Most CNN models: no scheduler (constant lr)
    - SwinUNet / HiFormer: no scheduler in code, but cosine is standard
    - SAM / MedSAM: often benefit from cosine or poly decay
"""

import torch.optim.lr_scheduler as lr_sched


def build_scheduler(optimizer, cfg, steps_per_epoch=None):
    """Build a LR scheduler from config.

    Args:
        optimizer:        torch.optim.Optimizer
        cfg:              dict or namespace with keys:
                              scheduler    : "cosine" | "step" | "poly" | "none" (default "none")
                              epochs       : total epochs
                              lr           : base lr
                              lr_min       : minimum lr (default lr * 1e-3)
                              warmup_epochs: optional warmup (default 0)
        steps_per_epoch:  int, required by some schedulers

    Returns:
        scheduler or None
    """
    if isinstance(cfg, dict):
        d = cfg
    else:
        d = vars(cfg)

    sched_name = d.get("scheduler", "none")
    if sched_name in ("none", None):
        return None

    epochs = d.get("epochs", 20)
    lr = d.get("lr", 1e-4)
    lr_min = d.get("lr_min", lr * 1e-3)

    total_steps = steps_per_epoch * epochs if steps_per_epoch else epochs

    sched_name = sched_name.lower()

    if sched_name == "cosine":
        return lr_sched.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr_min)

    if sched_name == "step":
        step_size = d.get("step_size", max(1, epochs // 3))
        gamma = d.get("gamma", 0.1)
        return lr_sched.StepLR(
            optimizer, step_size=step_size * (steps_per_epoch or 1), gamma=gamma
        )

    if sched_name == "poly":
        # Poly LR:  lr = init_lr * (1 - iter/max_iter) ** power
        power = d.get("poly_power", 0.9)

        def poly_lambda(cur_iter):
            return max(0.0, (1.0 - cur_iter / max(1, total_steps)) ** power)

        return lr_sched.LambdaLR(optimizer, lr_lambda=poly_lambda)

    raise ValueError(f"Unknown scheduler: {sched_name}")
