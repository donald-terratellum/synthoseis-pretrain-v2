"""Learning-rate scheduler factory helpers."""

from __future__ import annotations

import math

import torch.optim as optim


def _build_lr_scheduler(optimizer: optim.Optimizer, args):
    """Create an epoch-level LR scheduler.

    The default "poly" schedule matches common 3D medical segmentation
    training practice (e.g., nnU-Net style polynomial decay).
    """
    schedule = args.lr_schedule.strip().lower()
    if schedule == "constant":
        return None

    if schedule == "poly":
        total_epochs = max(1, int(args.epochs))
        warmup_epochs = max(0, int(args.lr_warmup_epochs))
        warmup_start = max(0.0, min(1.0, float(args.lr_warmup_start_factor)))
        power = float(args.lr_poly_power)
        if args.lr <= 0:
            min_factor = 0.0
        else:
            min_factor = max(0.0, min(1.0, float(args.lr_min) / float(args.lr)))

        def _poly_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                warmup_progress = (epoch_idx + 1) / warmup_epochs
                return warmup_start + (1.0 - warmup_start) * warmup_progress

            decay_steps = max(1, total_epochs - warmup_epochs - 1)
            progress = min(max((epoch_idx - warmup_epochs) / decay_steps, 0.0), 1.0)
            poly = (1.0 - progress) ** power
            return min_factor + (1.0 - min_factor) * poly

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly_lambda)

    if schedule == "cosine":
        total_epochs = max(1, int(args.epochs))
        warmup_epochs = max(0, int(args.lr_warmup_epochs))
        warmup_start = max(0.0, min(1.0, float(args.lr_warmup_start_factor)))
        if args.lr <= 0:
            min_factor = 0.0
        else:
            min_factor = max(0.0, min(1.0, float(args.lr_min) / float(args.lr)))

        def _cosine_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                warmup_progress = (epoch_idx + 1) / warmup_epochs
                return warmup_start + (1.0 - warmup_start) * warmup_progress

            decay_steps = max(1, total_epochs - warmup_epochs - 1)
            progress = min(max((epoch_idx - warmup_epochs) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lambda)

    raise ValueError(f"Unknown lr schedule: {args.lr_schedule}")
