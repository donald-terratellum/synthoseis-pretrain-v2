"""Train-step optimizer and EMA update helpers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.optim as optim


def _maybe_apply_optimizer_step(
    *,
    scaler: Any,
    optimizer: optim.Optimizer,
    model: nn.Module,
    grad_clip_norm: float,
    micro_batches: int,
    accum_steps: int,
    batch_idx: int,
    target_batches: int,
    optimizer_steps: int,
    ema: Any,
    ema_every: int,
) -> tuple[int, int]:
    """Apply optimizer/scaler step when accumulation condition is met.

    Returns updated (micro_batches, optimizer_steps).
    """
    micro_batches += 1
    do_step = (micro_batches >= accum_steps) or (batch_idx == target_batches - 1)
    if not do_step:
        return micro_batches, optimizer_steps

    if scaler is not None:
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    micro_batches = 0
    optimizer_steps += 1

    if ema is not None and optimizer_steps % ema_every == 0:
        ema.update(model)

    return micro_batches, optimizer_steps
