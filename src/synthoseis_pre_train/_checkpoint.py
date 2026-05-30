"""Checkpoint and timing helpers for training."""

from __future__ import annotations

import time
from pathlib import Path

import torch


def _save_checkpoint(path: Path, model, optimizer, scaler, epoch: int,
                     train_loss: float, val_loss: float,
                     train_paths: list[str] | None = None, val_paths: list[str] | None = None,
                     ds_idx: int = -1,
                     ema_state: dict | None = None) -> None:
    """Save a resumable checkpoint.  ds_idx=-1 means end-of-epoch."""
    torch.save({
        "epoch":       epoch,
        "ds_idx":      ds_idx,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "train_loss":  train_loss,
        "val_loss":    val_loss,
        "train_paths": train_paths,
        "val_paths":   val_paths,
        "ema_state":   ema_state,
    }, path)


def _format_elapsed_dhm(start_time: float) -> str:
    """Format elapsed wall time as DD:HH:MM.m (decimal minutes)."""
    elapsed = max(0.0, time.monotonic() - start_time)
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes_decimal = (elapsed % 3600) / 60.0
    return f"{days:02d}:{hours:02d}:{minutes_decimal:04.1f}"
