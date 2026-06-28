"""Checkpoint and timing helpers for training."""

from __future__ import annotations

import math
import shutil
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


def _maybe_update_best_val_checkpoint(
    *,
    output_dir: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_paths: list[str] | None = None,
    val_paths: list[str] | None = None,
    ema_state: dict | None = None,
) -> bool:
    """Update best-val checkpoint when val_loss strictly improves.

    On improvement, copies current best checkpoint to previous_best before
    overwriting best_val_epoch with a full resumable checkpoint payload.
    Returns True when a new best checkpoint is written.
    """
    if not math.isfinite(float(val_loss)):
        return False

    best_path = output_dir / "best_val_epoch.pt"
    previous_best_path = output_dir / "previous_best_val_epoch.pt"
    best_val_loss = float("inf")

    if best_path.exists():
        try:
            existing_best = torch.load(best_path, map_location="cpu")
        except Exception:
            existing_best = None
        if isinstance(existing_best, dict):
            prior_val = existing_best.get("val_loss")
            if prior_val is not None:
                try:
                    prior_val_f = float(prior_val)
                except (TypeError, ValueError):
                    prior_val_f = float("inf")
                if math.isfinite(prior_val_f):
                    best_val_loss = prior_val_f

    if float(val_loss) >= best_val_loss:
        return False

    if best_path.exists():
        shutil.copy2(best_path, previous_best_path)

    _save_checkpoint(
        best_path,
        model,
        optimizer,
        scaler,
        epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        train_paths=train_paths,
        val_paths=val_paths,
        ds_idx=-1,
        ema_state=ema_state,
    )
    return True


def _format_elapsed_dhm(start_time: float) -> str:
    """Format elapsed wall time as DD:HH:MM.m (decimal minutes)."""
    elapsed = max(0.0, time.monotonic() - start_time)
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes_decimal = (elapsed % 3600) / 60.0
    return f"{days:02d}:{hours:02d}:{minutes_decimal:04.1f}"
