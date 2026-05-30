"""Train-loop progress reporting and partial checkpoint helpers."""

from __future__ import annotations

import time
from pathlib import Path

from synthoseis_pre_train._checkpoint import _format_elapsed_dhm, _save_checkpoint


def _log_train_progress_and_maybe_checkpoint(
    *,
    batch_idx: int,
    target_batches: int,
    batch_loss: float,
    nz_pct_sum: float,
    total_batches: int,
    window_start: float,
    thermal_guard,
    output_dir: Path | None,
    model,
    optimizer,
    scaler,
    epoch: int,
    total_loss: float,
    train_paths: list[str] | None,
    val_paths: list[str] | None,
    ema_state: dict | None,
) -> float:
    """Log periodic train progress and write partial checkpoint when enabled.

    Returns updated window_start timestamp.
    """
    avg_pct = nz_pct_sum / max(total_batches, 1)
    elapsed_dhm = _format_elapsed_dhm(window_start)
    window_start = time.monotonic()

    temp_str = ""
    if thermal_guard is not None and thermal_guard.last_temp_c is not None:
        temp_str = f", CPU temp: {thermal_guard.last_temp_c:.1f}C"
    elif thermal_guard is not None and thermal_guard.last_pressure_level is not None:
        temp_str = f", Thermal pressure: {thermal_guard.last_pressure_level}"

    print(
        f"    Train batch {batch_idx + 1}/{target_batches}, Elapsed DHM: {elapsed_dhm}, "
        f"Loss: {batch_loss:.4f}, Augmentation non-zero percentage: {avg_pct:.1f}%{temp_str}"
    )

    if output_dir is not None:
        _save_checkpoint(
            output_dir / "partial_latest.pt",
            model,
            optimizer,
            scaler,
            epoch,
            train_loss=total_loss / max(total_batches, 1),
            val_loss=float("nan"),
            train_paths=train_paths,
            val_paths=val_paths,
            ds_idx=-1,
            ema_state=ema_state,
        )

    return window_start
