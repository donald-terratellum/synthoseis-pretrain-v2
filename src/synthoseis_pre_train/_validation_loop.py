"""Validation loop execution helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from synthoseis_pre_train._checkpoint import _format_elapsed_dhm
from synthoseis_pre_train._thermal import ThermalGuard
from synthoseis_pre_train.gpu_utils import autocast_context


@dataclass
class ValidationDatasetRunResult:
    ds_loss: float
    ds_batches: int
    first_input: torch.Tensor | None
    first_output: torch.Tensor | None
    first_target: torch.Tensor | None
    window_start: float


def _prepare_validation_dataset(
    loader,
    target_for_loader: int | None,
    ds_name: str,
    ds_idx: int,
    total_datasets: int,
) -> int | None:
    """Resolve target batches and emit standard per-dataset preflight logging.

    Returns the target number of batches to run, or None when the dataset should
    be skipped for this epoch.
    """
    if target_for_loader is not None and target_for_loader <= 0:
        return None

    try:
        loader_len = len(loader)
    except Exception as exc:
        print(f"\n  WARNING: skipping val dataset {ds_name} — loader unavailable ({exc})")
        return None

    target_ds_batches = loader_len if target_for_loader is None else min(loader_len, target_for_loader)
    if target_ds_batches <= 0:
        print(f"\n  WARNING: skipping val dataset {ds_name} — 0 available batches")
        return None

    keys_str = ", ".join(loader.dataset.available_cubes)
    print(f"\n  Val dataset {ds_name}, {keys_str} [{ds_idx + 1}/{total_datasets}]")
    return target_ds_batches


def _run_validation_dataset(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    target_ds_batches: int,
    window_start: float,
    ds_name: str,
    thermal_guard: ThermalGuard | None = None,
) -> ValidationDatasetRunResult:
    """Run validation batches for one dataset loader."""
    first_input = None
    first_output = None
    first_target = None
    ds_loss = 0.0
    ds_batches = 0
    ds_nonzero_pct_sum = 0.0

    try:
        loader_iter = iter(loader)
        for batch_idx in range(target_ds_batches):
            try:
                input_data, target, mask = next(loader_iter)
            except StopIteration:
                break
            input_data = input_data.unsqueeze(1).float().to(device, non_blocking=True)
            target = target.unsqueeze(1).float().to(device, non_blocking=True)
            mask = mask.unsqueeze(1).to(device, non_blocking=True)

            with autocast_context(device):
                output = model(input_data)
                loss = criterion(output, target)

            if thermal_guard is not None:
                thermal_guard.sample_temperature(batch_idx)

            batch_loss = loss.item()
            ds_loss += batch_loss
            ds_batches += 1

            x_nz = (input_data != 0).sum().item()
            y_nz = (target != 0).sum().item()
            batch_pct = (x_nz / y_nz * 100.0) if y_nz > 0 else 0.0
            ds_nonzero_pct_sum += batch_pct

            if first_input is None:
                first_input = input_data[0].detach().cpu()
                first_output = output[0].detach().cpu()
                first_target = target[0].detach().cpu()

            if (batch_idx + 1) % 10 == 0:
                avg_pct = ds_nonzero_pct_sum / ds_batches
                elapsed_dhm = _format_elapsed_dhm(window_start)
                window_start = time.monotonic()
                temp_str = ""
                if thermal_guard is not None and thermal_guard.last_temp_c is not None:
                    temp_str = f", CPU temp: {thermal_guard.last_temp_c:.1f}C"
                elif thermal_guard is not None and thermal_guard.last_pressure_level is not None:
                    temp_str = f", Thermal pressure: {thermal_guard.last_pressure_level}"
                print(
                    f"    Val batch {batch_idx}/{target_ds_batches}, Elapsed DHM: {elapsed_dhm}, "
                    f"Loss: {batch_loss:.4f}, Augmentation non-zero percentage: {avg_pct:.1f}%{temp_str}"
                )
    except Exception as exc:
        print(f"    WARNING: val dataset {ds_name} failed mid-epoch — {exc}")

    return ValidationDatasetRunResult(
        ds_loss=ds_loss,
        ds_batches=ds_batches,
        first_input=first_input,
        first_output=first_output,
        first_target=first_target,
        window_start=window_start,
    )
