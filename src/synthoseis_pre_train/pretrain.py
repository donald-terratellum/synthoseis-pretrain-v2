"""
Training Script for Seismic 3D Mamba Pre-training
===================================================
"""

import os
import random
import time
import math
import platform
import shutil
import csv
import re

from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from pathlib import Path
from typing import Any, cast

import matplotlib
matplotlib.use("Agg")
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from synthoseis_pre_train.gpu_utils import (
    get_default_device,
    get_memory_info,
    print_device_summary,
    autocast_context,
    create_grad_scaler,
)
from synthoseis_pre_train.models import create_model, _MAMBA_AVAILABLE, _resolve_encoder_stage_blocks
from synthoseis_pre_train.models import report_masked_voxel_stats
from synthoseis_pre_train._ema import ModelEMA
from synthoseis_pre_train._checkpoint import _maybe_update_best_val_checkpoint, _save_checkpoint
from synthoseis_pre_train._scheduler import _build_lr_scheduler
from synthoseis_pre_train._criterion import _build_criterion, _print_loss_and_backprop_summary
from synthoseis_pre_train._dataset_manager import (
    _discover_zarr_paths,
    _prune_oldest_to_target,
    _update_split,
    _active_paths,
    _resolve_target_counts,
    _build_loaders,
)
from synthoseis_pre_train._npy_dataset import NpySeismicDataset as _NpySeismicDataset  # noqa: F401
from synthoseis_pre_train._thermal import ThermalGuard, _print_thermal_monitor_status
from synthoseis_pre_train._dataset_figures import _log_per_dataset_figures
from synthoseis_pre_train._validation_figures import _log_validation_crosssections
from synthoseis_pre_train._validation_schedule import _compute_per_loader_targets
from synthoseis_pre_train._train_figures import _log_train_merged_figure
from synthoseis_pre_train._validation_loop import _prepare_validation_dataset, _run_validation_dataset
from synthoseis_pre_train._train_progress import _log_train_progress_and_maybe_checkpoint
from synthoseis_pre_train._train_batch_fetch import _fetch_train_batch
from synthoseis_pre_train._train_step import _maybe_apply_optimizer_step
from synthoseis_pre_train.losses import LPIPSLoss, compute_pmse_loss, gradient_difference_loss_3d


def _resolve_resume_checkpoint_path(resume_path: str | Path) -> Path:
    """Resolve a resume checkpoint path with backward-compatible fallbacks.

    If the requested checkpoint is missing, prefer the newest
    checkpoint_epoch_*.pt in the same directory. This keeps older runs
    resumable even when they do not have checkpoint_final_model.pt yet.
    """
    path = Path(resume_path)
    if path.exists():
        return path

    parent = path.parent
    if parent.exists():
        epoch_candidates: list[tuple[int, Path]] = []
        for candidate in parent.glob("checkpoint_epoch_*.pt"):
            match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", candidate.name)
            if match is None:
                continue
            epoch_candidates.append((int(match.group(1)), candidate))

        if epoch_candidates:
            epoch_candidates.sort(key=lambda item: item[0])
            fallback = epoch_candidates[-1][1]
            print(
                f"WARNING: requested resume checkpoint not found: {path} ; "
                f"falling back to latest epoch checkpoint: {fallback}"
            )
            return fallback

    raise FileNotFoundError(f"Resume checkpoint not found: {path}")


def _backup_dataset_folders(
    dataset_paths: list[str],
    *,
    data_root: Path,
    backup_root: Path,
    epoch: int | None = None,
) -> list[str]:
    """Copy full dataset folders to backup_root preserving timestamps.

    `dataset_paths` are expected to point to zarr entries such as
    `.../seismic__.../model_data.zarr`. We copy the parent dataset folder.
    """
    copied: list[str] = []
    data_root_resolved = data_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    epoch_prefix = f"epoch={epoch} " if epoch is not None else ""
    print(
        "  Backup copy starting: "
        f"{epoch_prefix}data_root={data_root_resolved} backup_root={backup_root.resolve()} "
        f"dataset_count={len(sorted(set(dataset_paths)))}"
    )

    for path_str in sorted(set(dataset_paths)):
        src_zarr_path = Path(path_str)
        src_dataset_dir = src_zarr_path.parent
        try:
            rel_dataset_dir = src_dataset_dir.resolve().relative_to(data_root_resolved)
        except Exception:
            print(
                "  WARNING: skipping backup for dataset outside --data_folder: "
                f"{src_dataset_dir}"
            )
            continue

        dst_dataset_dir = backup_root / rel_dataset_dir
        try:
            print(
                f"  Copying dataset: {epoch_prefix}{src_dataset_dir} -> {dst_dataset_dir}"
            )
            if dst_dataset_dir.exists():
                shutil.rmtree(dst_dataset_dir)
            dst_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dataset_dir, dst_dataset_dir, copy_function=shutil.copy2)
            shutil.copystat(src_dataset_dir, dst_dataset_dir)
            print(f"  Backed up dataset: {epoch_prefix}{src_dataset_dir} -> {dst_dataset_dir}")
            copied.append(path_str)
        except Exception as exc:
            print(f"  WARNING: failed to back up dataset {epoch_prefix}{src_dataset_dir}: {exc}")

    return copied


DEFAULT_BACKPROP_DEFAULTS: dict[str, object] = {
    "lr": 1e-4,
    "lr_schedule": "poly",
    "loss": "huber",
    "mae_smooth_kernel_weights": [1.0, 2.0, 1.0],
    "huber_delta": 1.0,
    "ssim_window_size": 7,
    "ssim_w1": 1.0,
    "ssim_w2": 0.0,
    "ssim_w3": 0.0,
    "stats_window_size": [9, 9, 9],
    "stats_mask_mode": "none",
    "stats_mean_weight": 1.0,
    "stats_std_weight": 1.0,
    "stats_min_weight": 1.0,
    "stats_max_weight": 1.0,
    "stats_mae_weight": 1.0,
    "stats_mse_weight": 1.0,
    "stats_std_ratio_clip": 10.0,
    "mc_mse_weight": 0.2,
    "mc_pmse_weight": 0.6,
    "mc_mae_weight": 0.2,
    "mc_lpips_weight": 0.0,
    "mc_lpips_net": "alex",
    "mc_lpips_calib_weight": 0.0,
    "mc_pmse_eps": 1e-8,
    "mc_tv_weight": 0.0,
    "mc_gdl_weight": 0.0,
    "unet_levels": 4,
    "encoder_depth_profile": "baseline",
    "grad_accum_steps": 1,
    "grad_clip_norm": 1.0,
    "ema_decay": 0.999,
    "ema_update_every": 1,
    "epoch_samples": None,
}


_COMPONENT_METRIC_CSV_HEADERS = [
    "date",
    "time",
    "tensorboard folder",
    "epoch",
    "unet_levels",
    "hidden_dims (as a space-delimited string)",
    "kernel_schedule (as a space-delimited string)",
    "encoder_depth_profile (profile - stage blocks)",
    "lr",
    "model parameter count (in millions, for example 11,324,033 is shown as 11.32)",
    "mse_weight",
    "pmse_weight",
    "mae_weight",
    "lpips_weight",
    "tv_weight",
    "gdl_weight",
    "train mse",
    "train pmse",
    "train mae/L1",
    "train LPIPS",
    "train gdl",
    "validation mse",
    "validation pmse",
    "validation mae/L1",
    "validation LPIPS",
    "validation gdl",
    "test mse",
    "test pmse",
    "test mae/L1",
    "test LPIPS",
    "test gdl",
]

_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_TV = [
    header
    for header in _COMPONENT_METRIC_CSV_HEADERS
    if header not in (
        "encoder_depth_profile (profile - stage blocks)",
        "lr",
        "tv_weight",
        "gdl_weight",
        "train gdl",
        "validation gdl",
    )
]
_LEGACY_COMPONENT_METRIC_CSV_HEADERS = [
    header
    for header in _COMPONENT_METRIC_CSV_HEADERS
    if header not in ("encoder_depth_profile (profile - stage blocks)", "lr", "gdl_weight", "train gdl", "validation gdl")
]
_LEGACY_COMPONENT_METRIC_CSV_HEADERS_WITH_GDL_WEIGHT = [
    header
    for header in _COMPONENT_METRIC_CSV_HEADERS
    if header not in ("encoder_depth_profile (profile - stage blocks)", "lr", "train gdl", "validation gdl")
]
_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_LR = [
    header
    for header in _COMPONENT_METRIC_CSV_HEADERS
    if header not in ("lr",)
]

# Headers without the 5 test columns (all runs before real-data test support).
_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_TEST = [
    header
    for header in _COMPONENT_METRIC_CSV_HEADERS
    if header not in ("test mse", "test pmse", "test mae/L1", "test LPIPS", "test gdl")
]


def _new_component_metric_totals() -> dict[str, float]:
    return {
        "count": 0.0,
        "mse": 0.0,
        "pmse": 0.0,
        "mae": 0.0,
        "lpips": 0.0,
        "gdl": 0.0,
    }


def _update_component_metric_totals(
    totals: dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_metric: nn.Module | None,
) -> None:
    batch_size = float(pred.shape[0]) if pred.ndim > 0 else 1.0
    mse = float(F.mse_loss(pred, target).item())
    pmse = float(compute_pmse_loss(pred, target).item())
    mae = float(F.l1_loss(pred, target).item())
    lpips = float(lpips_metric(pred, target).item()) if lpips_metric is not None else 0.0
    gdl = float(gradient_difference_loss_3d(pred, target).item())

    totals["count"] += batch_size
    totals["mse"] += mse * batch_size
    totals["pmse"] += pmse * batch_size
    totals["mae"] += mae * batch_size
    totals["lpips"] += lpips * batch_size
    totals["gdl"] += gdl * batch_size


def _finalize_component_metrics(totals: dict[str, float]) -> dict[str, float]:
    count = max(totals.get("count", 0.0), 1.0)
    return {
        "mse": totals.get("mse", 0.0) / count,
        "pmse": totals.get("pmse", 0.0) / count,
        "mae": totals.get("mae", 0.0) / count,
        "lpips": totals.get("lpips", 0.0) / count,
        "gdl": totals.get("gdl", 0.0) / count,
    }


def _sanitize_component_metric_cells(values: list[str]) -> list[str]:
    return [v.lstrip("\n") if isinstance(v, str) else str(v).lstrip("\n") for v in values]


def _normalize_existing_component_metrics_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        return

    raw_text = csv_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return

    split_text = re.sub(
        r"(?<!\n)(?=(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2}),)",
        "\n",
        raw_text,
    )
    parsed_rows = list(csv.reader(split_text.splitlines()))
    if not parsed_rows:
        return

    expected_len = len(_COMPONENT_METRIC_CSV_HEADERS)
    legacy_len = len(_LEGACY_COMPONENT_METRIC_CSV_HEADERS)
    legacy_gdl_weight_len = len(_LEGACY_COMPONENT_METRIC_CSV_HEADERS_WITH_GDL_WEIGHT)
    legacy_no_lr_len = len(_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_LR)
    normalized_rows: list[list[str]] = []
    rewrite_needed = split_text != raw_text

    header = _sanitize_component_metric_cells(parsed_rows[0])
    if header == _COMPONENT_METRIC_CSV_HEADERS:
        normalized_header = _COMPONENT_METRIC_CSV_HEADERS
    elif header == _LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_LR:
        normalized_header = _COMPONENT_METRIC_CSV_HEADERS
        rewrite_needed = True
    elif header == _LEGACY_COMPONENT_METRIC_CSV_HEADERS_WITH_GDL_WEIGHT:
        normalized_header = _COMPONENT_METRIC_CSV_HEADERS
        rewrite_needed = True
    elif header == _LEGACY_COMPONENT_METRIC_CSV_HEADERS:
        normalized_header = _COMPONENT_METRIC_CSV_HEADERS
        rewrite_needed = True
    else:
        normalized_header = _COMPONENT_METRIC_CSV_HEADERS
        rewrite_needed = True

    tv_idx = _COMPONENT_METRIC_CSV_HEADERS.index("tv_weight")
    gdl_idx = _COMPONENT_METRIC_CSV_HEADERS.index("gdl_weight")
    train_gdl_idx = _COMPONENT_METRIC_CSV_HEADERS.index("train gdl")
    validation_gdl_idx = _COMPONENT_METRIC_CSV_HEADERS.index("validation gdl")
    encoder_depth_profile_idx = _COMPONENT_METRIC_CSV_HEADERS.index("encoder_depth_profile (profile - stage blocks)")
    lr_idx = _COMPONENT_METRIC_CSV_HEADERS.index("lr")
    legacy_no_tv_len = len(_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_TV)
    legacy_no_test_len = len(_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_TEST)
    for raw_row in parsed_rows[1:]:
        row = _sanitize_component_metric_cells(raw_row)
        if not any(cell.strip() for cell in row):
            rewrite_needed = True
            continue
        if len(row) == legacy_no_lr_len:
            # Rows from schema immediately before LR column was introduced.
            row = row[:lr_idx] + [""] + row[lr_idx:]
            rewrite_needed = True
        elif len(row) == legacy_no_tv_len:
            # Oldest rows missing tv_weight, gdl_weight, train gdl, and validation gdl.
            row = row[:encoder_depth_profile_idx] + [""] + row[encoder_depth_profile_idx:]
            row = row[:lr_idx] + [""] + row[lr_idx:]
            row = row[:tv_idx] + ["0.000000", "0.000000"] + row[tv_idx:]
            row = row[:train_gdl_idx] + ["0.000000"] + row[train_gdl_idx:]
            row = row[:validation_gdl_idx] + ["0.000000"] + row[validation_gdl_idx:]
            rewrite_needed = True
        elif len(row) == legacy_len:
            # Rows with tv_weight but missing gdl_weight, train gdl, and validation gdl.
            row = row[:encoder_depth_profile_idx] + [""] + row[encoder_depth_profile_idx:]
            row = row[:lr_idx] + [""] + row[lr_idx:]
            row = row[:gdl_idx] + ["0.000000"] + row[gdl_idx:]
            row = row[:train_gdl_idx] + ["0.000000"] + row[train_gdl_idx:]
            row = row[:validation_gdl_idx] + ["0.000000"] + row[validation_gdl_idx:]
            rewrite_needed = True
        elif len(row) == legacy_gdl_weight_len:
            # Rows with gdl_weight but missing train/validation gdl metric columns.
            row = row[:encoder_depth_profile_idx] + [""] + row[encoder_depth_profile_idx:]
            row = row[:lr_idx] + [""] + row[lr_idx:]
            row = row[:train_gdl_idx] + ["0.000000"] + row[train_gdl_idx:]
            row = row[:validation_gdl_idx] + ["0.000000"] + row[validation_gdl_idx:]
            rewrite_needed = True
        elif len(row) == legacy_no_test_len:
            # Rows from before real-data test columns were introduced.
            row = row + ["", "", "", "", ""]
            rewrite_needed = True
        elif len(row) != expected_len:
            raise ValueError(
                f"Unexpected component metrics CSV row width in {csv_path}: "
                f"expected {expected_len}, {legacy_no_lr_len}, {legacy_len}, "
                f"{legacy_gdl_weight_len}, or {legacy_no_test_len} columns, got {len(row)}"
            )
        normalized_rows.append(row)

    if not rewrite_needed:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(normalized_header)
        writer.writerows(normalized_rows)


def _append_component_metrics_csv_row(
    csv_path: Path,
    tb_log_dir: Path,
    epoch: int,
    args,
    n_params: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None = None,
) -> None:
    now = datetime.now()
    hidden_dims_str = " ".join(str(int(v)) for v in tuple(args.hidden_dims))
    if args.kernel_sizes is None:
        kernel_schedule_vals = [3] * len(tuple(args.hidden_dims))
    else:
        kernel_schedule_vals = [int(v) for v in tuple(args.kernel_sizes)]
    kernel_schedule_str = " ".join(str(v) for v in kernel_schedule_vals)
    encoder_stage_blocks = _resolve_encoder_stage_blocks(
        int(args.unet_levels),
        tuple(args.encoder_stage_blocks) if getattr(args, "encoder_stage_blocks", None) is not None else None,
        str(getattr(args, "encoder_depth_profile", "baseline")),
    )
    encoder_depth_profile_str = (
        f"{str(getattr(args, 'encoder_depth_profile', 'baseline'))} - "
        f"{' '.join(str(int(v)) for v in encoder_stage_blocks)}"
    )

    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        str(tb_log_dir),
        str(int(epoch)),
        str(int(args.unet_levels)),
        hidden_dims_str,
        kernel_schedule_str,
        encoder_depth_profile_str,
        f"{float(lr):.8e}",
        f"{(float(n_params) / 1_000_000.0):.2f}",
        f"{float(getattr(args, 'mc_mse_weight', 0.0)):.6f}",
        f"{float(getattr(args, 'mc_pmse_weight', 0.0)):.6f}",
        f"{float(getattr(args, 'mc_mae_weight', 0.0)):.6f}",
        f"{float(getattr(args, 'mc_lpips_weight', 0.0)):.6f}",
        f"{float(getattr(args, 'mc_tv_weight', 0.0)):.6f}",
        f"{float(getattr(args, 'mc_gdl_weight', 0.0)):.6f}",
        f"{float(train_metrics.get('mse', float('nan'))):.8f}",
        f"{float(train_metrics.get('pmse', float('nan'))):.8f}",
        f"{float(train_metrics.get('mae', float('nan'))):.8f}",
        f"{float(train_metrics.get('lpips', float('nan'))):.8f}",
        f"{float(train_metrics.get('gdl', float('nan'))):.8f}",
        f"{float(val_metrics.get('mse', float('nan'))):.8f}",
        f"{float(val_metrics.get('pmse', float('nan'))):.8f}",
        f"{float(val_metrics.get('mae', float('nan'))):.8f}",
        f"{float(val_metrics.get('lpips', float('nan'))):.8f}",
        f"{float(val_metrics.get('gdl', float('nan'))):.8f}",
        # test columns — empty string when no test data was evaluated this run.
        f"{float(test_metrics.get('mse', float('nan'))):.8f}" if test_metrics else "",
        f"{float(test_metrics.get('pmse', float('nan'))):.8f}" if test_metrics else "",
        f"{float(test_metrics.get('mae', float('nan'))):.8f}" if test_metrics else "",
        f"{float(test_metrics.get('lpips', float('nan'))):.8f}" if test_metrics else "",
        f"{float(test_metrics.get('gdl', float('nan'))):.8f}" if test_metrics else "",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_existing_component_metrics_csv(csv_path)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        if write_header:
            writer.writerow(_sanitize_component_metric_cells(_COMPONENT_METRIC_CSV_HEADERS))
        writer.writerow(_sanitize_component_metric_cells(row))


def _default_batch_size_for_model(n_params: int) -> int:
    return 2 if int(n_params) < 14_000_000 else 1


def _fixed_validation_heuristic(val_metrics: dict[str, float]) -> float:
    return (
        2.0 * float(val_metrics.get("mae", float("nan")))
        + float(val_metrics.get("mse", float("nan")))
        + float(val_metrics.get("lpips", float("nan")))
    )


def _scale_optimizer_lr(
    optimizer: optim.Optimizer,
    scale: float,
    scheduler=None,
) -> tuple[float, float]:
    """Scale current optimizer LR and scheduler base LRs by a constant factor."""
    factor = float(scale)
    if factor <= 0.0:
        raise ValueError("LR scale factor must be > 0")

    old_lr = float(optimizer.param_groups[0]["lr"])
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * factor
    new_lr = float(optimizer.param_groups[0]["lr"])

    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [float(v) * factor for v in scheduler.base_lrs]

    return old_lr, new_lr


def run_training(config: dict[str, Any]) -> None:
    args_dict = config.get("args")
    if not isinstance(args_dict, dict):
        raise ValueError("run_training(config): expected config[args] dict")

    cli_provided_raw = config.get("cli_provided")
    if isinstance(cli_provided_raw, list):
        cli_provided = {str(item) for item in cli_provided_raw}
    else:
        cli_provided = set()

    backprop_defaults = dict(DEFAULT_BACKPROP_DEFAULTS)
    incoming_defaults = config.get("backprop_defaults")
    if isinstance(incoming_defaults, dict):
        backprop_defaults.update({str(k): v for k, v in incoming_defaults.items()})

    from types import SimpleNamespace

    args = SimpleNamespace(**args_dict)
    _run_training_with_args(args, cli_provided, backprop_defaults)


# Defensive runtime scrub: set all Malloc* vars to "0" (explicit disable signal
# to libmalloc) rather than unsetting — absent vars may still trigger warnings
# on some macOS versions; "0" is the documented way to disable stack logging.
if platform.system() == "Darwin":
    for _k in list(os.environ.keys()):
        if _k.startswith("Malloc"):
            os.environ[_k] = "0"
    # Ensure the two key vars are present even if not already in env
    os.environ["MallocStackLogging"] = "0"
    os.environ["MallocStackLoggingNoCompact"] = "0"



def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
    writer: SummaryWriter | None = None,
    epoch: int = 0,
    output_dir: Path | None = None,
    train_paths: list[str] | None = None,
    val_paths: list[str] | None = None,
    thermal_guard: ThermalGuard | None = None,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 0.0,
    ema: ModelEMA | None = None,
    ema_update_every: int = 1,
    max_batches: int | None = None,
    return_details: bool = False,
    reporting_lpips: nn.Module | None = None,
    log_image_this_epoch: bool = True,
) -> float | dict[str, float | int | bool]:
    """
    Train for one epoch using a single merged train DataLoader.

    The loader is expected to draw samples from all source datasets through
    ConcatDataset + shuffle so each optimizer step sees mixed data.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    accum_steps = max(1, int(grad_accum_steps))
    ema_every = max(1, int(ema_update_every))
    optimizer_steps = 0
    micro_batches = 0
    optimizer.zero_grad(set_to_none=True)

    window_start = time.monotonic()
    nz_pct_sum = 0.0
    last_input = None
    last_output = None
    last_target = None
    component_totals = _new_component_metric_totals()

    try:
        natural_batches = len(train_loader)
    except Exception as e:
        print(f"    WARNING: train loader length unavailable — {e}")
        natural_batches = 0

    if natural_batches == 0:
        avg_loss = float("nan")
        if return_details:
            return {
                "loss": avg_loss,
                "batches_processed": 0,
                "reload_requested": False,
                "mse": float("nan"),
                "pmse": float("nan"),
                "mae": float("nan"),
                "lpips": float("nan"),
                "gdl": float("nan"),
            }
        return avg_loss

    target_batches = natural_batches if max_batches is None else max(1, int(max_batches))
    iter_start_t0 = time.monotonic()
    loader_iter = iter(train_loader)
    iter_elapsed_min = (time.monotonic() - iter_start_t0) / 60.0
    print(f"    Train iterator/sampler startup: {iter_elapsed_min:04.1f}m")
    reload_requested = False
    for batch_idx in range(target_batches):
        _fetch_t0 = time.monotonic()  # TODO: remove this line
        fetch_result = _fetch_train_batch(
            loader_iter=loader_iter,
            train_loader=train_loader,
            device=device,
            batch_idx=batch_idx,
        )
        _fetch_elapsed_sec = time.monotonic() - _fetch_t0  # TODO: remove this line
        if _fetch_elapsed_sec > 1.25 or batch_idx < 3:  # TODO: remove this line
            _fetch_fmt = int(_fetch_elapsed_sec) if _fetch_elapsed_sec >= 1.0 else f"{_fetch_elapsed_sec*1000:.0f}ms"  # TODO: remove this line
            print(f"         . [diag] batch {batch_idx}: fetch_train_batch {_fetch_fmt}")  # TODO: remove this line
        loader_iter = fetch_result.loader_iter
        if fetch_result.should_break:
            reload_requested = fetch_result.reload_requested
            break
        input_data = fetch_result.input_data
        target = fetch_result.target
        mask = fetch_result.mask
        source_tags = fetch_result.source_tags
        if input_data is None or target is None or mask is None:
            reload_requested = True
            break

        with autocast_context(device):
            output = model(input_data)
            # loss = criterion(output[~mask], target[~mask])
            loss = criterion(output, target)  # TODO: switch to masked loss when stable ?
        if batch_idx < 10:
            report_masked_voxel_stats(input_data, target=target, mask=mask, source_tags=source_tags)
        batch_loss = loss.item()
        scaled_loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        micro_batches, optimizer_steps = _maybe_apply_optimizer_step(
            scaler=scaler,
            optimizer=optimizer,
            model=model,
            grad_clip_norm=grad_clip_norm,
            micro_batches=micro_batches,
            accum_steps=accum_steps,
            batch_idx=batch_idx,
            target_batches=target_batches,
            optimizer_steps=optimizer_steps,
            ema=ema,
            ema_every=ema_every,
        )

        temp_c = None
        if thermal_guard is not None:
            temp_c = thermal_guard.sample_temperature(batch_idx)

        if thermal_guard is not None:
            thermal_guard.maybe_pause(
                epoch=epoch,
                ds_idx=-1,
                batch_idx=batch_idx,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_paths=train_paths or [],
                val_paths=val_paths or [],
                temp_c=temp_c,
                ema_state=ema.state_dict() if ema is not None else None,
            )

        with torch.no_grad():
            _update_component_metric_totals(
                component_totals,
                output.detach(),
                target.detach(),
                reporting_lpips,
            )

        total_loss += batch_loss
        total_batches += 1

        with torch.no_grad():
            x_nz = (input_data != 0).sum().item()
            y_nz = (target != 0).sum().item()
            batch_pct = (x_nz / y_nz * 100.0) if y_nz > 0 else 0.0
        nz_pct_sum += batch_pct

        # Keep last batch tensors for end-of-epoch diagnostic plotting.
        last_input = input_data[0].detach().cpu()
        last_output = output[0].detach().cpu()
        last_target = target[0].detach().cpu()

        if (batch_idx + 1) % 10 == 0:
            window_start = _log_train_progress_and_maybe_checkpoint(
                batch_idx=batch_idx,
                target_batches=target_batches,
                batch_loss=batch_loss,
                nz_pct_sum=nz_pct_sum,
                total_batches=total_batches,
                window_start=window_start,
                thermal_guard=thermal_guard,
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                total_loss=total_loss,
                train_paths=train_paths,
                val_paths=val_paths,
                ema_state=ema.state_dict() if ema is not None else None,
            )

    if (
        bool(log_image_this_epoch)
        and
        writer is not None
        and last_input is not None
        and last_output is not None
        and last_target is not None
    ):
        avg_epoch_loss = total_loss / max(total_batches, 1)
        _log_train_merged_figure(
            writer,
            last_input,
            last_output,
            last_target,
            epoch,
            avg_epoch_loss,
        )

    avg_loss = total_loss / max(total_batches, 1)
    component_metrics = _finalize_component_metrics(component_totals)
    if return_details:
        return {
            "loss": avg_loss,
            "batches_processed": total_batches,
            "reload_requested": reload_requested,
            "mse": component_metrics["mse"],
            "pmse": component_metrics["pmse"],
            "mae": component_metrics["mae"],
            "lpips": component_metrics["lpips"],
            "gdl": component_metrics["gdl"],
        }
    return avg_loss


def validate(
    model: nn.Module,
    val_loaders: list,
    criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter | None = None,
    epoch: int = 0,
    thermal_guard: ThermalGuard | None = None,
    max_batches: int | None = None,
    return_details: bool = False,
    reporting_lpips: nn.Module | None = None,
    log_image_this_epoch: bool = True,
) -> float | dict[str, float]:
    """
    Validate the model across all validation datasets.

    At the end of each validation dataset, logs 4 separate cross-section figures
    to TensorBoard (input & output × center-X & center-Y). In the TensorBoard UI,
    select tag prefixes to toggle between input/output for each slice direction.
    """
    if not val_loaders:
        if return_details:
            return {
                "loss": float("nan"),
                "mse": float("nan"),
                "pmse": float("nan"),
                "mae": float("nan"),
                "lpips": float("nan"),
                "gdl": float("nan"),
            }
        return float('nan')

    model.eval()
    total_loss = 0.0
    total_batches = 0
    component_totals = _new_component_metric_totals()
    val_start = time.monotonic()
    window_start = val_start
    per_loader_targets = _compute_per_loader_targets(max_batches, len(val_loaders))

    with torch.no_grad():
        for ds_idx, (ds_name, loader) in enumerate(val_loaders):
            target_for_loader = per_loader_targets[ds_idx]
            target_ds_batches = _prepare_validation_dataset(
                loader=loader,
                target_for_loader=target_for_loader,
                ds_name=ds_name,
                ds_idx=ds_idx,
                total_datasets=len(val_loaders),
            )
            if target_ds_batches is None:
                continue

            ds_result = _run_validation_dataset(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                target_ds_batches=target_ds_batches,
                window_start=window_start,
                ds_name=ds_name,
                thermal_guard=thermal_guard,
                metric_updater=(
                    lambda pred, target: _update_component_metric_totals(
                        component_totals,
                        pred,
                        target,
                        reporting_lpips,
                    )
                ),
            )
            ds_loss = ds_result.ds_loss
            ds_batches = ds_result.ds_batches
            first_input = ds_result.first_input
            first_output = ds_result.first_output
            first_target = ds_result.first_target
            window_start = ds_result.window_start
            total_loss += ds_loss
            total_batches += ds_batches

            # --- Per-val-dataset: 4 separate TensorBoard images ---
            # Tags are structured so TensorBoard shows paired input/output
            # under the same group for each slice direction.
            if (
                bool(log_image_this_epoch)
                and
                writer is not None
                and first_input is not None
                and first_output is not None
                and first_target is not None
            ):
                avg_ds_loss = ds_loss / max(ds_batches, 1)
                _log_validation_crosssections(
                    writer,
                    ds_name,
                    first_input,
                    first_output,
                    first_target,
                    epoch,
                    avg_ds_loss,
                )

    avg_loss = total_loss / max(total_batches, 1)
    if return_details:
        metrics = _finalize_component_metrics(component_totals)
        return {
            "loss": float(avg_loss),
            "mse": float(metrics["mse"]),
            "pmse": float(metrics["pmse"]),
            "mae": float(metrics["mae"]),
            "lpips": float(metrics["lpips"]),
            "gdl": float(metrics["gdl"]),
        }
    return avg_loss


DEFAULT_ARRAY_KEYS = [
    "seismicCubes_cumsum__17_degrees",
    # "seismicCubes_cumsum__17_degrees_normalized",
    "seismicCubes_cumsum__29_degrees",
    # "seismicCubes_cumsum__29_degrees_normalized",
    "seismicCubes_cumsum__5_degrees",
    # "seismicCubes_cumsum__5_degrees_normalized",
    # "seismicCubes_cumsum_17_degrees_normalized_augmented",
    # "seismicCubes_cumsum_29_degrees_normalized_augmented",
    # "seismicCubes_cumsum_5_degrees_normalized_augmented",
    "seismicCubes_cumsum_fullstack",
    # "seismicCubes_cumsum_fullstack_noise_free"
]

DEFAULT_GEOLOGIC_SCORE_KEYS = [
    "geological_score",
    "geologic_score",
]


def _run_training_with_args(args, cli_provided: set[str], backprop_defaults: dict[str, object]) -> None:
    def _config_error(message: str) -> None:
        raise ValueError(message)

    if not (0.0 < args.val_split_ratio < 1.0):
        _config_error("--val_split_ratio must be between 0 and 1 (exclusive)")
    if args.train_batches_per_epoch is not None and args.train_batches_per_epoch <= 0:
        _config_error("--train_batches_per_epoch must be > 0")
    if args.val_batches_per_epoch is not None and args.val_batches_per_epoch <= 0:
        _config_error("--val_batches_per_epoch must be > 0")
    if args.test_batches_per_epoch is not None and args.test_batches_per_epoch <= 0:
        _config_error("--test_batches_per_epoch must be > 0")
    if args.refresh_every_batches < 0:
        _config_error("--refresh_every_batches must be >= 0")
    if args.kernel_sizes is not None:
        hidden_dims_val = tuple(args.hidden_dims)
        if len(args.kernel_sizes) != len(hidden_dims_val):
            _config_error(
                "--kernel_sizes length must match hidden dims "
                f"({len(hidden_dims_val)} values expected for {hidden_dims_val})"
            )
        if any(k <= 0 or k % 2 == 0 for k in args.kernel_sizes):
            _config_error("--kernel_sizes values must be positive odd integers")
    if len(tuple(args.hidden_dims)) != int(args.unet_levels):
        _config_error(
            "--hidden_dims length must match --unet_levels "
            f"(got hidden_dims={tuple(args.hidden_dims)}, unet_levels={args.unet_levels})"
        )
    if float(getattr(args, "mc_lpips_weight", 0.0)) > 0.0:
        forced_lr = 1.0e-5
        forced_lr_min = 5.0e-6
        if float(getattr(args, "lr", forced_lr)) != forced_lr or float(getattr(args, "lr_min", forced_lr_min)) != forced_lr_min:
            print(
                "LPIPS weight > 0 detected; overriding LR settings to "
                f"lr={forced_lr:.1e}, lr_min={forced_lr_min:.1e}"
            )
        args.lr = forced_lr
        args.lr_min = forced_lr_min
    if getattr(args, "encoder_stage_blocks", None) is not None:
        if len(tuple(args.encoder_stage_blocks)) != int(args.unet_levels):
            _config_error(
                "--encoder_stage_blocks length must match --unet_levels "
                f"(got encoder_stage_blocks={tuple(args.encoder_stage_blocks)}, "
                f"unet_levels={args.unet_levels})"
            )
        if any(int(v) <= 0 for v in tuple(args.encoder_stage_blocks)):
            _config_error("--encoder_stage_blocks values must be positive integers")
    encoder_downsample_factor = 2 ** (int(args.unet_levels) + 1)
    bottleneck_shape = tuple(int(s) // encoder_downsample_factor for s in args.sample_shape)
    if min(bottleneck_shape) < 2:
        _config_error(
            "sample_shape is too small for the requested --unet_levels; "
            "InstanceNorm requires >1 spatial element at bottleneck. "
            f"Got sample_shape={tuple(args.sample_shape)}, unet_levels={args.unet_levels}, "
            f"downsample_factor={encoder_downsample_factor}, bottleneck_shape={bottleneck_shape}."
        )
    if min(bottleneck_shape) < 4:
        print(
            "WARNING: very small bottleneck spatial size may hurt training stability: "
            f"sample_shape={tuple(args.sample_shape)}, unet_levels={args.unet_levels}, "
            f"bottleneck_shape={bottleneck_shape}."
        )
    if args.ssim_window_size < 3 or args.ssim_window_size % 2 == 0:
        _config_error("--ssim_window_size must be an odd integer >= 3")
    if min(args.sample_shape) < args.ssim_window_size:
        _config_error(
            "--ssim_window_size must be <= each sample_shape dimension "
            f"(got window={args.ssim_window_size}, sample_shape={tuple(args.sample_shape)})"
        )
    if args.loss == "ssim" and (args.ssim_w1 < 0 or args.ssim_w2 < 0 or args.ssim_w3 < 0):
        _config_error("--ssim_w1, --ssim_w2, and --ssim_w3 must be >= 0")
    if any(int(v) <= 0 for v in args.stats_window_size):
        _config_error("--stats_window_size entries must be positive integers")
    if args.stats_std_ratio_clip <= 1.0:
        _config_error("--stats_std_ratio_clip must be > 1.0")
    if min(
        args.stats_mean_weight,
        args.stats_std_weight,
        args.stats_min_weight,
        args.stats_max_weight,
        args.stats_mae_weight,
        args.stats_mse_weight,
    ) < 0:
        _config_error("--stats_*_weight values must be >= 0")
    if len(args.mae_smooth_kernel_weights) < 3 or len(args.mae_smooth_kernel_weights) % 2 == 0:
        _config_error("--mae_smooth_kernel_weights must contain an odd number of values >= 3")
    if any(v < 0 for v in args.mae_smooth_kernel_weights):
        _config_error("--mae_smooth_kernel_weights values must be >= 0")
    if sum(float(v) for v in args.mae_smooth_kernel_weights) <= 0:
        _config_error("--mae_smooth_kernel_weights must sum to > 0")
    if args.loss == "multi_component":
        if min(
            args.mc_mse_weight,
            args.mc_pmse_weight,
            args.mc_mae_weight,
            args.mc_lpips_weight,
            getattr(args, "mc_lpips_calib_weight", 0.0),
            args.mc_tv_weight,
            getattr(args, "mc_gdl_weight", 0.0),
        ) < 0:
            _config_error(
                "--mc_mse_weight, --mc_pmse_weight, --mc_mae_weight, --mc_lpips_weight, "
                "--mc_lpips_calib_weight, --mc_tv_weight, and --mc_gdl_weight must be >= 0"
            )
        if (
            args.mc_mse_weight
            + args.mc_pmse_weight
            + args.mc_mae_weight
            + args.mc_lpips_weight
            + getattr(args, "mc_lpips_calib_weight", 0.0)
            + args.mc_tv_weight
            + getattr(args, "mc_gdl_weight", 0.0)
        ) <= 0:
            _config_error(
                "At least one of --mc_mse_weight, --mc_pmse_weight, --mc_mae_weight, "
                "--mc_lpips_weight, --mc_lpips_calib_weight, --mc_tv_weight, or --mc_gdl_weight must be > 0"
            )
        if args.mc_pmse_eps <= 0:
            _config_error("--mc_pmse_eps must be > 0")
    if args.tb_image_epochs is not None:
        if any(int(v) <= 0 for v in args.tb_image_epochs):
            _config_error("--tb_image_epochs values must be positive epoch numbers (1-based)")

    tb_image_epochs_set: set[int] | None = None
    if args.tb_image_epochs is not None:
        tb_image_epochs_set = {int(v) for v in args.tb_image_epochs}
        print(f"TensorBoard image epochs (1-based): {sorted(tb_image_epochs_set)}")
    else:
        print("TensorBoard image epochs (1-based): all")

    if not args.data_paths and not args.data_folder:
        _config_error("At least one of --data_paths or --data_folder must be provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backup_dir_raw = getattr(args, "backup_dir", None)
    backup_dir_path: Path | None = None
    if backup_dir_raw:
        backup_dir_path = Path(str(backup_dir_raw)).expanduser()
        backup_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Dataset backup enabled: {backup_dir_path}")

    device = get_default_device(args.device)
    print_device_summary(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if device.type == "mps" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # --- Dataset split (done once; restored from checkpoint on resume) ---
    validation_subfolder = Path(args.data_folder) / "validation" if args.data_folder else None
    use_separate_validation_folder = bool(validation_subfolder is not None and validation_subfolder.is_dir())
    if args.data_folder:
        if use_separate_validation_folder:
            print(f"Validation dataset source: {validation_subfolder} (dedicated validation subfolder)")
        else:
            print(f"Validation dataset source: {args.data_folder} (fallback: no validation subfolder found)")

    # Build initial path list: explicit --data_paths + discover from --data_folder
    all_paths = list(dict.fromkeys(args.data_paths))  # deduplicate preserving order
    discovered_at_start: list[str] = []
    val_discovered_at_start: list[str] = []
    if args.data_folder:
        discovered_at_start = _discover_zarr_paths(args.data_folder, args.dataset_glob)
        if use_separate_validation_folder and validation_subfolder is not None:
            val_root = validation_subfolder.resolve()
            discovered_at_start = [
                p for p in discovered_at_start
                if not Path(p).resolve().is_relative_to(val_root)
            ]
            val_discovered_at_start = _discover_zarr_paths(str(validation_subfolder), args.dataset_glob)
        known = set(all_paths)
        all_paths = all_paths + [p for p in discovered_at_start if p not in known]

    backed_up_dataset_paths: set[str] = set(discovered_at_start + val_discovered_at_start)

    # Check for a saved split in the resume checkpoint BEFORE shuffling
    saved_train_paths = None
    saved_val_paths   = None
    if args.resume and Path(args.resume).exists():
        _peek = torch.load(args.resume, map_location="cpu")
        saved_train_paths = _peek.get("train_paths")
        saved_val_paths   = _peek.get("val_paths")
        del _peek

    if use_separate_validation_folder:
        initial_num_train = len(all_paths)
        initial_num_val = len(val_discovered_at_start)
    else:
        initial_num_train, initial_num_val = _resolve_target_counts(
            len(all_paths), args.val_split_ratio
        )

    if use_separate_validation_folder:
        train_paths = list(all_paths)
        val_paths = list(val_discovered_at_start)

        if saved_train_paths is not None:
            supplied_train = set(all_paths)
            kept_train = list(dict.fromkeys(p for p in saved_train_paths if p in supplied_train))
            new_train = [p for p in all_paths if p not in set(kept_train)]
            train_paths = kept_train + new_train

        if saved_val_paths is not None:
            supplied_val = set(val_discovered_at_start)
            kept_val = list(dict.fromkeys(p for p in saved_val_paths if p in supplied_val))
            new_val = [p for p in val_discovered_at_start if p not in set(kept_val)]
            val_paths = kept_val + new_val

        split_target_train = len(train_paths)
        split_target_val = len(val_paths)
        print(f"Initialized split with dedicated validation folder: {len(train_paths)} train, {len(val_paths)} val datasets.")
    elif saved_train_paths is not None and saved_val_paths is not None:
        supplied = set(all_paths)

        # Drop paths that no longer exist in the supplied list; deduplicate in case
        # a previous run saved a corrupt split with duplicate entries
        kept_train = list(dict.fromkeys(p for p in saved_train_paths if p in supplied))
        kept_val   = list(dict.fromkeys(p for p in saved_val_paths   if p in supplied))
        dropped    = [p for p in (saved_train_paths + saved_val_paths) if p not in supplied]
        if dropped:
            print(f"Checkpoint split: dropped {len(dropped)} path(s) no longer supplied:")
            for p in dropped:
                print(f"  - {Path(p).parent.name}")

        # Identify new paths not present in the checkpoint split at all
        checkpoint_all = set(saved_train_paths) | set(saved_val_paths)
        new_paths = [p for p in all_paths if p not in checkpoint_all]

        if new_paths:
            # Assign new paths to fill deficits (val first when both short)
            new_train, new_val = [], []
            for p in new_paths:
                train_need = initial_num_train - len(kept_train) - len(new_train)
                val_need   = initial_num_val   - len(kept_val)   - len(new_val)
                if val_need > 0:
                    new_val.append(p)
                elif train_need > 0:
                    new_train.append(p)
                else:
                    break
            kept_train += new_train
            kept_val   += new_val
            if new_train or new_val:
                print(f"Checkpoint split: {len(new_train) + len(new_val)} new dataset(s) assigned, "
                      f"{len(new_train)} to train, {len(new_val)} to val:")
                for p in new_train:
                    print(f"  train: {Path(p).parent.name}")
                for p in new_val:
                    print(f"    val: {Path(p).parent.name}")

        train_paths = kept_train
        val_paths   = kept_val
        split_target_train = len(train_paths)
        split_target_val = len(val_paths)
        print(f"Restored split: {len(train_paths)} train, {len(val_paths)} val datasets.")
    else:
        # Use newest (num_train + num_val) datasets; all_paths is oldest-first.
        # Assign newest num_val to val, next num_train to train.
        target_total = initial_num_train + initial_num_val
        pool = all_paths[-target_total:] if len(all_paths) > target_total else all_paths
        val_paths   = pool[-initial_num_val:]  if initial_num_val > 0 else []
        train_paths = pool[:-initial_num_val]  if initial_num_val > 0 else list(pool)
        train_paths = train_paths[-initial_num_train:] if len(train_paths) > initial_num_train else train_paths
        split_target_train = initial_num_train
        split_target_val = initial_num_val

    if use_separate_validation_folder:
        _train_startup = set(discovered_at_start) if args.data_folder else set(all_paths)
        _val_startup = set(val_discovered_at_start)
        _at = _active_paths(train_paths, split_target_train, _train_startup)
        _av = _active_paths(val_paths, split_target_val, _val_startup)
    else:
        _dset_startup = set(discovered_at_start) if args.data_folder else set(all_paths)
        _at = _active_paths(train_paths, split_target_train, _dset_startup)
        _av = _active_paths(val_paths,   split_target_val,   _dset_startup)
    print(
        f"Dataset split ({split_target_train} train, {split_target_val} val target): "
        f"{len(_at)} train, {len(_av)} val"
    )
    print(f"  Train: {[Path(p).parent.name for p in _at]}")
    if _av:
        print(f"  Val:   {[Path(p).parent.name for p in _av]}")
    print()

    # --- Model + memory diagnostic (must run before dataloaders so we can set batch size) ---
    print("Creating model...")
    if args.use_mamba and not _MAMBA_AVAILABLE:
        print("WARNING: --use_mamba requested but mamba_ssm not installed; falling back to ResBlock3d")
    if args.kernel_sizes is None:
        print("Kernel schedule: default legacy 3x3 across stages")
    else:
        print(f"Kernel schedule: {tuple(args.kernel_sizes)}")
    print(f"Channels schedule: {tuple(args.hidden_dims)}")
    print(f"U-Net levels: {int(args.unet_levels)}")
    print(f"Encoder depth profile: {str(getattr(args, 'encoder_depth_profile', 'baseline'))}")
    model = create_model(
        use_mamba=args.use_mamba,
        input_channels=1,
        hidden_dims=tuple(args.hidden_dims),
        unet_levels=int(args.unet_levels),
        kernel_sizes=tuple(args.kernel_sizes) if args.kernel_sizes is not None else None,
        encoder_depth_profile=str(getattr(args, "encoder_depth_profile", "baseline")),
        encoder_stage_blocks=(
            tuple(args.encoder_stage_blocks)
            if getattr(args, "encoder_stage_blocks", None) is not None
            else None
        ),
        spatial_size=tuple(args.sample_shape),
        deep_reconstruction_head=args.deep_reconstruction_head,
    ).to(device)
    print(f"Encoder stage blocks: {getattr(model.encoder, 'stage_block_schedule', ())}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    weights_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    grads_bytes    = weights_bytes
    adam_bytes     = 2 * weights_bytes
    fixed_bytes    = weights_bytes + grads_bytes + adam_bytes

    S = args.sample_shape
    hidden = tuple(args.hidden_dims)
    n = len(hidden)
    def _fm(b, c, s): return b * c * s[0] * s[1] * s[2] * 4
    # Encoder scales [0..n-1] + decoder mirrors [n-2..0]
    _scales = list(range(n)) + list(range(n - 2, -1, -1))
    act_per_sample = 2 * sum(
        _fm(1, hidden[i], [d // (2 ** i) for d in S])
        for i in _scales
    )
    io_per_sample  = 2 * int(np.prod(S)) * 4
    per_sample_var = act_per_sample + io_per_sample

    # Peak-overhead factor: empirically calibrated from OOM crashes on M4 24 GB.
    #   batch=7 (with grad checkpointing): MPS allocated 24.09 GiB at crash.
    #   formula raw per-sample: 1.433 GB.  Observed: 24.09/7 = 3.44 GB → ratio 2.40x.
    # Use 2.5 for a small margin above observed.
    PEAK_FACTOR = 2.5
    per_sample_peak = per_sample_var * PEAK_FACTOR

    def _total(bs): return fixed_bytes + bs * per_sample_peak

    mem_info   = get_memory_info(device)
    total_mem  = mem_info["total_bytes"]
    if total_mem is None:
        raise RuntimeError("Unable to determine total device memory (mem_info['total_bytes'] is None)")

    # MPS can exceed reported RAM via unified memory.  The actual ceiling
    # (PYTORCH_MPS_HIGH_WATERMARK_RATIO default) is ~1.17 × reported RAM.
    # Observed: 30.19 GiB limit on a 25.77 GB device → ratio 1.172.
    MPS_WATERMARK = 1.172 if device.type == "mps" else 1.0
    mps_ceiling   = total_mem * MPS_WATERMARK

    # "other allocations" (Python, CPU tensors, MPS driver bookkeeping).
    # Observed stable at ~6 GB across all OOM crashes.
    OTHER_ALLOCS  = 6 * 1024**3
    available     = mps_ceiling - OTHER_ALLOCS
    safe_limit    = available * 0.85   # 15% headroom within available MPS model budget

    # Respect the requested batch size.  Compute the max safe size only for diagnostics
    # and to clamp obviously unsafe requests.
    safe_max_bs = 1
    while _total(safe_max_bs + 1) < safe_limit:
        safe_max_bs += 1

    requested_batch_size_raw = getattr(args, "batch_size", None)
    if requested_batch_size_raw is None:
        requested_batch_size = _default_batch_size_for_model(n_params)
        print(
            f"Default batch size selected from model size ({n_params:,} params): {requested_batch_size}"
        )
    else:
        requested_batch_size = max(1, int(requested_batch_size_raw))

    if requested_batch_size > safe_max_bs:
        print(
            f"WARNING: requested batch size {requested_batch_size} exceeds estimated safe max "
            f"{safe_max_bs}; clamping to {safe_max_bs}."
        )
        batch_size = safe_max_bs
    else:
        batch_size = requested_batch_size

    pressure   = "OK" if _total(batch_size) < safe_limit else "PRESSURE"
    current_gb = _total(batch_size) / 1e9

    print(f"""Memory estimate (batch={batch_size}):
  Weights:              {weights_bytes/1e9:.2f} GB
  Gradients:            {grads_bytes/1e9:.2f} GB
  Adam states:          {adam_bytes/1e9:.2f} GB
  Activations+temps:    {batch_size * per_sample_peak/1e9:.2f} GB  ({batch_size} x {per_sample_peak/1e9:.2f} GB/sample, {PEAK_FACTOR}x peak factor)
  -------------------------------------------------
  Total estimated:      {current_gb:.2f} GB  [{pressure}]
  MPS ceiling:          {mps_ceiling/1e9:.2f} GB  ({MPS_WATERMARK}x reported RAM)
  Other allocations:    ~{OTHER_ALLOCS/1e9:.2f} GB  (Python + MPS driver)
  Available for model:  {available/1e9:.2f} GB  (safe limit: {safe_limit/1e9:.2f} GB)
  Headroom:             {(safe_limit - _total(batch_size))/1e9:.2f} GB
    Safe max batch size:  {safe_max_bs}
    Using batch size:     {batch_size}""", flush=True)
    print()

    # macOS multiprocessing workers crash with zarr + MPS (exit code 255).
    # Use num_workers=0 (main-process loading) on macOS; workers only on Linux.
    import platform
    _num_workers = 0 if platform.system() == "Darwin" else min(4, os.cpu_count() or 1)

    loader_kwargs = dict(
        batch_size=batch_size,
        sample_shape=tuple(args.sample_shape),
        num_workers=_num_workers,
        pin_memory=(device.type == "cuda"),
        epoch_samples=(int(args.epoch_samples) if getattr(args, "epoch_samples", None) is not None else None),
        normalize=True,
        target_std=1.0,
        trace_mask_ratio=0.07,
        cluster_prob=0.8,
        input_extrema_prob=float(args.input_extrema_prob),
        input_sparse_keep_prob=float(args.input_sparse_keep_prob),
        input_decimate_trilinear_prob=float(args.input_decimate_trilinear_prob),
        array_keys=args.array_keys,
        geologic_score_sampling=(not args.disable_geologic_score_sampling),
        geologic_score_min=float(args.geologic_score_min),
        geologic_score_key_candidates=args.geologic_score_keys,
        geologic_points_json_name=args.geologic_points_json_name,
        geologic_val_center_json_name=args.geologic_val_center_json_name,
        geologic_target_points=int(args.geologic_target_points),
        geologic_candidate_count=int(args.geologic_candidate_count),
        geologic_candidate_probes=int(args.geologic_candidate_probes),
        geologic_dist_thresh_start=int(args.geologic_dist_thresh_start),
        geologic_dist_thresh_floor=int(args.geologic_dist_thresh_floor),
    )

    print(
        "Input masking strategy probs (extrema/sparse/decimate): "
        f"{float(args.input_extrema_prob):.4f}/"
        f"{float(args.input_sparse_keep_prob):.4f}/"
        f"{float(args.input_decimate_trilinear_prob):.4f}"
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = _build_criterion(args)
    scaler = create_grad_scaler(device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    reporting_lpips = LPIPSLoss(enabled=True, net=str(getattr(args, "mc_lpips_net", "alex"))).to(device)
    reporting_lpips.eval()
    for p in reporting_lpips.parameters():
        p.requires_grad_(False)
    _print_loss_and_backprop_summary(args, cli_provided, backprop_defaults, scaler)
    thermal_guard = ThermalGuard(
        max_c=args.thermal_max_c,
        cooldown_sec=args.thermal_cooldown_sec,
        check_every_batches=args.thermal_check_every_batches,
        output_dir=output_dir,
        pressure_trip_level=args.thermal_pressure_trip_level,
    )
    _print_thermal_monitor_status(args.thermal_max_c, args.thermal_pressure_trip_level)

    # TensorBoard writer — view with: tensorboard --logdir checkpoints/runs
    tb_log_dir = output_dir / "runs"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"  Launch viewer: uv run tensorboard --logdir {tb_log_dir}")

    start_epoch = 0
    if args.resume:
        resume_path = _resolve_resume_checkpoint_path(args.resume)
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.load_state_dict(checkpoint["ema_state"])
        start_epoch = checkpoint["epoch"] + 1
        ds_idx_done = checkpoint.get("ds_idx", -1)
        if ds_idx_done >= 0:
            print(f"  Partial epoch {start_epoch}: completed datasets 0..{ds_idx_done}")
            print("  Note: epoch restarts from the beginning (datasets are randomly ordered)")
        print(f"  Continuing from epoch {start_epoch + 1}")

    scheduler = _build_lr_scheduler(optimizer, args)
    # Do not step scheduler here; stepping before any optimizer.step() triggers
    # a PyTorch warning and can skip the first scheduled LR value.
    if scheduler is not None and start_epoch > 0:
        scheduler.last_epoch = start_epoch - 1

    print("  Epoch sizing:")
    if args.train_batches_per_epoch is not None:
        print(
            f"    train: fixed {args.train_batches_per_epoch} batches "
            "(dataset list is fixed within each epoch; refreshed at epoch start)"
        )
    else:
        print("    train: all batches from merged train loader")
    if args.val_batches_per_epoch is not None:
        print(f"    val: fixed {args.val_batches_per_epoch} batches")
    if args.test_batches_per_epoch is not None:
        print(f"    test: fixed {args.test_batches_per_epoch} batches")
    else:
        print("    val: all batches from val loaders")

    print("\nStarting training...")
    training_start = time.monotonic()
    prev_val_heuristic: float | None = None
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_stamp = datetime.now().strftime("%Y-%-m-%-d %H:%M:%S")
        print(f"\nEpoch {epoch + 1}/{args.epochs} | LR: {current_lr:.3e} | {epoch_stamp}")
        _dset: set[str] = set()
        _dset_train: set[str] = set()
        _dset_val: set[str] = set()

        # Re-scan once per epoch, prune oldest on disk to fixed target count,
        # then keep the active train/val set fixed until next epoch.
        if args.data_folder:
            discovered = _discover_zarr_paths(args.data_folder, args.dataset_glob)
            if use_separate_validation_folder and validation_subfolder is not None:
                val_root = validation_subfolder.resolve()
                discovered_train = [
                    p for p in discovered
                    if not Path(p).resolve().is_relative_to(val_root)
                ]
                # Prune only the training folder; the validation subfolder is
                # managed externally and never pruned.
                discovered_train = _prune_oldest_to_target(
                    args.data_folder,
                    args.dataset_glob,
                    discovered_train,
                    split_target_train,
                )
                discovered_val = _discover_zarr_paths(str(validation_subfolder), args.dataset_glob)

                known_train = set(train_paths)
                train_paths = train_paths + [p for p in discovered_train if p not in known_train]
                known_val = set(val_paths)
                val_paths = val_paths + [p for p in discovered_val if p not in known_val]
                _dset_train = set(discovered_train)
                _dset_val = set(discovered_val)

                if backup_dir_path is not None:
                    backup_candidates = [p for p in (discovered_train + discovered_val) if p not in backed_up_dataset_paths]
                    print(
                        "Backup check: "
                        f"epoch={epoch + 1} "
                        f"data_root={Path(args.data_folder).resolve()} "
                        f"backup_root={backup_dir_path.resolve()} "
                        f"candidates={len(backup_candidates)}"
                    )
                    if backup_candidates:
                        copied = _backup_dataset_folders(
                            backup_candidates,
                            data_root=Path(args.data_folder),
                            backup_root=backup_dir_path,
                            epoch=epoch + 1,
                        )
                        backed_up_dataset_paths.update(copied)
                    else:
                        print(f"Backup status: epoch={epoch + 1} no new datasets to copy this epoch")
                else:
                    print(f"Backup status: epoch={epoch + 1} backup_dir_path is None; skipping dataset backup")
            else:
                keep_total = split_target_train + split_target_val
                discovered = _prune_oldest_to_target(
                    args.data_folder,
                    args.dataset_glob,
                    discovered,
                    keep_total,
                )
                train_paths, val_paths = _update_split(
                    discovered, train_paths, val_paths, split_target_train, split_target_val
                )
                _dset = set(discovered)

                if backup_dir_path is not None:
                    backup_candidates = [p for p in discovered if p not in backed_up_dataset_paths]
                    print(
                        "Backup check: "
                        f"epoch={epoch + 1} "
                        f"data_root={Path(args.data_folder).resolve()} "
                        f"backup_root={backup_dir_path.resolve()} "
                        f"candidates={len(backup_candidates)}"
                    )
                    if backup_candidates:
                        copied = _backup_dataset_folders(
                            backup_candidates,
                            data_root=Path(args.data_folder),
                            backup_root=backup_dir_path,
                            epoch=epoch + 1,
                        )
                        backed_up_dataset_paths.update(copied)
                    else:
                        print(f"Backup status: epoch={epoch + 1} no new datasets to copy this epoch")
                else:
                    print(f"Backup status: epoch={epoch + 1} backup_dir_path is None; skipping dataset backup")
        else:
            _dset = {p for p in (train_paths + val_paths) if Path(p).parent.exists()}

        if args.data_folder and use_separate_validation_folder:
            active_train = _active_paths(train_paths, split_target_train, _dset_train)
            active_val = _active_paths(val_paths, split_target_val, _dset_val)
        else:
            active_train = _active_paths(train_paths, split_target_train, _dset)
            active_val   = _active_paths(val_paths,   split_target_val,   _dset)
        print(f"Dataset split ({split_target_train} train, {split_target_val} val target): "
              f"{len(active_train)} train, {len(active_val)} val")
        print(f"  Train: {[Path(p).parent.name for p in active_train]}")
        if active_val:
            print(f"  Val:   {[Path(p).parent.name for p in active_val]}")
        train_loader = None
        val_loaders = []
        test_loaders: list[tuple[str, DataLoader]] = []

        if args.train_batches_per_epoch is None:
            train_loader, val_loaders, test_loaders = _build_loaders(
                active_train,
                active_val,
                loader_kwargs,
                train_batches_per_epoch=args.train_batches_per_epoch,
                val_batches_per_epoch=args.val_batches_per_epoch,
                test_batches_per_epoch=args.test_batches_per_epoch,
                real_train_paths=list(getattr(args, "real_train_paths", None) or []),
                real_test_paths=list(getattr(args, "real_test_paths", None) or []),
                real_epoch_samples=getattr(args, "real_epoch_samples", None),
            )
            if train_loader is None:
                print("  WARNING: No usable training datasets this epoch; skipping.")
                continue

            train_details = train_epoch(
                model, train_loader, optimizer, criterion, device,
                scaler=scaler, writer=writer, epoch=epoch, output_dir=output_dir,
                train_paths=train_paths, val_paths=val_paths,
                thermal_guard=thermal_guard,
                grad_accum_steps=args.grad_accum_steps,
                grad_clip_norm=args.grad_clip_norm,
                ema=ema,
                ema_update_every=args.ema_update_every,
                return_details=True,
                reporting_lpips=reporting_lpips,
                log_image_this_epoch=(tb_image_epochs_set is None or (epoch + 1) in tb_image_epochs_set),
            )
            if not isinstance(train_details, dict):
                raise RuntimeError("train_epoch(return_details=True) returned non-dict details")
            train_loss = float(train_details["loss"])
            train_metrics = {
                "mse": float(train_details["mse"]),
                "pmse": float(train_details["pmse"]),
                "mae": float(train_details["mae"]),
                "lpips": float(train_details["lpips"]),
                "gdl": float(train_details["gdl"]),
            }
        else:
            target_batches = max(1, int(args.train_batches_per_epoch))
            batches_done = 0
            weighted_loss_sum = 0.0
            weighted_mse_sum = 0.0
            weighted_pmse_sum = 0.0
            weighted_mae_sum = 0.0
            weighted_lpips_sum = 0.0
            weighted_gdl_sum = 0.0
            pending_chunk_reload = False

            while batches_done < target_batches:
                _reload_t0 = time.monotonic()
                train_loader, val_loaders, test_loaders = _build_loaders(
                    active_train,
                    active_val,
                    loader_kwargs,
                    train_batches_per_epoch=args.train_batches_per_epoch,
                    val_batches_per_epoch=args.val_batches_per_epoch,
                    test_batches_per_epoch=args.test_batches_per_epoch,
                    real_train_paths=list(getattr(args, "real_train_paths", None) or []),
                    real_test_paths=list(getattr(args, "real_test_paths", None) or []),
                    real_epoch_samples=getattr(args, "real_epoch_samples", None),
                )
                _reload_elapsed = time.monotonic() - _reload_t0
                if pending_chunk_reload:
                    print(f"  Reloaded train/val loaders in {_reload_elapsed:.2f}s")
                    pending_chunk_reload = False
                if train_loader is None:
                    print("  WARNING: No usable training datasets this epoch; skipping remaining batches.")
                    break

                remaining = target_batches - batches_done

                details = train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    scaler=scaler, writer=writer, epoch=epoch, output_dir=output_dir,
                    train_paths=train_paths, val_paths=val_paths,
                    thermal_guard=thermal_guard,
                    grad_accum_steps=args.grad_accum_steps,
                    grad_clip_norm=args.grad_clip_norm,
                    ema=ema,
                    ema_update_every=args.ema_update_every,
                    max_batches=remaining,
                    return_details=True,
                    reporting_lpips=reporting_lpips,
                    log_image_this_epoch=(tb_image_epochs_set is None or (epoch + 1) in tb_image_epochs_set),
                )
                if not isinstance(details, dict):
                    raise RuntimeError("train_epoch(return_details=True) returned non-dict details")
                chunk_batches = int(details["batches_processed"])
                if chunk_batches <= 0:
                    print("  WARNING: train epoch chunk processed 0 batches; stopping epoch early.")
                    break

                weighted_loss_sum += float(details["loss"]) * chunk_batches
                weighted_mse_sum += float(details["mse"]) * chunk_batches
                weighted_pmse_sum += float(details["pmse"]) * chunk_batches
                weighted_mae_sum += float(details["mae"]) * chunk_batches
                weighted_lpips_sum += float(details["lpips"]) * chunk_batches
                weighted_gdl_sum += float(details["gdl"]) * chunk_batches
                batches_done += chunk_batches

                if not bool(details["reload_requested"]):
                    break

                pending_chunk_reload = True

            train_loss = float(weighted_loss_sum / max(1, batches_done))
            train_metrics = {
                "mse": float(weighted_mse_sum / max(1, batches_done)),
                "pmse": float(weighted_pmse_sum / max(1, batches_done)),
                "mae": float(weighted_mae_sum / max(1, batches_done)),
                "lpips": float(weighted_lpips_sum / max(1, batches_done)),
                "gdl": float(weighted_gdl_sum / max(1, batches_done)),
            }

        if (
            writer is not None
            and train_loader is not None
            and (tb_image_epochs_set is None or (epoch + 1) in tb_image_epochs_set)
        ):
            _log_per_dataset_figures(
                model, train_loader, device, writer, epoch, train_loss
            )

        using_ema = ema is not None
        if using_ema:
            ema.store(model)
            ema.copy_to(model)
        val_details = validate(
            model, val_loaders, criterion, device,
            writer=writer, epoch=epoch, thermal_guard=thermal_guard,
            max_batches=args.val_batches_per_epoch,
            return_details=True,
            reporting_lpips=reporting_lpips,
            log_image_this_epoch=(tb_image_epochs_set is None or (epoch + 1) in tb_image_epochs_set),
        )
        if not isinstance(val_details, dict):
            raise RuntimeError("validate(return_details=True) returned non-dict details")
        val_loss = float(val_details["loss"])
        val_metrics = {
            "mse": float(val_details["mse"]),
            "pmse": float(val_details["pmse"]),
            "mae": float(val_details["mae"]),
            "lpips": float(val_details["lpips"]),
            "gdl": float(val_details["gdl"]),
        }
        if using_ema:
            ema.restore(model)

        # --- Test evaluation (real .npy datasets; no effect on LR or checkpoints) ---
        test_metrics: dict[str, float] | None = None
        if test_loaders:
            test_details = validate(
                model, test_loaders, criterion, device,
                writer=writer, epoch=epoch, thermal_guard=None,
                max_batches=args.test_batches_per_epoch,
                return_details=True,
                reporting_lpips=reporting_lpips,
                log_image_this_epoch=(tb_image_epochs_set is None or (epoch + 1) in tb_image_epochs_set),
            )
            if isinstance(test_details, dict):
                test_metrics = {
                    "mse":  float(test_details["mse"]),
                    "pmse": float(test_details["pmse"]),
                    "mae":  float(test_details["mae"]),
                    "lpips": float(test_details["lpips"]),
                    "gdl":  float(test_details["gdl"]),
                }

        # Log scalar losses to TensorBoard
        writer.add_scalar("loss/train", train_loss, global_step=epoch + 1)
        writer.add_scalar("lr", current_lr, global_step=epoch + 1)
        if val_loaders:
            writer.add_scalar("loss/val", val_loss, global_step=epoch + 1)
        if test_metrics is not None:
            writer.add_scalar("loss/test", float(test_details["loss"]), global_step=epoch + 1)  # type: ignore[possibly-undefined]

        if val_loaders:
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Train Loss: {train_loss:.4f}")
        print(
            "Component Metrics | "
            f"train(mse={train_metrics['mse']:.6f}, pmse={train_metrics['pmse']:.6f}, "
            f"mae={train_metrics['mae']:.6f}, lpips={train_metrics['lpips']:.6f}, gdl={train_metrics['gdl']:.6f})"
        )
        if val_loaders:
            print(
                "Component Metrics | "
                f"val(mse={val_metrics['mse']:.6f}, pmse={val_metrics['pmse']:.6f}, "
                f"mae={val_metrics['mae']:.6f}, lpips={val_metrics['lpips']:.6f}, gdl={val_metrics['gdl']:.6f})"
            )
        if test_metrics is not None:
            print(
                "Component Metrics | "
                f"test(mse={test_metrics['mse']:.6f}, pmse={test_metrics['pmse']:.6f}, "
                f"mae={test_metrics['mae']:.6f}, lpips={test_metrics['lpips']:.6f}, gdl={test_metrics['gdl']:.6f})"
            )

        val_heuristic = _fixed_validation_heuristic(val_metrics)
        heuristic_increased = False
        if math.isfinite(val_heuristic):
            writer.add_scalar("heuristic/val_fixed", float(val_heuristic), global_step=epoch + 1)
            print(f"Validation fixed heuristic (2*mae + mse + lpips): {val_heuristic:.8f}")
            if prev_val_heuristic is not None and math.isfinite(prev_val_heuristic) and val_heuristic > prev_val_heuristic:
                heuristic_increased = True
                print(
                    "Validation fixed heuristic increased "
                    f"({prev_val_heuristic:.8f} -> {val_heuristic:.8f}); scheduling LR halving."
                )
            prev_val_heuristic = val_heuristic

        if (epoch + 1) % 1 == 0:
            csv_path = Path("/Users/donaldpg/synthoseis-pretrain-v2/checkpoints/epoch_component_metrics.csv")
            _append_component_metrics_csv_row(
                csv_path=csv_path,
                tb_log_dir=tb_log_dir,
                epoch=epoch + 1,
                args=args,
                n_params=n_params,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
            print(f"Appended component metrics CSV row: {csv_path}")

        # End-of-epoch versioned checkpoint (never overwritten)
        epoch_ckpt = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
        _save_checkpoint(epoch_ckpt, model, optimizer, scaler, epoch,
                         train_loss=train_loss, val_loss=val_loss,
                         train_paths=train_paths, val_paths=val_paths,
                         ema_state=ema.state_dict() if ema is not None else None)
        print(f"Saved checkpoint: {epoch_ckpt}")

        if (epoch + 1) == int(args.epochs):
            final_ckpt = output_dir / "checkpoint_final_model.pt"
            _save_checkpoint(final_ckpt, model, optimizer, scaler, epoch,
                             train_loss=train_loss, val_loss=val_loss,
                             train_paths=train_paths, val_paths=val_paths,
                             ema_state=ema.state_dict() if ema is not None else None)
            print(f"Saved final resumable checkpoint: {final_ckpt}")

        if _maybe_update_best_val_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_paths=train_paths,
            val_paths=val_paths,
            ema_state=ema.state_dict() if ema is not None else None,
        ):
            print(f"Updated best validation checkpoint: {output_dir / 'best_val_epoch.pt'}")

        if scheduler is not None:
            scheduler.step()

        if heuristic_increased:
            old_lr, new_lr = _scale_optimizer_lr(optimizer, scale=0.5, scheduler=scheduler)
            print(f"Applied LR backoff x0.5 due to heuristic regression: {old_lr:.3e} -> {new_lr:.3e}")

        # --- timing ---
        now = time.monotonic()
        epochs_done = epoch + 1 - start_epoch
        epochs_left = args.epochs - (epoch + 1)
        epoch_elapsed = now - epoch_start
        total_elapsed = now - training_start
        avg_per_epoch = total_elapsed / epochs_done
        remaining_secs = avg_per_epoch * epochs_left

        def _fmt_duration(secs: float) -> str:
            secs = int(secs)
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h}h {m:02d}m {s:02d}s"
            if m:
                return f"{m}m {s:02d}s"
            return f"{s}s"

        eta_dt = datetime.now() + timedelta(seconds=remaining_secs)
        eta_str = eta_dt.strftime("%d %b %Y %H:%M")
        print(
            f"Epoch time: {_fmt_duration(epoch_elapsed)} | "
            f"Elapsed: {_fmt_duration(total_elapsed)} | "
            f"Remaining: {_fmt_duration(remaining_secs)} | "
            f"ETA: {eta_str}"
        )

    final_path = output_dir / "final_model.pt"
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)
        torch.save(model.state_dict(), final_path)
        ema.restore(model)
        raw_final_path = output_dir / "final_model_raw.pt"
        torch.save(model.state_dict(), raw_final_path)
        print(f"Saved raw non-EMA model: {raw_final_path}")
    else:
        torch.save(model.state_dict(), final_path)
    writer.close()
    print(f"Training complete. Final model: {final_path}")

