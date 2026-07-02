#!/usr/bin/env python3
"""Evaluate saved checkpoints on real seismic volumes and export inference metrics."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from synthoseis_pre_train.gpu_utils import autocast_context, get_default_device, print_device_summary
from synthoseis_pre_train.losses import LPIPSLoss, compute_pmse_loss, gradient_difference_loss_3d
from synthoseis_pre_train.models import create_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = ROOT / "checkpoints"
EPOCH_COMPONENT_METRICS_CSV = CHECKPOINTS_ROOT / "epoch_component_metrics.csv"
REAL_SEISMIC_COMPONENT_METRICS_CSV = CHECKPOINTS_ROOT / "real_seismic_component_metrics.csv"
REAL_SEISMIC_DIR = Path("/Users/donaldpg/synthoseis/real_data")
FAKE_DATA_TEST_DIR = Path("/Users/donaldpg/synthoseis/fake_data/test")
REAL_SEISMIC_DIRS = [REAL_SEISMIC_DIR, FAKE_DATA_TEST_DIR]
PATCH_EDGE = 128

TOP_10_BEST_CHECKPOINTS = [
    "checkpoints/checkpoint_copilot_5/checkpoint_epoch_0034.pt",
    "checkpoints/checkpoint_copilot_2/checkpoint_epoch_0029.pt",
    "checkpoints/checkpoint_copilot_1/checkpoint_epoch_0033.pt",
    "checkpoints/checkpoint_copilot_4/checkpoint_epoch_0035.pt",
    "checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001/checkpoint_epoch_0008.pt",
]

OUTPUT_COLUMNS = [
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
    "infer mse",
    "infer pmse",
    "infer mae/L1",
    "infer LPIPS",
    "infer gdl",
    "seismic_dataset",
]


@dataclass(frozen=True)
class CheckpointMeta:
    checkpoint_path: Path
    epoch: int
    tensorboard_folder: str
    unet_levels: int
    hidden_dims: tuple[int, ...]
    kernel_schedule: tuple[int, ...]
    encoder_depth_profile: str
    encoder_stage_blocks: tuple[int, ...] | None
    lr: str
    mse_weight: float
    pmse_weight: float
    mae_weight: float
    lpips_weight: float
    tv_weight: float
    gdl_weight: float
    lpips_net: str


def _checkpoint_epoch_from_name(path: Path) -> int:
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"Unable to parse epoch from checkpoint filename: {path}")
    return int(match.group(1))


def _normalize_rel_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            pass
    return path.as_posix().strip("/")


def _discover_checkpoint_paths(
    source_checkpoints: list[str],
    exclude_input_checkpoints: bool = True,
) -> list[Path]:
    source_paths = [ROOT / rel for rel in source_checkpoints]
    source_dirs = sorted({path.parent for path in source_paths})
    excluded = {path.resolve() for path in source_paths} if exclude_input_checkpoints else set()

    discovered: list[Path] = []
    for directory in source_dirs:
        for checkpoint in sorted(directory.glob("checkpoint_epoch_*.pt")):
            if checkpoint.resolve() in excluded:
                continue
            discovered.append(checkpoint)

    return sorted(discovered, key=lambda p: (p.parent.as_posix(), _checkpoint_epoch_from_name(p)))


def _parse_int_list(value: str, fallback_len: int | None = None, fallback_value: int = 3) -> tuple[int, ...]:
    tokens = [token for token in str(value).split() if token]
    if tokens:
        return tuple(int(token) for token in tokens)
    if fallback_len is None:
        return tuple()
    return tuple(fallback_value for _ in range(fallback_len))


def _parse_encoder_depth_profile(value: str) -> tuple[str, tuple[int, ...] | None]:
    text = str(value or "").strip()
    if not text:
        return "baseline", None
    if "-" not in text:
        return text, None
    profile, _, tail = text.partition("-")
    profile_name = profile.strip() or "baseline"
    block_tokens = [tok for tok in tail.strip().split() if tok]
    if not block_tokens:
        return profile_name, None
    try:
        blocks = tuple(int(tok) for tok in block_tokens)
    except ValueError:
        blocks = None
    return profile_name, blocks


def _csv_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _build_epoch_metrics_index(csv_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")

    index: dict[tuple[str, int], dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tb_folder = str(row.get("tensorboard folder", "")).strip()
            epoch_raw = str(row.get("epoch", "")).strip()
            if not tb_folder or not epoch_raw:
                continue
            try:
                epoch = int(float(epoch_raw))
            except ValueError:
                continue
            key = (_normalize_rel_path(tb_folder), epoch)
            # Later rows win so repeated rows keep the newest values.
            index[key] = row
    return index


def _resolve_checkpoint_meta(
    checkpoint_path: Path,
    metrics_index: dict[tuple[str, int], dict[str, str]],
) -> CheckpointMeta:
    epoch = _checkpoint_epoch_from_name(checkpoint_path)
    folder_rel = _normalize_rel_path(checkpoint_path.parent)
    tb_folder_rel = f"{folder_rel}/runs"
    row_key = (_normalize_rel_path(tb_folder_rel), epoch)
    row = metrics_index.get(row_key)
    if row is None:
        raise KeyError(
            "Missing epoch metadata in epoch_component_metrics.csv for "
            f"{checkpoint_path} (tensorboard folder={tb_folder_rel}, epoch={epoch})"
        )

    unet_levels = int(float(row.get("unet_levels", "4")))
    hidden_dims = _parse_int_list(row.get("hidden_dims (as a space-delimited string)", ""))
    kernel_schedule = _parse_int_list(
        row.get("kernel_schedule (as a space-delimited string)", ""),
        fallback_len=len(hidden_dims),
        fallback_value=3,
    )
    profile_value = str(row.get("encoder_depth_profile (profile - stage blocks)", "")).strip()
    profile_name, stage_blocks = _parse_encoder_depth_profile(profile_value)
    if not profile_value:
        profile_value = profile_name

    return CheckpointMeta(
        checkpoint_path=checkpoint_path,
        epoch=epoch,
        tensorboard_folder=tb_folder_rel,
        unet_levels=unet_levels,
        hidden_dims=hidden_dims,
        kernel_schedule=kernel_schedule,
        encoder_depth_profile=profile_value,
        encoder_stage_blocks=stage_blocks,
        lr=str(row.get("lr", "")),
        mse_weight=_csv_float(row, "mse_weight", 0.0),
        pmse_weight=_csv_float(row, "pmse_weight", 0.0),
        mae_weight=_csv_float(row, "mae_weight", 0.0),
        lpips_weight=_csv_float(row, "lpips_weight", 0.0),
        tv_weight=_csv_float(row, "tv_weight", 0.0),
        gdl_weight=_csv_float(row, "gdl_weight", 0.0),
        lpips_net="alex",
    )


def _load_real_seismic_paths(seis_dirs: list[Path]) -> list[Path]:
    """Return sorted .npy paths from one or more directories (missing dirs are skipped)."""
    found: list[Path] = []
    for seis_dir in seis_dirs:
        if not seis_dir.exists():
            print(f"WARNING: real seismic directory not found, skipping: {seis_dir}")
            continue
        found.extend(path for path in seis_dir.iterdir() if path.suffix == ".npy")
    return sorted(set(found))


def _iter_128_cubes(volume: np.ndarray, edge: int = PATCH_EDGE):
    shape = np.array(volume.shape, dtype=int)
    counts = shape // int(edge)
    nx, ny, nz = (int(counts[0]), int(counts[1]), int(counts[2]))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x0 = ix * edge
                y0 = iy * edge
                z0 = iz * edge
                center = (x0 + edge // 2, y0 + edge // 2, z0 + edge // 2)
                yield volume[x0:x0 + edge, y0:y0 + edge, z0:z0 + edge], center


def _normalize_with_volume_std(seismic_data: np.ndarray, volume_std: float, target_std: float = 1.0) -> np.ndarray:
    data = np.asarray(seismic_data, dtype=np.float32)
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    if np.isclose(data_min, 0.0) and np.isclose(data_max, 1.0):
        data = data - 0.5

    if float(volume_std) > 0.0:
        normalized = data / float(volume_std) * float(target_std)
    else:
        normalized = data
    return normalized.astype(np.float32, copy=False)


def _volume_stats(seismic_data: np.ndarray) -> tuple[float, float, float, float]:
    data64 = np.asarray(seismic_data, dtype=np.float64)
    return (
        float(np.min(data64)),
        float(np.mean(data64)),
        float(np.max(data64)),
        float(np.std(data64)),
    )


def _extrema_spacing_stats_z(vol: np.ndarray) -> tuple[float, float, float, float, float, float, float] | None:
    """Return spacing stats between consecutive peak/trough indices along Z."""
    if vol.ndim != 3:
        return None
    z_dim, x_dim, y_dim = vol.shape
    if z_dim <= 2:
        return None

    is_peak = (vol[1:-1, :, :] > vol[:-2, :, :]) & (vol[1:-1, :, :] > vol[2:, :, :])
    is_trough = (vol[1:-1, :, :] < vol[:-2, :, :]) & (vol[1:-1, :, :] < vol[2:, :, :])
    peaks_troughs = is_peak | is_trough

    distances: list[np.ndarray] = []
    for xi in range(x_dim):
        for yi in range(y_dim):
            idx = np.flatnonzero(peaks_troughs[:, xi, yi]) + 1
            if idx.size >= 2:
                distances.append(np.diff(idx))

    if not distances:
        return None

    d = np.concatenate(distances).astype(np.float32, copy=False)
    return (
        float(d.min()),
        float(np.percentile(d, 25.0)),
        float(d.mean()),
        float(np.median(d)),
        float(np.percentile(d, 75.0)),
        float(np.percentile(d, 95.0)),
        float(d.max()),
    )


def _build_inference_threshold(volume_shape: tuple, edge: int = PATCH_EDGE, target_examples: int = 15) -> float:
    """Return a per-cube Bernoulli keep-probability that targets ~target_examples inferences."""
    counts = [int(dim) // int(edge) for dim in volume_shape]
    total = max(1, int(counts[0]) * int(counts[1]) * int(counts[2]))
    return min(1.0, float(target_examples) / float(total))


def _load_existing_results_index(csv_path: Path) -> set[tuple[str, int, str]]:
    """Return set of (tensorboard_folder, epoch, seismic_dataset) already in the output CSV."""
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, int, str]] = set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tb = _normalize_rel_path(str(row.get("tensorboard folder", "")).strip())
                epoch_raw = str(row.get("epoch", "")).strip()
                dataset = str(row.get("seismic_dataset", "")).strip()
                if not tb or not epoch_raw or not dataset:
                    continue
                try:
                    epoch = int(float(epoch_raw))
                except ValueError:
                    continue
                keys.add((tb, epoch, dataset))
    except Exception:
        pass
    return keys


def _new_metric_totals() -> dict[str, float]:
    return {
        "count": 0.0,
        "mse": 0.0,
        "pmse": 0.0,
        "mae": 0.0,
        "lpips": 0.0,
        "gdl": 0.0,
    }


def _finalize_totals(totals: dict[str, float]) -> dict[str, float]:
    count = max(1.0, float(totals.get("count", 0.0)))
    return {
        "count": float(totals.get("count", 0.0)),
        "mse": float(totals.get("mse", 0.0)) / count,
        "pmse": float(totals.get("pmse", 0.0)) / count,
        "mae": float(totals.get("mae", 0.0)) / count,
        "lpips": float(totals.get("lpips", 0.0)) / count,
        "gdl": float(totals.get("gdl", 0.0)) / count,
    }


def _load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")


def _has_deep_reconstruction_head(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("head.0.") for key in state_dict)


def _build_model_from_meta(meta: CheckpointMeta, state_dict: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    model = create_model(
        use_mamba=False,
        input_channels=1,
        hidden_dims=tuple(meta.hidden_dims),
        unet_levels=int(meta.unet_levels),
        kernel_sizes=tuple(meta.kernel_schedule) if meta.kernel_schedule else None,
        encoder_depth_profile=meta.encoder_depth_profile.split("-", 1)[0].strip(),
        encoder_stage_blocks=meta.encoder_stage_blocks,
        spatial_size=(PATCH_EDGE, PATCH_EDGE, PATCH_EDGE),
        deep_reconstruction_head=_has_deep_reconstruction_head(state_dict),
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _timestamp_from_path(path: Path) -> datetime:
    stats = path.stat()
    if hasattr(stats, "st_birthtime"):
        ts = float(getattr(stats, "st_birthtime"))
    else:
        ts = float(stats.st_mtime)
    return datetime.fromtimestamp(ts)


def _evaluate_dataset(
    model: torch.nn.Module,
    dataset_path: Path,
    device: torch.device,
    lpips_metric: torch.nn.Module,
    target_examples: int = 15,
) -> dict[str, float]:
    volume = np.load(dataset_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D seismic volume in {dataset_path}, got shape={volume.shape}")
    volume_norm = _normalize_with_volume_std(volume, float(np.std(volume, dtype=np.float64)))
    volume_std = float(np.std(volume, dtype=np.float64))
    before_stats = _volume_stats(volume)
    after_stats = _volume_stats(volume_norm)

    cube_counts = tuple((np.array(volume.shape, dtype=int) // PATCH_EDGE).tolist())
    total_cubes = int(cube_counts[0]) * int(cube_counts[1]) * int(cube_counts[2])
    inference_threshold = _build_inference_threshold(volume.shape, edge=PATCH_EDGE, target_examples=target_examples)
    print(
        f"\n  Dataset {dataset_path.name}: shape={tuple(int(v) for v in volume.shape)}, "
        f"subcubes={cube_counts} (total={total_cubes}), "
        f"sample_threshold={inference_threshold:.3f} (~{target_examples} target), "
        f"volume_std={volume_std:.3f}",
        flush=True,
    )
    print(
        "     . min, mean, max, std: "
        f"before ({before_stats[0]:.3f}, {before_stats[1]:.3f}, {before_stats[2]:.3f}, {before_stats[3]:.3f}), "
        f"after ({after_stats[0]:.3f}, {after_stats[1]:.3f}, {after_stats[2]:.3f}, {after_stats[3]:.3f})",
        flush=True,
    )
    spacing_stats = _extrema_spacing_stats_z(np.asarray(volume, dtype=np.float32))
    if spacing_stats is None:
        print("     . peak/trough Z-spacing (idx): unavailable (insufficient extrema)", flush=True)
    else:
        dmin, dp25, dmean, dmedian, dp75, dp95, dmax = spacing_stats
        print(
            "     . peak/trough Z-spacing (idx): "
            f"min={dmin:.1f}, P25={dp25:.1f}, mean={dmean:.2f}, median={dmedian:.1f}, "
            f"P75={dp75:.1f}, P95={dp95:.1f}, max={dmax:.1f}",
            flush=True,
        )

    totals = _new_metric_totals()
    for cube, center in _iter_128_cubes(volume, edge=PATCH_EDGE):
        if np.random.uniform() >= inference_threshold:
            continue
        cube_norm = _normalize_with_volume_std(cube, volume_std)
        cube_t = torch.from_numpy(cube_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast_context(device):
                pred = model(cube_t)
            mse = float(F.mse_loss(pred, cube_t).item())
            pmse = float(compute_pmse_loss(pred, cube_t).item())
            mae = float(F.l1_loss(pred, cube_t).item())
            lpips = float(lpips_metric(pred, cube_t).item())
            gdl = float(gradient_difference_loss_3d(pred, cube_t).item())

        totals["count"] += 1.0
        totals["mse"] += mse
        totals["pmse"] += pmse
        totals["mae"] += mae
        totals["lpips"] += lpips
        totals["gdl"] += gdl

        center_fmt = f"[{center[0]:3d}, {center[1]:3d}, {center[2]:3d}]"
        print(
            f"    center={center_fmt} mse={mse:.3f} pmse={pmse:.3f} mae={mae:.3f} "
            f"lpips={lpips:.3f} gdl={gdl:.3f}",
            flush=True,
        )

    metrics = _finalize_totals(totals)
    print(
        f"     . Dataset average ({int(metrics['count'])} cubes): mse={metrics['mse']:.3f} "
        f"pmse={metrics['pmse']:.3f} mae={metrics['mae']:.3f} "
        f"lpips={metrics['lpips']:.3f} gdl={metrics['gdl']:.3f}",
        flush=True,
    )
    return metrics


def _append_output_rows(csv_path: Path, rows: list[list[str]]) -> None:
    """Append rows to the output CSV, writing header only when the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if write_header:
            writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)


def _metrics_row(
    meta: CheckpointMeta,
    checkpoint_dt: datetime,
    param_count: int,
    metrics: dict[str, float],
    seismic_dataset: str,
) -> list[str]:
    return [
        checkpoint_dt.strftime("%Y-%m-%d"),
        checkpoint_dt.strftime("%H:%M:%S"),
        meta.tensorboard_folder,
        str(int(meta.epoch)),
        str(int(meta.unet_levels)),
        " ".join(str(v) for v in meta.hidden_dims),
        " ".join(str(v) for v in meta.kernel_schedule),
        meta.encoder_depth_profile,
        str(meta.lr),
        f"{float(param_count) / 1_000_000.0:.2f}",
        f"{meta.mse_weight:.6f}",
        f"{meta.pmse_weight:.6f}",
        f"{meta.mae_weight:.6f}",
        f"{meta.lpips_weight:.6f}",
        f"{meta.tv_weight:.6f}",
        f"{meta.gdl_weight:.6f}",
        f"{float(metrics['mse']):.8f}",
        f"{float(metrics['pmse']):.8f}",
        f"{float(metrics['mae']):.8f}",
        f"{float(metrics['lpips']):.8f}",
        f"{float(metrics['gdl']):.8f}",
        seismic_dataset,
    ]


def run(args: argparse.Namespace) -> None:
    metrics_index = _build_epoch_metrics_index(Path(args.epoch_metrics_csv))
    checkpoint_paths = _discover_checkpoint_paths(
        TOP_10_BEST_CHECKPOINTS,
        exclude_input_checkpoints=bool(args.exclude_input_checkpoints),
    )
    seismic_dirs = [Path(d) for d in args.real_seismic_dirs]
    seismic_paths = _load_real_seismic_paths(seismic_dirs)
    output_csv_path = Path(args.output_csv)
    existing_index = _load_existing_results_index(output_csv_path)

    checkpoint_paths = [p for p in checkpoint_paths if _checkpoint_epoch_from_name(p) % 10 == 0]
    print(f"Discovered {len(checkpoint_paths)} checkpoint epochs across source folders (epoch % 10 == 0 only)")
    print(f"Discovered {len(seismic_paths)} real seismic datasets")
    if existing_index:
        print(f"Found {len(existing_index)} existing result(s) in {output_csv_path} — skipping those combos")

    device = get_default_device(args.device)
    print_device_summary(args.device)

    lpips_metric = LPIPSLoss(enabled=True, net="alex").to(device)
    lpips_metric.eval()
    for param in lpips_metric.parameters():
        param.requires_grad_(False)

    rows_written = 0
    grand_totals = _new_metric_totals()
    for checkpoint_path in checkpoint_paths:
        meta = _resolve_checkpoint_meta(checkpoint_path, metrics_index)

        # Determine which seismic datasets still need inference for this checkpoint.
        tb_norm = _normalize_rel_path(meta.tensorboard_folder)
        seismic_to_run = [
            p for p in seismic_paths
            if (tb_norm, meta.epoch, p.name) not in existing_index
        ]
        if not seismic_to_run:
            print(
                f"\nSkipping {checkpoint_path.relative_to(ROOT)} — "
                "all datasets already computed",
                flush=True,
            )
            continue

        state_dict = _load_checkpoint_state_dict(checkpoint_path, device)
        model = _build_model_from_meta(meta, state_dict, device)
        n_params = sum(p.numel() for p in model.parameters())
        dt = _timestamp_from_path(checkpoint_path)
        checkpoint_totals = _new_metric_totals()

        print("\n#################\n", flush=True)
        print(f"Evaluating {checkpoint_path.relative_to(ROOT)}", flush=True)
        for seismic_path in seismic_to_run:
            metrics = _evaluate_dataset(model, seismic_path, device, lpips_metric)
            cube_count = float(metrics.get("count", 0.0))
            checkpoint_totals["count"] += cube_count
            checkpoint_totals["mse"] += float(metrics["mse"]) * cube_count
            checkpoint_totals["pmse"] += float(metrics["pmse"]) * cube_count
            checkpoint_totals["mae"] += float(metrics["mae"]) * cube_count
            checkpoint_totals["lpips"] += float(metrics["lpips"]) * cube_count
            checkpoint_totals["gdl"] += float(metrics["gdl"]) * cube_count

            grand_totals["count"] += cube_count
            grand_totals["mse"] += float(metrics["mse"]) * cube_count
            grand_totals["pmse"] += float(metrics["pmse"]) * cube_count
            grand_totals["mae"] += float(metrics["mae"]) * cube_count
            grand_totals["lpips"] += float(metrics["lpips"]) * cube_count
            grand_totals["gdl"] += float(metrics["gdl"]) * cube_count

            row = _metrics_row(
                meta=meta,
                checkpoint_dt=dt,
                param_count=n_params,
                metrics=metrics,
                seismic_dataset=seismic_path.name,
            )
            _append_output_rows(output_csv_path, [row])
            rows_written += 1
            # Mark as computed so a re-run within the same process also skips it.
            existing_index.add((tb_norm, meta.epoch, seismic_path.name))
            print(f"  Wrote row to {output_csv_path}", flush=True)

        checkpoint_metrics = _finalize_totals(checkpoint_totals)
        print(
            f"Checkpoint average ({int(checkpoint_metrics['count'])} cubes): "
            f"mse={checkpoint_metrics['mse']:.3f} pmse={checkpoint_metrics['pmse']:.3f} "
            f"mae={checkpoint_metrics['mae']:.3f} lpips={checkpoint_metrics['lpips']:.3f} "
            f"gdl={checkpoint_metrics['gdl']:.3f}",
            flush=True,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grand_metrics = _finalize_totals(grand_totals)
    print(
        f"Overall average ({int(grand_metrics['count'])} cubes): mse={grand_metrics['mse']:.3f} "
        f"pmse={grand_metrics['pmse']:.3f} mae={grand_metrics['mae']:.3f} "
        f"lpips={grand_metrics['lpips']:.3f} gdl={grand_metrics['gdl']:.3f}",
        flush=True,
    )
    if rows_written:
        print(f"Wrote {rows_written} new row(s) to {output_csv_path}")
    else:
        print("No new rows to write — all combinations already computed.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute real seismic inference component metrics across checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cuda, mps, cpu")
    parser.add_argument(
        "--real_seismic_dirs",
        type=str,
        nargs="+",
        default=[str(d) for d in REAL_SEISMIC_DIRS],
        help="One or more directories containing real seismic .npy volumes (default: real_data and fake_data/test)",
    )
    parser.add_argument(
        "--epoch_metrics_csv",
        type=str,
        default=str(EPOCH_COMPONENT_METRICS_CSV),
        help="Path to training epoch component metrics CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(REAL_SEISMIC_COMPONENT_METRICS_CSV),
        help="Destination CSV path for inference component metrics",
    )
    parser.add_argument(
        "--exclude_input_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the exact checkpoint files listed in TOP_10_BEST_CHECKPOINTS (default: enabled)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
