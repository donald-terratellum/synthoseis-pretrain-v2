"""CLI entrypoint for seismic pre-training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synthoseis_pre_train.pretrain import (
    DEFAULT_ARRAY_KEYS,
    DEFAULT_GEOLOGIC_SCORE_KEYS,
    run_training,
)


def _normalize_data_paths(data_paths: list[str], dataset_glob: str) -> list[str]:
    normalized: list[str] = []
    for raw_path in data_paths:
        path = Path(raw_path)
        if path.is_dir() and path.suffix != ".zarr":
            discovered = sorted(
                (str(candidate) for candidate in path.glob(dataset_glob)),
                key=lambda candidate: Path(candidate).parent.stat().st_mtime,
            )
            if discovered:
                normalized.extend(discovered)
                continue
        normalized.append(raw_path)

    return list(dict.fromkeys(normalized))


def _collect_cli_option_names(argv: list[str]) -> set[str]:
    provided: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].strip()
        if name:
            provided.add(name.replace("-", "_"))
    return provided


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Seismic 3D Mamba")
    parser.add_argument("--data_paths", type=str, nargs='*', default=[],
                       help="Explicit zarr paths or parent folders containing zarr datasets (optional if --data_folder provided)")
    parser.add_argument("--data_folder", type=str, default=None,
                       help="Folder scanned each epoch for zarr datasets (enables dynamic discovery)")
    parser.add_argument("--dataset_glob", type=str, default="seismic__*/model_data.zarr",
                       help="Glob pattern relative to --data_folder for zarr discovery (default: seismic__*/model_data.zarr)")
    parser.add_argument("--array_keys", type=str, nargs='+', default=DEFAULT_ARRAY_KEYS,
                       help="One or more 3D array keys inside each Zarr dataset; one is picked randomly per sample")
    parser.add_argument(
        "--disable_geologic_score_sampling",
        action="store_true",
        help="Disable geologic-score-driven center selection and fall back to fully random crop centers.",
    )
    parser.add_argument(
        "--geologic_score_min",
        type=float,
        default=0.5,
        help="Minimum geologic score required for candidate points (default: 0.5).",
    )
    parser.add_argument(
        "--geologic_score_keys",
        type=str,
        nargs='+',
        default=DEFAULT_GEOLOGIC_SCORE_KEYS,
        help="Candidate zarr keys for geologic score volume lookup (first found is used).",
    )
    parser.add_argument(
        "--geologic_points_json_name",
        type=str,
        default="geologic_score_selected_points.json",
        help="Filename for persisted ranked geologic-score points beside each dataset.",
    )
    parser.add_argument(
        "--geologic_val_center_json_name",
        type=str,
        default="geologic_score_val_center.json",
        help="Filename for persisted validation center beside each dataset.",
    )
    parser.add_argument(
        "--geologic_target_points",
        type=int,
        default=500,
        help="Target number of ranked points to persist per dataset (default: 500).",
    )
    parser.add_argument(
        "--geologic_candidate_count",
        type=int,
        default=3000,
        help="Number of spread candidates generated before score filtering (default: 3000).",
    )
    parser.add_argument(
        "--geologic_candidate_probes",
        type=int,
        default=24,
        help="Probe count per best-candidate step when generating spread points (default: 24).",
    )
    parser.add_argument(
        "--geologic_dist_thresh_start",
        type=int,
        default=96,
        help="Initial distance threshold for ranked-point diversity selection (default: 96).",
    )
    parser.add_argument(
        "--geologic_dist_thresh_floor",
        type=int,
        default=32,
        help="Distance threshold floor for diversity backoff (default: 32).",
    )
    parser.add_argument("--val_split_ratio", type=float, default=0.2,
                       help="Validation split ratio over discovered datasets (default: 0.2)")
    parser.add_argument("--train_batches_per_epoch", type=int, default=None,
                       help="Optional fixed number of train batches per epoch. If set, loader cycles as needed.")
    parser.add_argument("--val_batches_per_epoch", type=int, default=None,
                       help="Optional fixed number of validation batches per epoch.")
    parser.add_argument("--refresh_every_batches", type=int, default=10,
                       help="Deprecated compatibility flag; dataset discovery/pruning now happens only at epoch boundaries.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Batch size. If omitted, the trainer selects 2 when the model has fewer than 14,000,000 "
            "parameters and 1 otherwise."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument(
        "--epoch_samples",
        type=int,
        default=None,
        help=(
            "Optional cap on logical samples per dataset epoch (before batching). "
            "Useful to avoid very large sampler allocations on massive zarr volumes."
        ),
    )
    parser.add_argument(
        "--tb_image_epochs",
        type=int,
        nargs='*',
        default=None,
        help=(
            "Optional epoch numbers (1-based) for TensorBoard image logging. "
            "When omitted, images are logged every epoch. Scalars/metrics are always logged every epoch. "
            "Example: --tb_image_epochs 1 5 8 9 10"
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--lr_schedule", type=str, default="poly",
                       choices=["poly", "cosine", "constant"],
                       help="Epoch LR schedule (default: poly with warmup, common for 3D medical UNet training)")
    parser.add_argument("--lr_poly_power", type=float, default=0.9,
                       help="Polynomial decay power when --lr_schedule=poly (default: 0.9)")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                       help="Minimum LR floor for poly/cosine schedules (default: 1e-6)")
    parser.add_argument("--lr_warmup_epochs", type=int, default=5,
                       help="Warmup epochs before decay schedules (default: 5)")
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1,
                       help="Warmup start as fraction of base LR (default: 0.1)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps (effective batch = batch_size * this value)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                       help="Clip gradient global norm to this value; set <=0 to disable")
    parser.add_argument(
        "--loss",
        type=str,
        default="huber",
        choices=["mse", "mae", "mae_smooth", "huber", "ssim", "sliding_stats", "smae", "multi_component"],
        help=(
            "Loss function: mse (MSELoss), mae (L1Loss), mae_smooth (smoothed L1), huber (SmoothL1Loss), "
            "ssim (w1*(1-SSIM)+w2*MSE+w3*L1), sliding_stats (local moments/extrema), "
            "smae (Smooth MAE, e*tanh(e/2), arXiv:2303.09935), or multi_component (weighted MSE+PMSE+MAE+optional LPIPS+amplitude calibration) "
            "over 3D volumes (default: huber)"
        ),
    )
    parser.add_argument(
        "--mae_smooth_kernel_weights",
        type=float,
        nargs='+',
        default=[1.0, 2.0, 1.0],
        help="Odd-length 1D smoothing kernel weights for --loss=mae_smooth (default: 1 2 1)",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="Delta parameter for SmoothL1Loss when --loss=huber (default: 1.0)",
    )
    parser.add_argument(
        "--ssim_window_size",
        type=int,
        default=7,
        help="Odd cubic SSIM window edge length for --loss=ssim (default: 7)",
    )
    parser.add_argument(
        "--ssim_w1",
        type=float,
        default=1.0,
        help="Weight w1 for (1-SSIM) in hybrid SSIM loss (default: 1.0)",
    )
    parser.add_argument(
        "--ssim_w2",
        type=float,
        default=0.0,
        help="Weight w2 for MSE term in hybrid SSIM loss (default: 0.0)",
    )
    parser.add_argument(
        "--ssim_w3",
        type=float,
        default=0.0,
        help="Weight w3 for L1 term in hybrid SSIM loss (default: 0.0)",
    )
    parser.add_argument(
        "--stats_window_size",
        type=int,
        nargs=3,
        default=[9, 9, 9],
        metavar=("D", "H", "W"),
        help="Sliding window size for --loss=sliding_stats (default: 9 9 9)",
    )
    parser.add_argument(
        "--stats_mask_mode",
        type=str,
        choices=["none", "valid"],
        default="none",
        help="Mask behavior for --loss=sliding_stats: none (ignore mask) or valid (use valid mask if provided)",
    )
    parser.add_argument("--stats_mean_weight", type=float, default=1.0, help="Weight for local-mean term in sliding_stats")
    parser.add_argument("--stats_std_weight", type=float, default=1.0, help="Weight for local-std-ratio term in sliding_stats")
    parser.add_argument("--stats_min_weight", type=float, default=1.0, help="Weight for local-minima term in sliding_stats")
    parser.add_argument("--stats_max_weight", type=float, default=1.0, help="Weight for local-maxima term in sliding_stats")
    parser.add_argument("--stats_mae_weight", type=float, default=1.0, help="Weight for voxelwise MAE term in sliding_stats")
    parser.add_argument("--stats_mse_weight", type=float, default=1.0, help="Weight for voxelwise MSE term in sliding_stats")
    parser.add_argument("--stats_std_ratio_clip", type=float, default=10.0, help="Clipping bound for local std-ratio in sliding_stats (default: 10.0)")
    parser.add_argument("--mc_mse_weight", type=float, default=0.2,
                       help="Weight for MSE term in --loss=multi_component (default: 0.2)")
    parser.add_argument("--mc_pmse_weight", type=float, default=0.6,
                       help="Weight for PMSE term in --loss=multi_component (default: 0.6)")
    parser.add_argument("--mc_mae_weight", type=float, default=0.2,
                       help="Weight for MAE term in --loss=multi_component (default: 0.2)")
    parser.add_argument("--mc_lpips_weight", type=float, default=0.0,
                       help="Weight for LPIPS (perceptual) term in --loss=multi_component (default: 0.0). LPIPS alone does not constrain amplitude scale/offset; add --mc_lpips_calib_weight to enforce mean/std alignment when this weight dominates.")
    parser.add_argument(
        "--mc_lpips_calib_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for LPIPS amplitude calibration term in --loss=multi_component "
            "(penalizes mean/std mismatch to keep amplitude scale and offset aligned; default: 0.0)"
        ),
    )
    parser.add_argument("--mc_lpips_net", type=str, default="alex",
                       choices=["alex", "vgg", "squeeze"],
                       help="LPIPS backbone for --loss=multi_component (default: alex)")
    parser.add_argument("--mc_pmse_eps", type=float, default=1e-8,
                       help="Epsilon clamp for PMSE denominator in --loss=multi_component (default: 1e-8)")
    parser.add_argument("--mc_tv_weight", type=float, default=0.0,
                       help="Weight for Total Variation regularisation term in --loss=multi_component (default: 0.0; try 1e-4 to suppress checkerboard artifacts)")
    parser.add_argument("--mc_gdl_weight", type=float, default=0.0,
                       help="Weight for Gradient Difference Loss (GDL) term in --loss=multi_component (default: 0.0). GDL penalises differences in spatial gradient magnitude between prediction and target, suppressing vertical stripe artifacts from zeroed trace-cluster masking while preserving real geological edges. Recommended range: 0.05-0.2.")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                       help="EMA decay for model weights; set <=0 to disable")
    parser.add_argument("--ema_update_every", type=int, default=1,
                       help="Update EMA every N optimizer steps (default: 1)")
    parser.add_argument("--sample_shape", type=int, nargs=3, default=[128, 128, 128],
                       help="Sample shape (x y z)")
    parser.add_argument(
        "--input_extrema_prob",
            "--input-extrema-prob",
        type=float,
        default=1.0,
        help="Relative probability for extrema-only input masking strategy (default: 1.0)",
    )
    parser.add_argument(
        "--input_sparse_keep_prob",
            "--input-sparse-keep-prob",
        type=float,
        default=0.0,
        help="Relative probability for sparse-keep input masking strategy (default: 0.0)",
    )
    parser.add_argument(
        "--input_decimate_trilinear_prob",
            "--input-decimate-trilinear-prob",
        type=float,
        default=0.0,
        help="Relative probability for decimate+trilinear input masking strategy (default: 0.0)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs='+',
        default=[32, 64, 128, 256],
        help=(
            "Channel widths per encoder stage (e.g. --hidden_dims 32 64 128 256). "
            "Number of values must match --unet_levels. Default: 32 64 128 256."
        ),
    )
    parser.add_argument(
        "--unet_levels",
        type=int,
        default=4,
        choices=range(3, 7),
        help="Number of encoder/decoder levels for SeismicUNet3d (default: 4, range: 3-6).",
    )
    parser.add_argument(
        "--kernel_sizes",
        type=int,
        nargs='+',
        default=None,
        help=(
            "Optional odd kernel schedule per hidden-dim stage (e.g. --kernel_sizes 7 5 3 3). "
            "Length must match model hidden dims. Default keeps legacy 3x3 kernels."
        ),
    )
    parser.add_argument(
        "--encoder_depth_profile",
        type=str,
        choices=["baseline", "deeper", "deepest"],
        default="baseline",
        help=(
            "Named encoder stage-block profile. baseline keeps canonical depths. "
            "deeper/deepest are supported for unet_levels 3/4."
        ),
    )
    parser.add_argument(
        "--encoder_stage_blocks",
        type=int,
        nargs='+',
        default=None,
        help=(
            "Optional explicit encoder stage block counts. Length must match --unet_levels. "
            "Overrides --encoder_depth_profile when provided."
        ),
    )
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--use_mamba", action="store_true",
                       help="Use U-Mamba hybrid blocks in encoder (requires CUDA + mamba_ssm; falls back to ResBlock3d on MPS/CPU)")
    parser.add_argument("--thermal_max_c", type=float, default=85.0,
                       help="Pause when CPU temperature exceeds this in Celsius; set <=0 to disable")
    parser.add_argument("--thermal_cooldown_sec", type=int, default=300,
                       help="Cooldown sleep duration in seconds after a thermal pause")
    parser.add_argument("--thermal_check_every_batches", type=int, default=10,
                       help="Check CPU temperature every N training batches")
    parser.add_argument("--thermal_pressure_trip_level", type=str, default="serious",
                       choices=["off", "nominal", "fair", "serious", "critical"],
                       help="Pause on thermal pressure at or above this level (default: serious). Use 'off' to disable pressure-based pausing")

    parser.add_argument(
        "--deep-reconstruction-head",
        action="store_true",
        help="Use a deep reconstruction head (2 Conv3d layers with norm and activation) instead of a single Conv3d layer.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    args.data_paths = _normalize_data_paths(args.data_paths, args.dataset_glob)

    backprop_defaults = {
        "lr": parser.get_default("lr"),
        "lr_schedule": parser.get_default("lr_schedule"),
        "loss": parser.get_default("loss"),
        "mae_smooth_kernel_weights": parser.get_default("mae_smooth_kernel_weights"),
        "huber_delta": parser.get_default("huber_delta"),
        "ssim_window_size": parser.get_default("ssim_window_size"),
        "ssim_w1": parser.get_default("ssim_w1"),
        "ssim_w2": parser.get_default("ssim_w2"),
        "ssim_w3": parser.get_default("ssim_w3"),
        "stats_window_size": parser.get_default("stats_window_size"),
        "stats_mask_mode": parser.get_default("stats_mask_mode"),
        "stats_mean_weight": parser.get_default("stats_mean_weight"),
        "stats_std_weight": parser.get_default("stats_std_weight"),
        "stats_min_weight": parser.get_default("stats_min_weight"),
        "stats_max_weight": parser.get_default("stats_max_weight"),
        "stats_mae_weight": parser.get_default("stats_mae_weight"),
        "stats_mse_weight": parser.get_default("stats_mse_weight"),
        "stats_std_ratio_clip": parser.get_default("stats_std_ratio_clip"),
        "mc_mse_weight": parser.get_default("mc_mse_weight"),
        "mc_pmse_weight": parser.get_default("mc_pmse_weight"),
        "mc_mae_weight": parser.get_default("mc_mae_weight"),
        "mc_lpips_weight": parser.get_default("mc_lpips_weight"),
        "mc_lpips_calib_weight": parser.get_default("mc_lpips_calib_weight"),
        "mc_lpips_net": parser.get_default("mc_lpips_net"),
        "mc_pmse_eps": parser.get_default("mc_pmse_eps"),
        "mc_tv_weight": parser.get_default("mc_tv_weight"),
        "mc_gdl_weight": parser.get_default("mc_gdl_weight"),
        "unet_levels": parser.get_default("unet_levels"),
        "encoder_depth_profile": parser.get_default("encoder_depth_profile"),
        "grad_accum_steps": parser.get_default("grad_accum_steps"),
        "grad_clip_norm": parser.get_default("grad_clip_norm"),
        "ema_decay": parser.get_default("ema_decay"),
        "ema_update_every": parser.get_default("ema_update_every"),
    }

    config = {
        "args": vars(args),
        "cli_provided": sorted(_collect_cli_option_names(cli_argv)),
        "backprop_defaults": backprop_defaults,
    }
    run_training(config)


if __name__ == "__main__":
    main()
