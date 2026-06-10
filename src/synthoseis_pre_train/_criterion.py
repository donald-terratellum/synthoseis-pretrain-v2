"""Loss factory and training summary helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from synthoseis_pre_train.losses import (
    MAESmoothLoss3D,
    MultiComponentLoss3D,
    SMAELoss,
    SSIMHybridLoss3D,
    SlidingWindowStatsLoss3D,
)


class _ScaledL1Loss(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.base = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base(pred, target) * self.scale


def _build_criterion(args) -> nn.Module:
    """Return the loss criterion selected via --loss."""
    loss_fn = getattr(args, "loss", "huber")

    if loss_fn == "mse":
        return nn.MSELoss()
    if loss_fn == "mae":
        return _ScaledL1Loss(scale=2.0)  # Match MSE map scale for easier comparison
    if loss_fn == "mae_smooth":
        return MAESmoothLoss3D()
    if loss_fn == "huber":
        return nn.HuberLoss()
    if loss_fn == "ssim":
        return SSIMHybridLoss3D(
            window_size=int(getattr(args, "ssim_window_size", 7)),
            w1=float(getattr(args, "ssim_w1", 1.0)),
            w2=float(getattr(args, "ssim_w2", 0.0)),
            w3=float(getattr(args, "ssim_w3", 0.0)),
        )
    if loss_fn == "sliding_stats":
        stats_window_raw = getattr(args, "stats_window_size", [9, 9, 9])
        stats_window_tuple: tuple[int, int, int] = (
            int(stats_window_raw[0]),
            int(stats_window_raw[1]),
            int(stats_window_raw[2]),
        )
        return SlidingWindowStatsLoss3D(
            window_size=stats_window_tuple,
            mean_weight=float(getattr(args, "stats_mean_weight", 1.0)),
            std_weight=float(getattr(args, "stats_std_weight", 1.0)),
            min_weight=float(getattr(args, "stats_min_weight", 1.0)),
            max_weight=float(getattr(args, "stats_max_weight", 1.0)),
            mae_weight=float(getattr(args, "stats_mae_weight", 1.0)),
            mse_weight=float(getattr(args, "stats_mse_weight", 1.0)),
            std_ratio_clip=float(getattr(args, "stats_std_ratio_clip", 10.0)),
            mask_mode=str(getattr(args, "stats_mask_mode", "none")),
        )
    if loss_fn == "smae":
        return SMAELoss()
    if loss_fn == "multi_component":
        return MultiComponentLoss3D(
            mse_weight=float(getattr(args, "mc_mse_weight", 0.2)),
            pmse_weight=float(getattr(args, "mc_pmse_weight", 0.6)),
            mae_weight=float(getattr(args, "mc_mae_weight", 0.2)),
            lpips_weight=float(getattr(args, "mc_lpips_weight", 0.0)),
            lpips_net=str(getattr(args, "mc_lpips_net", "alex")),
            pmse_eps=float(getattr(args, "mc_pmse_eps", 1e-8)),
        )
    raise ValueError(f"Unknown loss function: {loss_fn!r}")


def _print_loss_and_backprop_summary(
    args,
    cli_provided: set[str],
    defaults: dict[str, object],
    scaler,
) -> None:
    """Print grouped summary of effective optimization and backprop settings."""

    def _src(name: str) -> str:
        return "user" if name in cli_provided else "default"

    amp_enabled = scaler is not None
    grad_accum = max(1, int(args.grad_accum_steps))
    clip_desc = f"{args.grad_clip_norm:g}" if args.grad_clip_norm > 0 else "disabled"
    ema_enabled = args.ema_decay > 0
    ema_desc = f"{args.ema_decay:g}" if ema_enabled else "disabled"

    label_width = 18

    def _kv(label: str, value: str) -> None:
        print(f"    {label:<{label_width}} : {value}")

    print("Training configuration:")
    print("  Optimization:")
    _kv("optimizer", "Adam (fixed)")
    _kv("lr", f"{args.lr:.3e} ({_src('lr')}, default={defaults['lr']:.3e})")
    _kv(
        "lr schedule",
        f"{args.lr_schedule} ({_src('lr_schedule')}, default={defaults['lr_schedule']})",
    )
    if args.lr_schedule != "constant":
        _kv(
            "schedule details",
            f"min={args.lr_min:.3e}, warmup={args.lr_warmup_epochs} epoch(s), "
            f"warmup_start_factor={args.lr_warmup_start_factor:g}",
        )
        if args.lr_schedule == "poly":
            _kv("poly power", f"{args.lr_poly_power:g}")

    print("  Loss and backprop:")
    _loss_name = getattr(args, "loss", "huber")
    if _loss_name == "huber":
        _delta = float(getattr(args, "huber_delta", 1.0))
        _loss_desc = (
            f"huber/SmoothL1 (delta={_delta:g}, "
            f"{_src('huber_delta')}, default={defaults['huber_delta']:g})"
        )
    elif _loss_name == "mae_smooth":
        _loss_desc = "mae_smooth"
    elif _loss_name == "ssim":
        _loss_desc = "ssim-hybrid"
    elif _loss_name == "sliding_stats":
        _loss_desc = "sliding-window-stats"
    elif _loss_name == "multi_component":
        _loss_desc = "multi-component"
    else:
        _loss_desc = _loss_name
    _kv("loss", f"{_loss_desc} ({_src('loss')}, default={defaults['loss']})")
    if _loss_name == "mae_smooth":
        _kernel = [float(v) for v in getattr(args, "mae_smooth_kernel_weights", [1.0, 2.0, 1.0])]
        _kernel_str = " ".join(f"{v:g}" for v in _kernel)
        print(f"{' ':4}{' ':<{label_width}}   - kernel_1d=[{_kernel_str}]")
    if _loss_name == "ssim":
        _ssim_window = int(getattr(args, "ssim_window_size", 7))
        _ssim_w1 = float(getattr(args, "ssim_w1", 1.0))
        _ssim_w2 = float(getattr(args, "ssim_w2", 0.0))
        _ssim_w3 = float(getattr(args, "ssim_w3", 0.0))
        print(f"{' ':4}{' ':<{label_width}}   - window={_ssim_window},weights: ")
        print(
            f"{' ':4}{' ':<{label_width}}     "
            f"(ssim_term={_ssim_w1:g}, mse_term={_ssim_w2:g}, mae_term={_ssim_w3:g})"
        )
    if _loss_name == "sliding_stats":
        _win = [int(v) for v in getattr(args, "stats_window_size", [9, 9, 9])]
        _mode = str(getattr(args, "stats_mask_mode", "none"))
        _mw = float(getattr(args, "stats_mean_weight", 1.0))
        _sw = float(getattr(args, "stats_std_weight", 1.0))
        _minw = float(getattr(args, "stats_min_weight", 1.0))
        _maxw = float(getattr(args, "stats_max_weight", 1.0))
        _maew = float(getattr(args, "stats_mae_weight", 1.0))
        _msew = float(getattr(args, "stats_mse_weight", 1.0))
        _clip = float(getattr(args, "stats_std_ratio_clip", 10.0))
        print(f"{' ':4}{' ':<{label_width}}   - window={tuple(_win)}, mask_mode={_mode}, std_ratio_clip={_clip:g}")
        print(
            f"{' ':4}{' ':<{label_width}}     "
            f"(mean={_mw:g}, std={_sw:g}, min={_minw:g}, max={_maxw:g}, mae={_maew:g}, mse={_msew:g})"
        )
    if _loss_name == "multi_component":
        _mse = float(getattr(args, "mc_mse_weight", 0.2))
        _pmse = float(getattr(args, "mc_pmse_weight", 0.6))
        _mae = float(getattr(args, "mc_mae_weight", 0.2))
        _lpips = float(getattr(args, "mc_lpips_weight", 0.0))
        _lpips_net = str(getattr(args, "mc_lpips_net", "alex"))
        _pmse_eps = float(getattr(args, "mc_pmse_eps", 1e-8))
        print(
            f"{' ':4}{' ':<{label_width}}   - "
            f"weights(mse={_mse:g}, pmse={_pmse:g}, mae={_mae:g}, lpips={_lpips:g})"
        )
        print(
            f"{' ':4}{' ':<{label_width}}     "
            f"(lpips_net={_lpips_net}, pmse_eps={_pmse_eps:g})"
        )
    _kv("AMP", f"{'on' if amp_enabled else 'off'} (auto)")
    _kv(
        "grad_accum_steps",
        f"{grad_accum} ({_src('grad_accum_steps')}, default={defaults['grad_accum_steps']})",
    )
    _kv(
        "grad_clip_norm",
        f"{clip_desc} ({_src('grad_clip_norm')}, default={defaults['grad_clip_norm']})",
    )

    print("  EMA:")
    _kv("enabled", "yes" if ema_enabled else "no")
    _kv("decay", f"{ema_desc} ({_src('ema_decay')}, default={defaults['ema_decay']})")
    _kv(
        "update every",
        f"{max(1, int(args.ema_update_every))} step(s) "
        f"({_src('ema_update_every')}, default={defaults['ema_update_every']})",
    )
