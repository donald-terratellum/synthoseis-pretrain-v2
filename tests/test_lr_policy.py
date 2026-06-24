from pathlib import Path
from types import SimpleNamespace
import csv

import pytest
import torch

from synthoseis_pre_train.pretrain import (
    _COMPONENT_METRIC_CSV_HEADERS,
    _LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_LR,
    _fixed_validation_heuristic,
    _normalize_existing_component_metrics_csv,
    _scale_optimizer_lr,
)
from studies.run_random_training_sweep import RunConfig, _build_cli_command


def _build_sweep_args() -> SimpleNamespace:
    return SimpleNamespace(
        epochs=10,
        val_split_ratio=0.3,
        train_batches_per_epoch=100,
        val_batches_per_epoch=40,
        mc_lpips_net="alex",
        input_extrema_prob=0.6,
        input_sparse_keep_prob=0.3,
        input_decimate_trilinear_prob=0.1,
        grad_accum_steps=1,
        grad_clip_norm=1.0,
        ema_decay=0.995,
        ema_update_every=1,
        lr_schedule="poly",
        lr_warmup_epochs=3,
        lr_warmup_start_factor=0.025,
        lr_poly_power=0.9,
        lr_min=2.5e-5,
        lr=5.0e-5,
        data_folder=Path("/tmp/fake_data"),
        batch_size=None,
    )


def _cfg(lpips_weight: float) -> RunConfig:
    return RunConfig(
        run_id="r1",
        unet_levels=4,
        encoder_depth_profile="deeper",
        hidden_dims=(32, 64, 128, 256),
        mse_weight=0.0,
        pmse_weight=0.0,
        mae_weight=1.0 - lpips_weight,
        lpips_weight=lpips_weight,
        gdl_weight=0.0,
        tv_weight=0.0,
        output_dir=Path("/tmp/out"),
    )


def _arg_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def test_build_cli_command_forces_low_lr_when_lpips_enabled():
    cmd = _build_cli_command(_cfg(lpips_weight=0.01), _build_sweep_args())
    assert _arg_value(cmd, "--lr") == "1e-05"
    assert _arg_value(cmd, "--lr_min") == "5e-06"


def test_build_cli_command_keeps_user_lr_when_lpips_disabled():
    cmd = _build_cli_command(_cfg(lpips_weight=0.0), _build_sweep_args())
    assert _arg_value(cmd, "--lr") == "5e-05"
    assert _arg_value(cmd, "--lr_min") == "2.5e-05"


def test_scale_optimizer_lr_updates_scheduler_base_lrs():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    old_lr, new_lr = _scale_optimizer_lr(optimizer, scale=0.5, scheduler=scheduler)

    assert old_lr == pytest.approx(1e-4)
    assert new_lr == pytest.approx(5e-5)
    assert scheduler.base_lrs[0] == pytest.approx(5e-5)


def test_fixed_validation_heuristic_matches_formula():
    score = _fixed_validation_heuristic({"mae": 0.2, "mse": 0.3, "lpips": 0.4})
    assert score == pytest.approx(1.1)


def test_component_metrics_csv_has_lr_after_encoder_depth_profile():
    enc_idx = _COMPONENT_METRIC_CSV_HEADERS.index("encoder_depth_profile (profile - stage blocks)")
    lr_idx = _COMPONENT_METRIC_CSV_HEADERS.index("lr")
    assert lr_idx == enc_idx + 1


def test_normalize_component_metrics_csv_inserts_blank_lr_for_legacy_schema(tmp_path):
    csv_path = tmp_path / "epoch_component_metrics.csv"
    lr_idx = _COMPONENT_METRIC_CSV_HEADERS.index("lr")
    row = [f"v{i}" for i in range(len(_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_LR))]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_LEGACY_COMPONENT_METRIC_CSV_HEADERS_NO_LR)
        writer.writerow(row)

    _normalize_existing_component_metrics_csv(csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == _COMPONENT_METRIC_CSV_HEADERS
    assert len(rows[1]) == len(_COMPONENT_METRIC_CSV_HEADERS)
    assert rows[1][lr_idx] == ""
