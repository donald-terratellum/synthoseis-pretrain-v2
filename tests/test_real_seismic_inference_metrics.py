import csv
from pathlib import Path

import numpy as np

from studies.list_real_seismic_inference_metrics import (
    _build_epoch_metrics_index,
    _discover_checkpoint_paths,
    _normalize_with_volume_std,
    _resolve_checkpoint_meta,
)


def test_discover_checkpoint_paths_collects_all_epochs_and_excludes_seed_list(tmp_path, monkeypatch):
    folder_a = tmp_path / "checkpoints" / "checkpoint_copilot_1"
    folder_b = tmp_path / "checkpoints" / "checkpoint_copilot_2"
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)

    c1 = folder_a / "checkpoint_epoch_0001.pt"
    c2 = folder_a / "checkpoint_epoch_0002.pt"
    c3 = folder_b / "checkpoint_epoch_0001.pt"
    for path in (c1, c2, c3):
        path.write_bytes(b"stub")

    source = [
        str(c1.relative_to(tmp_path)),
        str(c3.relative_to(tmp_path)),
    ]
    monkeypatch.setattr("studies.list_real_seismic_inference_metrics.ROOT", tmp_path)

    discovered = _discover_checkpoint_paths(source, exclude_input_checkpoints=True)
    assert discovered == [c2]

    discovered_all = _discover_checkpoint_paths(source, exclude_input_checkpoints=False)
    assert discovered_all == [c1, c2, c3]


def test_resolve_checkpoint_meta_uses_tensorboard_folder_and_epoch(tmp_path, monkeypatch):
    csv_path = tmp_path / "checkpoints" / "epoch_component_metrics.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tensorboard folder",
                "epoch",
                "unet_levels",
                "hidden_dims (as a space-delimited string)",
                "kernel_schedule (as a space-delimited string)",
                "encoder_depth_profile (profile - stage blocks)",
                "lr",
                "mse_weight",
                "pmse_weight",
                "mae_weight",
                "lpips_weight",
                "tv_weight",
                "gdl_weight",
            ]
        )
        writer.writerow(
            [
                "checkpoints/checkpoint_copilot_1/runs",
                "10",
                "4",
                "32 64 128 256",
                "3 3 3 3",
                "deeper - 3 4 8 4",
                "1.00000000e-04",
                "0.200000",
                "0.600000",
                "0.200000",
                "0.000000",
                "0.000000",
                "0.000000",
            ]
        )

    monkeypatch.setattr("studies.list_real_seismic_inference_metrics.ROOT", tmp_path)
    index = _build_epoch_metrics_index(csv_path)

    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_copilot_1" / "checkpoint_epoch_0010.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"stub")

    meta = _resolve_checkpoint_meta(checkpoint_path, index)
    assert meta.epoch == 10
    assert meta.unet_levels == 4
    assert meta.hidden_dims == (32, 64, 128, 256)
    assert meta.kernel_schedule == (3, 3, 3, 3)
    assert meta.encoder_depth_profile == "deeper - 3 4 8 4"
    assert meta.encoder_stage_blocks == (3, 4, 8, 4)
    assert meta.tensorboard_folder == "checkpoints/checkpoint_copilot_1/runs"


def test_normalize_with_volume_std_uses_global_std_scale():
    cube = np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32)

    normalized = _normalize_with_volume_std(cube, volume_std=2.0)

    assert normalized.dtype == np.float32
    assert np.allclose(normalized, cube / 2.0)


def test_normalize_with_volume_std_shifts_unit_interval_data_by_half():
    cube = np.array([[[0.0, 1.0], [0.25, 0.75]]], dtype=np.float32)

    normalized = _normalize_with_volume_std(cube, volume_std=0.5)

    expected = (cube - 0.5) / 0.5
    assert np.allclose(normalized, expected)
