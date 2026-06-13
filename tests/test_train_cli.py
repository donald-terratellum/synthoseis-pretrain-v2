from pathlib import Path

import train_cli
from train_cli import _normalize_data_paths


def test_normalize_data_paths_expands_parent_folder(tmp_path: Path):
    parent = tmp_path / "fake_data"
    dataset_a = parent / "seismic__001" / "model_data.zarr"
    dataset_b = parent / "seismic__002" / "model_data.zarr"
    dataset_a.mkdir(parents=True)
    dataset_b.mkdir(parents=True)

    normalized = _normalize_data_paths([str(parent)], "seismic__*/model_data.zarr")

    assert normalized == [str(dataset_a), str(dataset_b)]


def test_normalize_data_paths_keeps_explicit_zarr_store(tmp_path: Path):
    store = tmp_path / "model_data.zarr"
    store.mkdir()

    normalized = _normalize_data_paths([str(store)], "seismic__*/model_data.zarr")

    assert normalized == [str(store)]


def test_main_forwards_normalized_paths(tmp_path: Path, monkeypatch):
    parent = tmp_path / "fake_data"
    dataset = parent / "seismic__001" / "model_data.zarr"
    dataset.mkdir(parents=True)

    captured: dict = {}

    def _fake_run_training(config: dict) -> None:
        captured.update(config)

    monkeypatch.setattr(train_cli, "run_training", _fake_run_training)

    train_cli.main(["--data_paths", str(parent), "--epochs", "1"])

    assert captured["args"]["data_paths"] == [str(dataset)]


def test_parser_defaults_for_multi_component_and_unet_levels():
    parser = train_cli._build_parser()
    args = parser.parse_args(["--data_paths", "dummy.zarr"])

    assert args.loss == "huber"
    assert args.mc_mse_weight == 0.2
    assert args.mc_pmse_weight == 0.6
    assert args.mc_mae_weight == 0.2
    assert args.mc_lpips_weight == 0.0
    assert args.mc_lpips_net == "alex"
    assert args.mc_pmse_eps == 1e-8
    assert args.unet_levels == 4
    assert args.encoder_depth_profile == "baseline"
    assert args.encoder_stage_blocks is None


def test_parser_accepts_multi_component_and_custom_unet_levels():
    parser = train_cli._build_parser()
    args = parser.parse_args([
        "--data_paths",
        "dummy.zarr",
        "--loss",
        "multi_component",
        "--mc_mse_weight",
        "0.1",
        "--mc_pmse_weight",
        "0.7",
        "--mc_mae_weight",
        "0.2",
        "--mc_lpips_weight",
        "0.0",
        "--unet_levels",
        "5",
        "--hidden_dims",
        "16",
        "32",
        "64",
        "128",
        "256",
        "--encoder_depth_profile",
        "deeper",
        "--encoder_stage_blocks",
        "3",
        "4",
        "8",
        "4",
        "3",
    ])

    assert args.loss == "multi_component"
    assert args.mc_mse_weight == 0.1
    assert args.mc_pmse_weight == 0.7
    assert args.mc_mae_weight == 0.2
    assert args.unet_levels == 5
    assert tuple(args.hidden_dims) == (16, 32, 64, 128, 256)
    assert args.encoder_depth_profile == "deeper"
    assert tuple(args.encoder_stage_blocks) == (3, 4, 8, 4, 3)
