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
