from pathlib import Path

import torch
from torch import nn, optim

from synthoseis_pre_train._checkpoint import _maybe_update_best_val_checkpoint


def _build_minimal_training_state():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = None
    return model, optimizer, scaler


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def test_best_val_checkpoint_created_when_missing(tmp_path: Path):
    model, optimizer, scaler = _build_minimal_training_state()

    updated = _maybe_update_best_val_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        train_loss=0.9,
        val_loss=0.8,
        train_paths=["train_a"],
        val_paths=["val_a"],
        ema_state=None,
    )

    best_path = tmp_path / "best_val_epoch.pt"
    previous_path = tmp_path / "previous_best_val_epoch.pt"

    assert updated is True
    assert best_path.exists()
    assert not previous_path.exists()

    ckpt = _load_checkpoint(best_path)
    assert ckpt["epoch"] == 0
    assert ckpt["train_loss"] == 0.9
    assert ckpt["val_loss"] == 0.8
    assert ckpt["train_paths"] == ["train_a"]
    assert ckpt["val_paths"] == ["val_a"]
    assert "optimizer" in ckpt
    assert "model" in ckpt


def test_best_val_checkpoint_rotates_to_previous_before_overwrite(tmp_path: Path):
    model, optimizer, scaler = _build_minimal_training_state()

    first_updated = _maybe_update_best_val_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=1,
        train_loss=1.0,
        val_loss=0.6,
        train_paths=["train_old"],
        val_paths=["val_old"],
        ema_state=None,
    )
    assert first_updated is True

    second_updated = _maybe_update_best_val_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=2,
        train_loss=0.7,
        val_loss=0.5,
        train_paths=["train_new"],
        val_paths=["val_new"],
        ema_state=None,
    )

    best_path = tmp_path / "best_val_epoch.pt"
    previous_path = tmp_path / "previous_best_val_epoch.pt"

    assert second_updated is True
    assert best_path.exists()
    assert previous_path.exists()

    previous_ckpt = _load_checkpoint(previous_path)
    best_ckpt = _load_checkpoint(best_path)

    assert previous_ckpt["val_loss"] == 0.6
    assert previous_ckpt["epoch"] == 1
    assert best_ckpt["val_loss"] == 0.5
    assert best_ckpt["epoch"] == 2


def test_best_val_checkpoint_not_updated_without_improvement(tmp_path: Path):
    model, optimizer, scaler = _build_minimal_training_state()

    _maybe_update_best_val_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=3,
        train_loss=0.7,
        val_loss=0.4,
        train_paths=["train"],
        val_paths=["val"],
        ema_state=None,
    )

    updated = _maybe_update_best_val_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=4,
        train_loss=0.6,
        val_loss=0.4,
        train_paths=["train"],
        val_paths=["val"],
        ema_state=None,
    )

    best_path = tmp_path / "best_val_epoch.pt"
    previous_path = tmp_path / "previous_best_val_epoch.pt"
    best_ckpt = _load_checkpoint(best_path)

    assert updated is False
    assert not previous_path.exists()
    assert best_ckpt["epoch"] == 3
    assert best_ckpt["val_loss"] == 0.4
