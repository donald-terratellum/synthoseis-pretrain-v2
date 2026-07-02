"""Unit tests for NpySeismicDataset."""

from __future__ import annotations

import numpy as np
import pytest

from synthoseis_pre_train._npy_dataset import NpySeismicDataset


def _write_npy(tmp_path, name: str, shape: tuple, dtype=np.float32, fill: float | None = None) -> str:
    arr = np.random.default_rng(42).standard_normal(shape).astype(dtype) if fill is None else np.full(shape, fill, dtype=dtype)
    path = tmp_path / name
    np.save(str(path), arr)
    return str(path)


def test_basic_getitem_shapes(tmp_path):
    path = _write_npy(tmp_path, "vol.npy", (200, 200, 200))
    ds = NpySeismicDataset(path, sample_shape=(128, 128, 128), epoch_samples=4)
    assert len(ds) == 4
    inp, tgt, msk = ds[0]
    assert inp.shape == (128, 128, 128)
    assert tgt.shape == (128, 128, 128)
    assert msk.shape == (128, 128, 128)
    assert inp.dtype == np.float32
    assert msk.dtype == np.bool_


def test_volume_std_normalisation(tmp_path):
    """After cache build, the stored normalized volume should have std ≈ 1."""
    path = _write_npy(tmp_path, "vol.npy", (160, 160, 160))
    ds = NpySeismicDataset(path, sample_shape=(128, 128, 128), epoch_samples=1)
    volume = np.asarray(ds._volume[:], dtype=np.float32)
    assert abs(float(volume.std()) - 1.0) < 0.05


def test_unit_interval_gets_half_shift(tmp_path):
    """Volumes with min=0 / max=1 should be shifted by -0.5 before std scaling."""
    arr = np.random.default_rng(0).uniform(0.0, 1.0, (160, 160, 160)).astype(np.float32)
    # Force exact 0.0 and 1.0 at two voxels so the unit-interval check fires.
    arr[0, 0, 0] = 0.0
    arr[-1, -1, -1] = 1.0
    path = tmp_path / "unit.npy"
    np.save(str(path), arr)
    ds = NpySeismicDataset(path, sample_shape=(128, 128, 128), epoch_samples=1)
    # After -0.5 shift + std normalisation the mean should be near 0.
    volume = np.asarray(ds._volume[:], dtype=np.float32)
    assert abs(float(volume.mean())) < 0.1


def test_volume_shape_exposed(tmp_path):
    path = _write_npy(tmp_path, "vol.npy", (300, 200, 150))
    ds = NpySeismicDataset(path, sample_shape=(128, 128, 128), epoch_samples=1)
    assert ds.volume_shape == (300, 200, 150)


def test_rejects_non_npy(tmp_path):
    fake = tmp_path / "data.zarr"
    fake.mkdir()
    with pytest.raises(ValueError, match="npy"):
        NpySeismicDataset(str(fake))


def test_rejects_volume_too_small(tmp_path):
    path = _write_npy(tmp_path, "small.npy", (64, 64, 64))
    with pytest.raises(ValueError, match="sample_shape"):
        NpySeismicDataset(path, sample_shape=(128, 128, 128))


def test_epoch_samples_caps_len(tmp_path):
    path = _write_npy(tmp_path, "big.npy", (400, 400, 400))
    ds = NpySeismicDataset(path, sample_shape=(128, 128, 128), epoch_samples=10)
    assert len(ds) == 10


def test_no_epoch_samples_uses_valid_positions(tmp_path):
    path = _write_npy(tmp_path, "small.npy", (130, 130, 130))
    ds = NpySeismicDataset(path, sample_shape=(128, 128, 128))
    # valid starts = 3 * 3 * 3 = 27
    assert len(ds) == 27


def test_builds_and_reuses_zarr_cache(tmp_path):
    path = _write_npy(tmp_path, "vol.npy", (160, 160, 160))
    cache_path = tmp_path / "vol.real.zarr"

    ds1 = NpySeismicDataset(path, sample_shape=(128, 128, 128), epoch_samples=1)
    assert cache_path.exists()
    mtime_1 = cache_path.stat().st_mtime_ns

    ds2 = NpySeismicDataset(path, sample_shape=(128, 128, 128), epoch_samples=1)
    mtime_2 = cache_path.stat().st_mtime_ns

    assert ds1.volume_shape == ds2.volume_shape
    assert mtime_2 == mtime_1
