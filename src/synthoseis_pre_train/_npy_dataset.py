"""PyTorch Dataset wrapping real seismic .npy volumes via a streaming zarr cache."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class NpySeismicDataset(Dataset):
    """Random-crop masked-reconstruction dataset backed by a .npy seismic volume.

    Mirrors the behavior of ``SeismicDataset`` (zarr-backed) so both can be
    merged in a single ``ConcatDataset`` and sampled together:

        * .npy is converted once into a normalized chunked zarr cache for low-RAM
            random crop access during training.
        * Volume is normalised by its global standard deviation (zero-mean assumed).
    * If the global min/max is exactly [0, 1] the volume is first shifted by -0.5.
    * Each ``__getitem__`` draws a random 128³ crop centre, applies trace-cluster
      dropout, then the configured extrema/sparse/decimate input strategy.
        * Returns ``(input_data, target, mask)`` as numpy arrays shaped
            ``(Z, X, Y)`` – the same contract as ``SeismicDataset``.

    ``volume_shape`` is exposed as the raw 3D shape of the source .npy array so
    callers can compute voxel-count sampling weights without loading full data.
    """

    def __init__(
        self,
        npy_path: str,
        sample_shape: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = True,
        trace_mask_ratio: float = 0.07,
        cluster_prob: float = 0.8,
        input_extrema_prob: float = 1.0,
        input_sparse_keep_prob: float = 0.0,
        input_decimate_trilinear_prob: float = 0.0,
        epoch_samples: int | None = None,
    ) -> None:
        self._path = Path(npy_path)
        if not self._path.exists():
            raise FileNotFoundError(f"NpySeismicDataset: file not found: {self._path}")
        if self._path.suffix != ".npy":
            raise ValueError(f"NpySeismicDataset: expected .npy file, got: {self._path}")
        self.data_path = str(self._path)
        self.available_cubes = [self._path.stem]

        self.sample_shape = tuple(int(s) for s in sample_shape)
        self.augment = bool(augment)
        self.trace_mask_ratio = float(trace_mask_ratio)
        self.cluster_prob = float(cluster_prob)
        self.epoch_samples = int(epoch_samples) if epoch_samples is not None else None

        for prob_name, prob_val in (
            ("input_extrema_prob", input_extrema_prob),
            ("input_sparse_keep_prob", input_sparse_keep_prob),
            ("input_decimate_trilinear_prob", input_decimate_trilinear_prob),
        ):
            if not 0.0 <= float(prob_val) <= 1.0:
                raise ValueError(f"{prob_name} must be in [0, 1].")
        prob_sum = float(input_extrema_prob + input_sparse_keep_prob + input_decimate_trilinear_prob)
        if prob_sum <= 0.0:
            raise ValueError("At least one input strategy probability must be > 0.")

        self._input_strategy_probs = np.array(
            [float(input_extrema_prob), float(input_sparse_keep_prob), float(input_decimate_trilinear_prob)],
            dtype=np.float64,
        ) / prob_sum

        # Probe shape/stats from mmap without loading whole volume into RAM.
        probe = np.load(str(self._path), mmap_mode="r")
        if probe.ndim != 3:
            raise ValueError(
                f"NpySeismicDataset: expected 3D array in {self._path}, got ndim={probe.ndim}"
            )
        self.volume_shape: Tuple[int, int, int] = tuple(int(s) for s in probe.shape)  # type: ignore[assignment]

        # Validate that every sample dimension fits in the volume.
        for dim_size, sample_size, axis_name in zip(
            self.volume_shape, self.sample_shape, ("axis-0", "axis-1", "axis-2")
        ):
            if dim_size < sample_size:
                raise ValueError(
                    f"NpySeismicDataset: volume {self._path} {axis_name} size {dim_size} "
                    f"< sample_shape {sample_size}"
                )

        # Compute normalisation stats once (full volume scan).
        vol_f64 = probe.astype(np.float64, copy=False)
        vol_min = float(vol_f64.min())
        vol_max = float(vol_f64.max())
        self._needs_unit_shift = bool(np.isclose(vol_min, 0.0) and np.isclose(vol_max, 1.0))
        self._volume_std = float(np.std(vol_f64))
        self._cache_path = self._path.with_suffix(".real.zarr")
        self._volume = self._open_or_build_streaming_cache(probe)
        del probe, vol_f64

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        z, x, y = self.volume_shape
        sz, sx, sy = self.sample_shape
        n_positions = max(1, (z - sz + 1)) * max(1, (x - sx + 1)) * max(1, (y - sy + 1))
        if self.epoch_samples is not None:
            return max(1, min(n_positions, self.epoch_samples))
        return n_positions

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sz, sx, sy = self.sample_shape
        z_dim, x_dim, y_dim = self.volume_shape

        # Random crop centre with half-sample margin.
        z0 = random.randint(0, max(0, z_dim - sz))
        x0 = random.randint(0, max(0, x_dim - sx))
        y0 = random.randint(0, max(0, y_dim - sy))

        target_np = np.asarray(self._volume[z0:z0 + sz, x0:x0 + sx, y0:y0 + sy], dtype=np.float32).copy()

        # --- Apply input masking strategy ---
        input_np = self._apply_input_strategy(target_np)

        # --- Trace-cluster dropout ---
        from synthoseis_pre_train.masking import apply_trace_cluster_dropout_to_mask
        from synthoseis_pre_train.masking import create_extrema_mask_3d

        extrema_mask = create_extrema_mask_3d(input_np)
        extrema_mask = apply_trace_cluster_dropout_to_mask(
            extrema_mask,
            trace_mask_ratio=self.trace_mask_ratio,
            cluster_prob=self.cluster_prob,
        )

        input_masked = np.zeros_like(input_np, dtype=np.float32)
        input_masked[extrema_mask] = input_np[extrema_mask]

        return input_masked.astype(np.float32, copy=False), target_np.astype(np.float32, copy=False), extrema_mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_input_strategy(self, data: np.ndarray) -> np.ndarray:
        from synthoseis_pre_train.masking import (
            keep_trace_extrema_only,
            apply_input_random_sparse_keep,
            apply_input_decimate_trilinear,
        )
        idx = int(np.random.choice(np.array([0, 1, 2], dtype=np.int64), p=self._input_strategy_probs))
        if idx == 0:
            return keep_trace_extrema_only(data)
        if idx == 1:
            return apply_input_random_sparse_keep(data, method="poisson")
        return apply_input_decimate_trilinear(data)

    def _open_or_build_streaming_cache(self, probe: np.ndarray):
        """Open a normalized zarr cache for low-RAM random crop access.

        The cache is rebuilt when source metadata does not match.
        """
        cache_key = str(self._path.resolve())
        source_mtime_ns = int(self._path.stat().st_mtime_ns)
        source_size = int(self._path.stat().st_size)

        if self._cache_path.exists():
            try:
                root = zarr.open_group(str(self._cache_path), mode="r")
                meta = root.attrs
                vol = root["volume"]
                meta_ok = (
                    str(meta.get("source_path", "")) == cache_key
                    and int(meta.get("source_mtime_ns", -1)) == source_mtime_ns
                    and int(meta.get("source_size", -1)) == source_size
                    and tuple(int(v) for v in vol.shape) == self.volume_shape
                )
                if meta_ok:
                    return vol
            except Exception:
                pass

        self._build_cache_from_probe(
            probe=probe,
            cache_key=cache_key,
            source_mtime_ns=source_mtime_ns,
            source_size=source_size,
        )
        root = zarr.open_group(str(self._cache_path), mode="r")
        return root["volume"]

    def _build_cache_from_probe(
        self,
        probe: np.ndarray,
        cache_key: str,
        source_mtime_ns: int,
        source_size: int,
    ) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(self._cache_path), mode="w")

        z_dim, x_dim, y_dim = self.volume_shape
        sz, sx, sy = self.sample_shape
        chunks = (
            max(1, min(int(sz), z_dim)),
            max(1, min(int(sx), x_dim)),
            max(1, min(int(sy), y_dim)),
        )

        volume = root.create_array(
            "volume",
            shape=self.volume_shape,
            chunks=chunks,
            dtype="f4",
            overwrite=True,
        )

        for z0 in range(0, z_dim, chunks[0]):
            z1 = min(z_dim, z0 + chunks[0])
            for x0 in range(0, x_dim, chunks[1]):
                x1 = min(x_dim, x0 + chunks[1])
                for y0 in range(0, y_dim, chunks[2]):
                    y1 = min(y_dim, y0 + chunks[2])
                    block = np.asarray(probe[z0:z1, x0:x1, y0:y1], dtype=np.float32)
                    if self._needs_unit_shift:
                        block = block - 0.5
                    if self._volume_std > 0.0:
                        block = block / self._volume_std
                    volume[z0:z1, x0:x1, y0:y1] = block

        root.attrs.update(
            {
                "source_path": cache_key,
                "source_mtime_ns": int(source_mtime_ns),
                "source_size": int(source_size),
                "volume_std": float(self._volume_std),
                "needs_unit_shift": bool(self._needs_unit_shift),
            }
        )
