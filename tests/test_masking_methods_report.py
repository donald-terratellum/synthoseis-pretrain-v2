"""Integration report for masking/corruption methods on a real synthoseis zarr cube.

Usage:
    uv run pytest tests/test_masking_methods_report.py -s -v

This test prints one report line per method with:
  - retained/masked percentages before and after
    - changed percentage after (neither retained nor masked)
  - elapsed wall time in milliseconds
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import zarr

from synthoseis_pre_train.masking import (
    _build_sparse_indices_poisson_like,
    _build_sparse_indices_uniform_threshold,
    apply_input_decimate_trilinear,
    apply_input_random_sparse_keep,
    apply_trace_cluster_dropout_to_mask,
    keep_trace_extrema_only,
)


DEFAULT_DATA_PATH = (
    "/Users/donaldpg/synthoseis/fake_data/validation/"
    "seismic__2026.41133304__synthoseis_run_0570/model_data.zarr"
)
SUBVOLUME_SIZE = 128


def _pct(n: int, total: int) -> float:
    return (100.0 * float(n)) / float(total)


def _find_cumsum_array(root: zarr.Group):
    keys = []
    if hasattr(root, "array_keys"):
        keys = list(root.array_keys())
    if not keys:
        keys = [k for k in list(root.keys()) if hasattr(root[k], "shape")]

    cumsum_keys = [k for k in keys if "cumsum" in str(k).lower()]
    assert cumsum_keys, "No zarr array key containing 'cumsum' was found."
    return root[cumsum_keys[0]], cumsum_keys[0]


def _center_slices(shape_xyz: tuple[int, int, int], side: int) -> tuple[slice, slice, slice]:
    sx, sy, sz = (int(shape_xyz[0]), int(shape_xyz[1]), int(shape_xyz[2]))
    side_x = min(int(side), sx)
    side_y = min(int(side), sy)
    side_z = min(int(side), sz)

    x0 = (sx - side_x) // 2
    y0 = (sy - side_y) // 2
    z0 = (sz - side_z) // 2
    return slice(x0, x0 + side_x), slice(y0, y0 + side_y), slice(z0, z0 + side_z)


@pytest.fixture(scope="module")
def zyx_sample_nonzero() -> np.ndarray:
    path = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    if not os.path.exists(path):
        pytest.skip(f"Test data not found at {path}")

    root = zarr.open(path, mode="r")
    arr, key = _find_cumsum_array(root)
    assert len(arr.shape) == 3, f"Expected 3D array for key '{key}', got shape={arr.shape}"

    # synthoseis stores (x, y, z); masking module expects (z, x, y).
    sx = _center_slices((int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])), SUBVOLUME_SIZE)
    xyz = np.asarray(arr[sx[0], sx[1], sx[2]], dtype=np.float32)
    zxy = np.transpose(xyz, (2, 0, 1)).astype(np.float32, copy=False)

    # Shift away from exact zero so zero is an unambiguous masked sentinel.
    return zxy + np.float32(1.0)


def test_masking_methods_report_on_real_zarr(zyx_sample_nonzero: np.ndarray):
    data = np.ascontiguousarray(zyx_sample_nonzero, dtype=np.float32)
    nvox = int(data.size)
    shape = tuple(int(v) for v in data.shape)

    rows = []

    # 1) _build_sparse_indices_poisson_like
    np.random.seed(101)
    target_count = max(1, int(np.rint(0.2 * nvox)))
    t0 = time.perf_counter()
    idx_poisson = _build_sparse_indices_poisson_like(shape, target_count=target_count, radius_scale=0.85)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    retained_after = int(np.unique(idx_poisson).size)
    changed_after = 0.0
    rows.append(
        (
            "_build_sparse_indices_poisson_like",
            100.0,
            0.0,
            _pct(retained_after, nvox),
            changed_after,
            100.0 - _pct(retained_after, nvox),
            elapsed_ms,
        )
    )

    # 2) keep_trace_extrema_only
    t0 = time.perf_counter()
    out_extrema = keep_trace_extrema_only(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    retained_after = int(np.isclose(out_extrema, data, rtol=0.0, atol=1e-7).sum())
    masked_after = int(np.isclose(out_extrema, 0.0, rtol=0.0, atol=1e-7).sum())
    changed_after = _pct(nvox - retained_after - masked_after, nvox)
    rows.append(
        (
            "keep_trace_extrema_only",
            100.0,
            0.0,
            _pct(retained_after, nvox),
            changed_after,
            _pct(masked_after, nvox),
            elapsed_ms,
        )
    )

    # 3) _build_sparse_indices_uniform_threshold
    np.random.seed(202)
    keep_fraction = 0.2
    t0 = time.perf_counter()
    idx_uniform = _build_sparse_indices_uniform_threshold(shape, keep_fraction=keep_fraction)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    retained_after = int(np.unique(idx_uniform).size)
    changed_after = 0.0
    rows.append(
        (
            "_build_sparse_indices_uniform_threshold",
            100.0,
            0.0,
            _pct(retained_after, nvox),
            changed_after,
            100.0 - _pct(retained_after, nvox),
            elapsed_ms,
        )
    )

    # 4) apply_input_random_sparse_keep
    np.random.seed(303)
    t0 = time.perf_counter()
    out_sparse = apply_input_random_sparse_keep(
        data,
        fraction_min=0.2,
        fraction_max=0.2,
        method="uniform",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    retained_after = int(np.isclose(out_sparse, data, rtol=0.0, atol=1e-7).sum())
    masked_after = int(np.isclose(out_sparse, 0.0, rtol=0.0, atol=1e-7).sum())
    changed_after = _pct(nvox - retained_after - masked_after, nvox)
    rows.append(
        (
            "apply_input_random_sparse_keep",
            100.0,
            0.0,
            _pct(retained_after, nvox),
            changed_after,
            _pct(masked_after, nvox),
            elapsed_ms,
        )
    )

    # 5) apply_input_decimate_trilinear
    t0 = time.perf_counter()
    out_decimate = apply_input_decimate_trilinear(data, parity=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    retained_after = int(np.isclose(out_decimate, data, rtol=0.0, atol=1e-7).sum())
    masked_after = int(np.isclose(out_decimate, 0.0, rtol=0.0, atol=1e-7).sum())
    changed_after = _pct(nvox - retained_after - masked_after, nvox)
    rows.append(
        (
            "apply_input_decimate_trilinear",
            100.0,
            0.0,
            _pct(retained_after, nvox),
            changed_after,
            _pct(masked_after, nvox),
            elapsed_ms,
        )
    )

    # 6) apply_trace_cluster_dropout_to_mask
    base_mask = np.ones(shape, dtype=bool)
    t0 = time.perf_counter()
    out_mask = apply_trace_cluster_dropout_to_mask(
        base_mask.copy(),
        trace_mask_ratio=0.07,
        cluster_prob=0.8,
        random_seed=404,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    retained_after = int(out_mask.sum())
    masked_after = int(out_mask.size - retained_after)
    changed_after = 0.0
    rows.append(
        (
            "apply_trace_cluster_dropout_to_mask",
            100.0,
            0.0,
            _pct(retained_after, nvox),
            changed_after,
            _pct(masked_after, nvox),
            elapsed_ms,
        )
    )

    print("\n[masking-methods-report]")
    print("method | before_retained% | before_masked% | after_retained% | after_changed% | after_masked% | elapsed_ms")
    for method, b_ret, b_msk, a_ret, a_chg, a_msk, ms in rows:
        print(
            f"{method} | {b_ret:.3f} | {b_msk:.3f} | {a_ret:.3f} | {a_chg:.3f} | {a_msk:.3f} | {ms:.3f}"
        )

    # Sanity checks that each method executed and produced reasonable percentages.
    assert len(rows) == 6
    for _, b_ret, b_msk, a_ret, a_chg, a_msk, ms in rows:
        assert 0.0 <= b_ret <= 100.0
        assert 0.0 <= b_msk <= 100.0
        assert 0.0 <= a_ret <= 100.0
        assert 0.0 <= a_chg <= 100.0
        assert 0.0 <= a_msk <= 100.0
        assert abs((a_ret + a_chg + a_msk) - 100.0) < 1e-6
        assert ms >= 0.0