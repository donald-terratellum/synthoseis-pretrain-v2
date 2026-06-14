"""
Seismic 3D masking and input corruption strategies.

Tensor convention is (z, x, y) across this module.
"""

from __future__ import annotations

import numpy as np
import time
from typing import Tuple, Optional


def _get_rng(random_seed: Optional[int]):
    return np.random.RandomState(random_seed) if random_seed is not None else np.random


def create_extrema_mask_3d(seismic_data: np.ndarray) -> np.ndarray:
    """Build a mask that keeps only local extrema along z for each (x, y) trace."""
    z, x, y = seismic_data.shape
    mask = np.zeros((z, x, y), dtype=bool)
    if z <= 2:
        return mask

    # When squeeze/t2d leaves zero planes at z boundaries, fill the immediately
    # adjacent zero planes to avoid false extrema at the 0->nonzero transition.
    nonzero_z = np.where(seismic_data.any(axis=(1, 2)))[0]
    boundary_lo = None
    boundary_hi = None
    if nonzero_z.size > 0:
        z_lo = int(nonzero_z[0])
        z_hi = int(nonzero_z[-1])
        if z_lo > 0 or z_hi < z - 1:
            work = seismic_data.copy()
            if z_lo > 0:
                work[z_lo - 1] = work[z_lo]
                boundary_lo = z_lo - 1
            if z_hi < z - 1:
                work[z_hi + 1] = work[z_hi]
                boundary_hi = z_hi + 1
        else:
            work = seismic_data
    else:
        work = seismic_data

    is_peak = (work[1:-1, :, :] > work[:-2, :, :]) & (work[1:-1, :, :] > work[2:, :, :])
    is_trough = (work[1:-1, :, :] < work[:-2, :, :]) & (work[1:-1, :, :] < work[2:, :, :])
    mask[1:-1, :, :] = is_peak | is_trough

    if boundary_lo is not None:
        mask[boundary_lo, :, :] = False
    if boundary_hi is not None:
        mask[boundary_hi, :, :] = False

    return mask


def apply_trace_cluster_dropout_to_mask(
    mask: np.ndarray,
    trace_mask_ratio: float = 0.07,
    cluster_prob: float = 0.8,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Apply 3x3 XY cluster trace dropout by setting whole traces to False."""
    z, x, y = mask.shape
    del z

    if trace_mask_ratio <= 0.0 or cluster_prob <= 0.0:
        return mask

    rng = _get_rng(random_seed)
    n_total = x * y
    n_traces_to_mask = int(n_total * float(trace_mask_ratio))
    if n_traces_to_mask <= 0:
        return mask
    n_traces_to_mask = min(n_traces_to_mask, n_total)

    trace_indices = rng.choice(n_total, size=n_traces_to_mask, replace=False)
    for idx in trace_indices:
        trace_x = int(idx % x)
        trace_y = int(idx // x)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx = trace_x + dx
                cy = trace_y + dy
                if 0 <= cx < x and 0 <= cy < y and rng.random() < cluster_prob:
                    mask[:, cx, cy] = False
    return mask


def create_mask_3d(
    seismic_data: np.ndarray,
    trace_mask_ratio: float = 0.07,
    cluster_prob: float = 0.8,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Backward-compatible mask: extrema retention followed by cluster trace dropout."""
    mask = create_extrema_mask_3d(seismic_data)
    return apply_trace_cluster_dropout_to_mask(
        mask,
        trace_mask_ratio=trace_mask_ratio,
        cluster_prob=cluster_prob,
        random_seed=random_seed,
    )


def keep_trace_extrema_only(seismic_data: np.ndarray) -> np.ndarray:
    """Keep only z-local extrema values, zeroing all non-extrema voxels."""
    # t0 = time.perf_counter()  # TODO: remove this line
    extrema_mask = create_extrema_mask_3d(seismic_data)
    out = np.zeros_like(seismic_data, dtype=np.float32)
    out[extrema_mask] = np.asarray(seismic_data, dtype=np.float32)[extrema_mask]
    # elapsed_ms = (time.perf_counter() - t0) * 1000.0  # TODO: remove this line
    # print(f"[diag] keep_trace_extrema_only elapsed_ms={elapsed_ms:.3f}")  # TODO: remove this line
    return out


def _flat_to_zyx(flat_idx: int, x_dim: int, y_dim: int) -> tuple[int, int, int]:
    z_idx = int(flat_idx // (x_dim * y_dim))
    rem = int(flat_idx % (x_dim * y_dim))
    x_idx = int(rem // y_dim)
    y_idx = int(rem % y_dim)
    return z_idx, x_idx, y_idx


def _build_sparse_indices_poisson_like(
    shape: tuple[int, int, int],
    target_count: int,
    radius_scale: float,
) -> np.ndarray:
    # t0 = time.perf_counter()  # TODO: remove this line
    z_dim, x_dim, y_dim = (int(shape[0]), int(shape[1]), int(shape[2]))
    nvox = int(z_dim * x_dim * y_dim)
    if target_count >= nvox:
        out = np.arange(nvox, dtype=np.int64)
        # elapsed_ms = (time.perf_counter() - t0) * 1000.0  # TODO: remove this line
        # print(f"[diag] _build_sparse_indices_poisson_like elapsed_ms={elapsed_ms:.3f}")  # TODO: remove this line
        return out

    keep_fraction = float(target_count) / float(max(nvox, 1))
    # High-density keep fractions are pathological for rejection-based Poisson
    # sampling on large grids. Fall back to uniform sparse selection to keep
    # runtime bounded while preserving expected keep ratio.
    if keep_fraction >= 0.08:
        out = _build_sparse_indices_uniform_threshold((z_dim, x_dim, y_dim), keep_fraction)
        if out.size > target_count:
            np.random.shuffle(out)
            out = out[:target_count]
        elif out.size < target_count:
            selected_mask = np.zeros(nvox, dtype=bool)
            selected_mask[out] = True
            remaining = np.flatnonzero(~selected_mask)
            np.random.shuffle(remaining)
            needed = target_count - out.size
            out = np.concatenate([out, remaining[:needed].astype(np.int64, copy=False)])
        # elapsed_ms = (time.perf_counter() - t0) * 1000.0  # TODO: remove this line
        # print(f"[diag] _build_sparse_indices_poisson_like elapsed_ms={elapsed_ms:.3f}")  # TODO: remove this line
        return out.astype(np.int64, copy=False)

    radius = float(radius_scale) * float(np.cbrt(float(nvox) / float(target_count)))
    radius2 = radius * radius
    cell_size = max(radius / float(np.sqrt(3.0)), 1e-6)
    neighbor_span = int(np.ceil(radius / cell_size))
    max_checks = min(nvox, max(4096, 2 * int(target_count)))

    buckets: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    selected = []
    selected_mask = np.zeros(nvox, dtype=bool)
    candidate_stream = np.random.permutation(nvox)
    checks = 0

    for flat_idx in candidate_stream:
        if checks >= max_checks or len(selected) >= target_count:
            break
        checks += 1

        cz, cx, cy = _flat_to_zyx(int(flat_idx), x_dim, y_dim)
        bz = int(np.floor(cz / cell_size))
        bx = int(np.floor(cx / cell_size))
        by = int(np.floor(cy / cell_size))

        keep = True
        for dz in range(-neighbor_span, neighbor_span + 1):
            for dx in range(-neighbor_span, neighbor_span + 1):
                for dy in range(-neighbor_span, neighbor_span + 1):
                    key = (bz + dz, bx + dx, by + dy)
                    if key not in buckets:
                        continue
                    oz, ox, oy = buckets[key]
                    dist2 = float((cz - oz) ** 2 + (cx - ox) ** 2 + (cy - oy) ** 2)
                    if dist2 < radius2:
                        keep = False
                        break
                    if not keep:
                        break
                if not keep:
                    break
            if not keep:
                break

        if not keep:
            continue

        selected.append(int(flat_idx))
        selected_mask[int(flat_idx)] = True
        key = (bz, bx, by)
        buckets[key] = (cz, cx, cy)

    if len(selected) < target_count:
        remaining = np.flatnonzero(~selected_mask)
        np.random.shuffle(remaining)
        needed = target_count - len(selected)
        selected.extend(int(v) for v in remaining[:needed])

    out = np.asarray(selected[:target_count], dtype=np.int64)
    # elapsed_ms = (time.perf_counter() - t0) * 1000.0  # TODO: remove this line
    # print(f"[diag] _build_sparse_indices_poisson_like elapsed_ms={elapsed_ms:.3f}")  # TODO: remove this line
    return out


def _build_sparse_indices_uniform_threshold(shape: tuple[int, int, int], keep_fraction: float) -> np.ndarray:
    # t0 = time.perf_counter()  # TODO: remove this line
    probs = np.random.uniform(0.0, 1.0, tuple(int(v) for v in shape))
    out = np.flatnonzero((probs <= float(keep_fraction)).reshape(-1)).astype(np.int64, copy=False)
    # elapsed_ms = (time.perf_counter() - t0) * 1000.0  # TODO: remove this line
    # print(f"[diag] _build_sparse_indices_uniform_threshold elapsed_ms={elapsed_ms:.3f}")  # TODO: remove this line
    return out


def apply_input_random_sparse_keep(
    seismic_data: np.ndarray,
    fraction_min: float = 0.10,
    fraction_max: float = 0.30,
    method: str = "poisson",
    poisson_radius_scale: float = 0.85,
) -> np.ndarray:
    """Keep a sparse set of voxels and zero the rest."""
    # t0 = time.perf_counter()  # TODO: remove this line
    z_dim, x_dim, y_dim = (int(seismic_data.shape[0]), int(seismic_data.shape[1]), int(seismic_data.shape[2]))
    nvox = int(z_dim * x_dim * y_dim)

    if float(fraction_min) == float(fraction_max):
        keep_fraction = float(fraction_min)
    else:
        keep_fraction = float(np.random.uniform(float(fraction_min), float(fraction_max)))
    keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))

    target_count = int(np.clip(np.rint(keep_fraction * nvox), 1, nvox))

    selected_method = method
    if selected_method == "random":
        selected_method = "poisson" if np.random.random() < 0.5 else "uniform"

    if selected_method == "poisson":
        selected = _build_sparse_indices_poisson_like((z_dim, x_dim, y_dim), target_count, poisson_radius_scale)
    elif selected_method == "uniform":
        selected = _build_sparse_indices_uniform_threshold((z_dim, x_dim, y_dim), keep_fraction)
    else:
        raise ValueError("method must be one of: 'random', 'poisson', 'uniform'.")

    flat_in = np.ascontiguousarray(seismic_data, dtype=np.float32).reshape(-1)
    out = np.zeros(seismic_data.shape, dtype=np.float32)
    flat_out = out.reshape(-1)
    flat_out[selected] = flat_in[selected]
    # elapsed_ms = (time.perf_counter() - t0) * 1000.0  # TODO: remove this line
    # print(f"[diag] apply_input_random_sparse_keep elapsed_ms={elapsed_ms:.3f}")  # TODO: remove this line
    return out


def _parity_indices(length: int, parity: int) -> np.ndarray:
    idx = np.arange(int(parity), int(length), 2, dtype=np.int64)
    if idx.size == 0:
        return np.array([0], dtype=np.int64)
    return idx


def apply_input_decimate_trilinear(seismic_data: np.ndarray, parity: Optional[int] = None) -> np.ndarray:
    """Decimate by parity in z/x/y and reconstruct by trilinear interpolation."""
    # t0 = time.perf_counter()  # TODO: remove this line
    z_dim, x_dim, y_dim = (int(seismic_data.shape[0]), int(seismic_data.shape[1]), int(seismic_data.shape[2]))
    p = int(np.random.randint(0, 2)) if parity is None else int(parity)
    if p not in (0, 1):
        raise ValueError("parity must be 0 or 1 when provided.")

    idx_z = _parity_indices(z_dim, p)
    idx_x = _parity_indices(x_dim, p)
    idx_y = _parity_indices(y_dim, p)

    anchors = seismic_data[np.ix_(idx_z, idx_x, idx_y)].astype(np.float32, copy=False)

    full_y = np.arange(y_dim, dtype=np.float32)
    full_x = np.arange(x_dim, dtype=np.float32)
    full_z = np.arange(z_dim, dtype=np.float32)
    anchor_y = idx_y.astype(np.float32)
    anchor_x = idx_x.astype(np.float32)
    anchor_z = idx_z.astype(np.float32)

    interp_y = np.empty((idx_z.size, idx_x.size, y_dim), dtype=np.float32)
    for iz in range(idx_z.size):
        for ix in range(idx_x.size):
            interp_y[iz, ix, :] = np.interp(
                full_y,
                anchor_y,
                anchors[iz, ix, :],
                left=float(anchors[iz, ix, 0]),
                right=float(anchors[iz, ix, -1]),
            ).astype(np.float32)

    interp_x = np.empty((idx_z.size, x_dim, y_dim), dtype=np.float32)
    for iz in range(idx_z.size):
        for iy in range(y_dim):
            interp_x[iz, :, iy] = np.interp(
                full_x,
                anchor_x,
                interp_y[iz, :, iy],
                left=float(interp_y[iz, 0, iy]),
                right=float(interp_y[iz, -1, iy]),
            ).astype(np.float32)

    out = np.empty((z_dim, x_dim, y_dim), dtype=np.float32)
    for ix in range(x_dim):
        for iy in range(y_dim):
            out[:, ix, iy] = np.interp(
                full_z,
                anchor_z,
                interp_x[:, ix, iy],
                left=float(interp_x[0, ix, iy]),
                right=float(interp_x[-1, ix, iy]),
            ).astype(np.float32)

    out[np.ix_(idx_z, idx_x, idx_y)] = seismic_data[np.ix_(idx_z, idx_x, idx_y)]
    # elapsed_ms = (time.perf_counter() - t0) * 1000.0  # TODO: remove this line
    # print(f"[diag] apply_input_decimate_trilinear elapsed_ms={elapsed_ms:.3f}")  # TODO: remove this line
    return out


def apply_input_trace_dropout(
    seismic_data: np.ndarray,
    trace_mask_ratio: float = 0.07,
    cluster_prob: float = 0.8,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Zero complete z-traces in 3x3 XY clusters on the input volume."""
    keep_mask = np.ones(seismic_data.shape, dtype=bool)
    keep_mask = apply_trace_cluster_dropout_to_mask(
        keep_mask,
        trace_mask_ratio=trace_mask_ratio,
        cluster_prob=cluster_prob,
        random_seed=random_seed,
    )
    out = np.asarray(seismic_data, dtype=np.float32).copy()
    out[~keep_mask] = 0.0
    return out


def apply_mask_to_seismic(
    seismic_data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a boolean mask to seismic data and return masked/original/mask."""
    masked_data = seismic_data.copy()
    masked_data[~mask] = fill_value

    original_data = seismic_data.copy()
    return masked_data, original_data, mask


def normalize_seismic(
    seismic_data: np.ndarray,
    target_std: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """Normalize seismic amplitudes to target standard deviation."""
    mean = np.mean(seismic_data)
    mean = 0.0  # Centering to zero mean for seismic data
    std = np.std(seismic_data)

    if std > 0:
        normalized = (seismic_data - mean) / std * target_std
    else:
        normalized = seismic_data

    return normalized.astype(np.float32), mean, std
