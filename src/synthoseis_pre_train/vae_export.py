"""Export VAE patch datasets from Synthoseis model_data.zarr stores.

This module builds a destination zarr with one dataset named ``patches``
having shape ``(n_subsets, x, y, z)`` and dtype ``float32``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr


DEFAULT_SUBSET_SIZE_XYZ = (32, 32, 64)
DEFAULT_N_SUBSETS = 2500
DEFAULT_VAE_OUTPUT_ROOT = Path("/Users/donaldpg/synthoseis-3dvae-poc/data")
DEFAULT_GEOLOGIC_SCORE_KEYS = ("geological_score", "geologic_score")
DEFAULT_SEISMIC_KEY = "seismicCubes_cumsum_fullstack"
DEFAULT_SCORE_MIN = 0.5
DEFAULT_CANDIDATE_COUNT = 3000
DEFAULT_CANDIDATE_PROBES = 24


def default_radius_from_subset_size(subset_size_xyz: tuple[int, int, int]) -> float:
    """Return default geologic distance threshold from subset dimensions."""
    return float(min(subset_size_xyz)) / 3.0


def default_output_zarr_path(dataset_id: int, output_root: Path) -> Path:
    """Build default destination path: train_XXXX.zarr."""
    return output_root / f"train_{dataset_id:04d}.zarr"


def infer_dataset_id_from_path(dataset_zarr: Path) -> Optional[int]:
    """Best-effort dataset id extraction from zarr path or parent name."""
    text = str(dataset_zarr)
    match = re.search(r"synthoseis_run_(\d{4})", text)
    if match:
        return int(match.group(1))
    return None


def _resolve_score_key(root: zarr.Group, candidates: Iterable[str]) -> str:
    keys = set(root.array_keys())
    for key in candidates:
        if key in keys and len(root[key].shape) == 3:
            return key
    raise KeyError(f"No geologic score array found from candidates: {list(candidates)}")


def _resolve_seismic_key(root: zarr.Group, preferred_key: str, score_key: str) -> str:
    keys = list(root.array_keys())
    if preferred_key in keys and len(root[preferred_key].shape) == 3:
        return preferred_key

    for key in keys:
        if key == score_key:
            continue
        if len(root[key].shape) == 3 and key.startswith("seismic"):
            return key

    for key in keys:
        if key == score_key:
            continue
        if len(root[key].shape) == 3:
            return key

    raise KeyError("No usable 3D seismic volume key found.")


def _center_bounds(
    shape_xyz: tuple[int, int, int],
    subset_size_xyz: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    margin = np.asarray([v // 2 for v in subset_size_xyz], dtype=np.int32)
    low = margin
    high = np.asarray(shape_xyz, dtype=np.int32) - margin - 1
    if np.any(high < low):
        raise ValueError(
            f"Dataset shape {shape_xyz} too small for subset size {subset_size_xyz}."
        )
    return low, high


def _random_point_within(
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.array(
        [rng.integers(int(lo), int(hi) + 1) for lo, hi in zip(low, high)],
        dtype=np.int32,
    )


def _best_candidate_point(
    existing: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
    probe_count: int,
) -> np.ndarray:
    probes = np.stack(
        [_random_point_within(low, high, rng) for _ in range(probe_count)],
        axis=0,
    )
    if existing.size == 0:
        return probes[0]

    diffs = probes[:, None, :].astype(np.float32) - existing[None, :, :].astype(np.float32)
    min_distances = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
    return probes[int(np.argmax(min_distances))]


def _generate_candidate_points(
    shape_xyz: tuple[int, int, int],
    subset_size_xyz: tuple[int, int, int],
    candidate_count: int,
    probe_count: int,
    seed: int,
) -> np.ndarray:
    low, high = _center_bounds(shape_xyz, subset_size_xyz)
    rng = np.random.default_rng(seed)

    points: list[np.ndarray] = []
    existing = np.empty((0, 3), dtype=np.int32)
    for _ in range(candidate_count):
        point = _best_candidate_point(existing, low, high, rng, probe_count)
        points.append(point)
        existing = np.asarray(points, dtype=np.int32)
    return existing


def _select_centers_by_score_and_radius(
    points_xyz: np.ndarray,
    score_values: np.ndarray,
    n_subsets: int,
    min_score: float,
    radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = score_values >= float(min_score)
    points = points_xyz[keep]
    scores = score_values[keep]

    if len(points) == 0:
        points = points_xyz
        scores = score_values

    if len(points) == 0:
        raise ValueError("No candidate points were generated.")

    order = np.argsort(scores)[::-1]
    ordered_points = points[order]
    ordered_scores = scores[order]

    accepted_points: list[np.ndarray] = [ordered_points[0]]
    accepted_scores: list[float] = [float(ordered_scores[0])]

    radius = float(max(radius, 0.0))
    for idx in range(1, len(ordered_points)):
        if len(accepted_points) >= n_subsets:
            break
        point = ordered_points[idx]
        reference = np.asarray(accepted_points, dtype=np.int32)
        diffs = reference.astype(np.float32) - point.astype(np.float32)
        min_dist = float(np.sqrt(np.sum(diffs * diffs, axis=1)).min())
        if min_dist > radius:
            accepted_points.append(point)
            accepted_scores.append(float(ordered_scores[idx]))

    if len(accepted_points) < n_subsets:
        rng = np.random.default_rng(seed + 17)
        probs = np.asarray(ordered_scores, dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        if probs.sum() <= 0:
            probs = np.full(len(ordered_points), 1.0 / len(ordered_points), dtype=np.float64)
        else:
            probs = probs / probs.sum()

        needed = n_subsets - len(accepted_points)
        extra_idx = rng.choice(len(ordered_points), size=needed, replace=True, p=probs)
        for idx in extra_idx:
            accepted_points.append(ordered_points[int(idx)])
            accepted_scores.append(float(ordered_scores[int(idx)]))

    return (
        np.asarray(accepted_points[:n_subsets], dtype=np.int32),
        np.asarray(accepted_scores[:n_subsets], dtype=np.float32),
    )


def _extract_patch(
    volume_xyz: np.ndarray,
    center_xyz: np.ndarray,
    subset_size_xyz: tuple[int, int, int],
) -> np.ndarray:
    sx, sy, sz = subset_size_xyz
    x0 = int(center_xyz[0]) - sx // 2
    y0 = int(center_xyz[1]) - sy // 2
    z0 = int(center_xyz[2]) - sz // 2
    patch = volume_xyz[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz]
    if patch.shape != (sx, sy, sz):
        raise RuntimeError(
            f"Patch shape mismatch at center {tuple(int(v) for v in center_xyz)}: "
            f"got {patch.shape}, expected {(sx, sy, sz)}"
        )
    return np.asarray(patch, dtype=np.float32)


def _create_or_get_patches_array(dst: zarr.Group, shape: tuple[int, int, int, int]):
    chunks = (1, shape[1], shape[2], shape[3])
    if hasattr(dst, "create_array"):
        return dst.create_array("patches", shape=shape, dtype="f4", chunks=chunks)
    return dst.create_dataset("patches", shape=shape, dtype="f4", chunks=chunks)


def export_vae_dataset(
    dataset_zarr_path: Path,
    dataset_id: int,
    subset_size_xyz: tuple[int, int, int] = DEFAULT_SUBSET_SIZE_XYZ,
    n_subsets: int = DEFAULT_N_SUBSETS,
    radius: Optional[float] = None,
    output_zarr_path: Optional[Path] = None,
    vae_output_root: Path = DEFAULT_VAE_OUTPUT_ROOT,
    score_key_candidates: tuple[str, ...] = DEFAULT_GEOLOGIC_SCORE_KEYS,
    seismic_key: str = DEFAULT_SEISMIC_KEY,
    score_min: float = DEFAULT_SCORE_MIN,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    candidate_probes: int = DEFAULT_CANDIDATE_PROBES,
    seed: int = 7,
) -> Path:
    """Export one VAE-ready patch zarr for a single model_data.zarr dataset."""
    dataset_zarr_path = Path(dataset_zarr_path)
    if not dataset_zarr_path.exists():
        raise FileNotFoundError(f"Dataset zarr not found: {dataset_zarr_path}")

    if n_subsets <= 0:
        raise ValueError("n_subsets must be > 0")

    if any(v <= 0 for v in subset_size_xyz):
        raise ValueError(f"subset_size_xyz must contain positive values: {subset_size_xyz}")

    if radius is None:
        radius = default_radius_from_subset_size(subset_size_xyz)

    output_zarr = (
        Path(output_zarr_path)
        if output_zarr_path is not None
        else default_output_zarr_path(dataset_id, Path(vae_output_root))
    )
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(dataset_zarr_path), mode="r")
    score_key = _resolve_score_key(root, score_key_candidates)
    chosen_seismic_key = _resolve_seismic_key(root, seismic_key, score_key)

    score_arr = root[score_key]
    seismic_arr = root[chosen_seismic_key]

    usable_shape = tuple(
        min(int(a), int(b)) for a, b in zip(score_arr.shape, seismic_arr.shape)
    )
    candidate_count = int(max(candidate_count, n_subsets))
    points_xyz = _generate_candidate_points(
        shape_xyz=usable_shape,
        subset_size_xyz=subset_size_xyz,
        candidate_count=candidate_count,
        probe_count=int(candidate_probes),
        seed=int(seed),
    )

    score_values = np.asarray(
        score_arr[points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]],
        dtype=np.float32,
    )
    centers_xyz, center_scores = _select_centers_by_score_and_radius(
        points_xyz=points_xyz,
        score_values=score_values,
        n_subsets=int(n_subsets),
        min_score=float(score_min),
        radius=float(radius),
        seed=int(seed),
    )

    dst = zarr.open(str(output_zarr), mode="w")
    patches = _create_or_get_patches_array(
        dst,
        shape=(int(n_subsets), int(subset_size_xyz[0]), int(subset_size_xyz[1]), int(subset_size_xyz[2])),
    )

    for idx, center in enumerate(centers_xyz):
        patches[idx] = _extract_patch(np.asarray(seismic_arr), center, subset_size_xyz)

    dst.attrs["source_dataset"] = str(dataset_zarr_path)
    dst.attrs["dataset_id"] = int(dataset_id)
    dst.attrs["subset_size_xyz"] = [int(v) for v in subset_size_xyz]
    dst.attrs["radius"] = float(radius)
    dst.attrs["n_subsets"] = int(n_subsets)
    dst.attrs["score_key_used"] = str(score_key)
    dst.attrs["seismic_key_used"] = str(chosen_seismic_key)
    dst.attrs["selected_center_count"] = int(len(centers_xyz))
    dst.attrs["selected_score_mean"] = float(np.mean(center_scores)) if len(center_scores) else 0.0

    return output_zarr


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VAE patch zarr from model_data.zarr")
    parser.add_argument("--dataset-zarr", required=True, help="Path to source model_data.zarr")
    parser.add_argument("--dataset-id", type=int, default=None, help="Dataset id (used in default output naming)")

    parser.add_argument("--subset-size-x", type=int, default=DEFAULT_SUBSET_SIZE_XYZ[0])
    parser.add_argument("--subset-size-y", type=int, default=DEFAULT_SUBSET_SIZE_XYZ[1])
    parser.add_argument("--subset-size-z", type=int, default=DEFAULT_SUBSET_SIZE_XYZ[2])

    parser.add_argument("--radius", type=float, default=None, help="Distance threshold in voxels")
    parser.add_argument("--n-subsets", type=int, default=DEFAULT_N_SUBSETS)
    parser.add_argument("--output-zarr", type=str, default=None)
    parser.add_argument("--vae-output-root", type=str, default=str(DEFAULT_VAE_OUTPUT_ROOT))

    parser.add_argument("--score-min", type=float, default=DEFAULT_SCORE_MIN)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--candidate-probes", type=int, default=DEFAULT_CANDIDATE_PROBES)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--seismic-key", type=str, default=DEFAULT_SEISMIC_KEY)
    parser.add_argument(
        "--geoscore-key-candidates",
        type=str,
        default=",".join(DEFAULT_GEOLOGIC_SCORE_KEYS),
        help="Comma-separated geologic score key candidates",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    subset_size_xyz = (int(args.subset_size_x), int(args.subset_size_y), int(args.subset_size_z))

    dataset_zarr = Path(args.dataset_zarr)
    dataset_id = args.dataset_id
    if dataset_id is None:
        dataset_id = infer_dataset_id_from_path(dataset_zarr)
    if dataset_id is None:
        raise ValueError(
            "Unable to infer dataset id from path; provide --dataset-id explicitly."
        )

    key_candidates = tuple(
        key.strip() for key in str(args.geoscore_key_candidates).split(",") if key.strip()
    )
    if not key_candidates:
        raise ValueError("--geoscore-key-candidates must include at least one key")

    output = export_vae_dataset(
        dataset_zarr_path=dataset_zarr,
        dataset_id=int(dataset_id),
        subset_size_xyz=subset_size_xyz,
        n_subsets=int(args.n_subsets),
        radius=args.radius,
        output_zarr_path=Path(args.output_zarr) if args.output_zarr else None,
        vae_output_root=Path(args.vae_output_root),
        score_key_candidates=key_candidates,
        seismic_key=str(args.seismic_key),
        score_min=float(args.score_min),
        candidate_count=int(args.candidate_count),
        candidate_probes=int(args.candidate_probes),
        seed=int(args.seed),
    )

    print(f"Wrote VAE dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
