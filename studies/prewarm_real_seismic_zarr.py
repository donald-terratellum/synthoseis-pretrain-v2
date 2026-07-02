#!/usr/bin/env python3
"""Prewarm real seismic .npy files into streaming .real.zarr caches.

This script creates (or refreshes) sibling .real.zarr caches used by
NpySeismicDataset so training can stream random crops from disk with low RAM.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from synthoseis_pre_train._npy_dataset import NpySeismicDataset


DEFAULT_TRAIN_PATH = "/Users/donaldpg/synthoseis/real_data"
DEFAULT_TEST_PATH = "/Users/donaldpg/synthoseis/fake_data/test"


@dataclass(frozen=True)
class PrewarmResult:
    source: Path
    cache: Path
    elapsed_sec: float
    status: str
    message: str = ""


def _collect_npy_files(paths: list[str], recursive: bool = True) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.suffix == ".npy":
            found.append(path)
            continue
        if path.is_dir():
            iterator = path.rglob("*.npy") if recursive else path.glob("*.npy")
            found.extend(p.resolve() for p in iterator if p.is_file())
            continue
        print(f"WARNING: skipping missing or unsupported path: {path}")

    # Deduplicate while preserving sorted order for deterministic output.
    return sorted(set(found))


def _prewarm_one(npy_path: Path, sample_shape: tuple[int, int, int], dry_run: bool = False) -> PrewarmResult:
    cache_path = npy_path.with_suffix(".real.zarr")
    start = time.monotonic()
    try:
        if dry_run:
            elapsed = time.monotonic() - start
            return PrewarmResult(
                source=npy_path,
                cache=cache_path,
                elapsed_sec=elapsed,
                status="dry-run",
            )

        # Instantiation triggers one-time cache build if cache is missing/stale.
        NpySeismicDataset(
            str(npy_path),
            sample_shape=sample_shape,
            augment=False,
            epoch_samples=1,
        )
        elapsed = time.monotonic() - start
        if cache_path.exists():
            return PrewarmResult(source=npy_path, cache=cache_path, elapsed_sec=elapsed, status="ok")
        return PrewarmResult(
            source=npy_path,
            cache=cache_path,
            elapsed_sec=elapsed,
            status="error",
            message="cache was not created",
        )
    except Exception as exc:  # pragma: no cover - defensive runtime reporting
        elapsed = time.monotonic() - start
        return PrewarmResult(
            source=npy_path,
            cache=cache_path,
            elapsed_sec=elapsed,
            status="error",
            message=str(exc),
        )


def _run_group(group_name: str, npy_files: list[Path], sample_shape: tuple[int, int, int], dry_run: bool) -> list[PrewarmResult]:
    print()
    print(f"=== {group_name} ===")
    print(f"Found {len(npy_files)} .npy files")
    results: list[PrewarmResult] = []
    for idx, npy_path in enumerate(npy_files, start=1):
        print(f"[{idx:04d}/{len(npy_files):04d}] {npy_path}")
        result = _prewarm_one(npy_path, sample_shape=sample_shape, dry_run=dry_run)
        results.append(result)
        if result.status == "error":
            print(f"  ERROR ({result.elapsed_sec:.2f}s): {result.message}")
        else:
            print(f"  {result.status} ({result.elapsed_sec:.2f}s): {result.cache}")
    return results


def _summarize(all_results: list[PrewarmResult]) -> int:
    total = len(all_results)
    ok = sum(1 for r in all_results if r.status == "ok")
    dry = sum(1 for r in all_results if r.status == "dry-run")
    err = sum(1 for r in all_results if r.status == "error")
    elapsed = sum(r.elapsed_sec for r in all_results)

    print()
    print("=== Summary ===")
    print(f"Total: {total}")
    print(f"OK: {ok}")
    print(f"Dry-run: {dry}")
    print(f"Errors: {err}")
    print(f"Elapsed: {elapsed:.2f}s")

    if err:
        print()
        print("Failed files:")
        for r in all_results:
            if r.status == "error":
                print(f"- {r.source}: {r.message}")
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-time conversion of real seismic .npy files into .real.zarr caches "
            "for low-RAM streaming during training and test evaluation."
        )
    )
    parser.add_argument(
        "--train-paths",
        nargs="*",
        default=[DEFAULT_TRAIN_PATH],
        help="Train .npy file(s) or folder(s) to prewarm (default: real_data root).",
    )
    parser.add_argument(
        "--test-paths",
        nargs="*",
        default=[DEFAULT_TEST_PATH],
        help="Test .npy file(s) or folder(s) to prewarm (default: fake_data/test root).",
    )
    parser.add_argument(
        "--sample-shape",
        nargs=3,
        type=int,
        default=[128, 128, 128],
        metavar=("Z", "X", "Y"),
        help="Crop shape used to validate compatibility during prewarm (default: 128 128 128).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan top-level directories for .npy files (default scans recursively).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and intended cache targets without creating caches.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    sample_shape = tuple(int(v) for v in args.sample_shape)
    recursive = not bool(args.non_recursive)
    dry_run = bool(args.dry_run)

    train_files = _collect_npy_files(list(args.train_paths), recursive=recursive)
    test_files = _collect_npy_files(list(args.test_paths), recursive=recursive)

    train_results = _run_group("Training", train_files, sample_shape=sample_shape, dry_run=dry_run)
    test_results = _run_group("Test", test_files, sample_shape=sample_shape, dry_run=dry_run)

    return _summarize(train_results + test_results)


if __name__ == "__main__":
    raise SystemExit(main())
