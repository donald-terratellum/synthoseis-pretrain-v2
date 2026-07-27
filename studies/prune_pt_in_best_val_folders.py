#!/usr/bin/env python3
"""Prune .pt files in folders containing best_val_epoch.pt.

Rule:
- Keep best_val_epoch.pt
- Keep checkpoint_epoch_XXXX.pt only when epoch is a multiple of N
    (default N=5, configurable via --keep-every)
- Delete all other .pt files in those folders

Safety:
- Dry-run by default
- Use --apply to perform deletions

This script preserves the resumable end-of-training artifacts:
- best_val_epoch.pt
- final_model.pt
- checkpoint_final_model.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EPOCH_RE = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")
DEFAULT_CHECKPOINTS_ROOT = Path("/Volumes/Crucial X9/pretrain_v2_checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete .pt files in directories containing best_val_epoch.pt, "
            "keeping only best_val_epoch.pt and checkpoint_epoch_XXXX.pt where XXXX % N == 0"
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_CHECKPOINTS_ROOT,
        help=f"Root folder to scan for best_val_epoch.pt files (default: {DEFAULT_CHECKPOINTS_ROOT})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=5,
        help="Keep checkpoint_epoch_XXXX.pt only when XXXX is a multiple of this value (default: 5)",
    )
    return parser.parse_args()


def should_keep(pt_file: Path, keep_every: int = 5) -> bool:
    name = pt_file.name
    if name in ("best_val_epoch.pt", "final_model.pt", "checkpoint_final_model.pt"):
        return True

    m = EPOCH_RE.match(name)
    if m is None:
        return False

    epoch = int(m.group(1))
    return epoch % int(keep_every) == 0


def main() -> int:
    args = parse_args()
    root = args.root
    keep_every = int(args.keep_every)

    if keep_every <= 0:
        print(f"ERROR: --keep-every must be > 0, got: {keep_every}")
        return 2

    if not root.exists() or not root.is_dir():
        print(f"ERROR: root does not exist or is not a directory: {root}")
        return 2

    best_files = sorted(root.rglob("best_val_epoch.pt"))
    if not best_files:
        print(f"No best_val_epoch.pt files found under {root}")
        return 0

    deleted = 0
    kept = 0
    scanned = 0
    bytes_recovered = 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Found {len(best_files)} folders containing best_val_epoch.pt")

    for best_file in best_files:
        folder = best_file.parent
        print(f"\nFolder: {folder}")

        for pt_file in sorted(folder.glob("*.pt")):
            scanned += 1
            file_size = pt_file.stat().st_size
            if should_keep(pt_file, keep_every=keep_every):
                kept += 1
                print(f"  KEEP   {pt_file.name}  ({file_size / 1024**2:.1f} MB)")
                continue

            bytes_recovered += file_size
            if args.apply:
                pt_file.unlink(missing_ok=True)
                print(f"  DELETE {pt_file.name}  ({file_size / 1024**2:.1f} MB)")
            else:
                print(f"  WOULD_DELETE {pt_file.name}  ({file_size / 1024**2:.1f} MB)")
            deleted += 1

    recovered_gb = bytes_recovered / 1024**3
    print(
        f"\nDone. folders={len(best_files)} scanned_pt={scanned} kept={kept} "
        f"{'deleted' if args.apply else 'would_delete'}={deleted} "
        f"space_{'recovered' if args.apply else 'recoverable'}={recovered_gb:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
