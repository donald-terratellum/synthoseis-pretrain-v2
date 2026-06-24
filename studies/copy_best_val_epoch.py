#!/usr/bin/env python3
"""Copy top validation checkpoints to sibling best_val_epoch.pt files.

For each checkpoint in ``top_10_best_checkpoints``, this script writes:
  <checkpoint_parent>/best_val_epoch.pt

By default it performs real copies; use --dry-run to preview actions.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


top_10_best_checkpoints = [
    "checkpoints/checkpoint_copilot_5/checkpoint_epoch_0034.pt",
    "checkpoints/checkpoint_copilot_2/checkpoint_epoch_0029.pt",
    "checkpoints/checkpoint_copilot_1/checkpoint_epoch_0033.pt",
    "checkpoints/checkpoint_copilot_4/checkpoint_epoch_0035.pt",
    "checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000/checkpoint_epoch_0010.pt",
    "checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001/checkpoint_epoch_0008.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy each top checkpoint to best_val_epoch.pt in the same folder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended copy actions without writing files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite existing best_val_epoch.pt files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    copied = 0
    skipped = 0
    missing = 0

    for ckpt_str in top_10_best_checkpoints:
        src = Path(ckpt_str)
        dst = src.parent / "best_val_epoch.pt"

        if not src.exists():
            print(f"MISSING: {src}")
            missing += 1
            continue

        if dst.exists() and args.skip_existing:
            print(f"SKIP (exists): {dst}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"DRY-RUN copy: {src} -> {dst}")
            copied += 1
            continue

        shutil.copy2(src, dst)
        print(f"COPIED: {src} -> {dst}")
        copied += 1

    print(
        f"Done. copied={copied}, skipped={skipped}, missing={missing}, total={len(top_10_best_checkpoints)}"
    )
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
