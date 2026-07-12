#!/usr/bin/env python3
"""Orchestrate unattended iterative retraining for the top-10 model configurations.

Workflow per pass:
1) Generate N new SynthoSeis datasets with incrementing start-index.
2) Train each configured model to a cumulative epoch target.
3) Prune checkpoints in best-val folders.
4) Persist state so interrupted runs can resume safely.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    output_dir: str
    unet_levels: int
    hidden_dims: tuple[int, ...]
    encoder_depth_profile: str
    mc_mae_weight: float
    mc_lpips_weight: float
    mc_tv_weight: float
    mc_gdl_weight: float
    lr: float


TOP_10_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("checkpoint_copilot_1", "checkpoints/checkpoint_copilot_1", 3, (40, 80, 160), "deeper", 0.995, 0.005, 0.0, 0.0, 1e-5),
    ModelSpec("checkpoint_copilot_2", "checkpoints/checkpoint_copilot_2", 3, (40, 88, 176), "deeper", 0.995, 0.005, 0.001, 0.0, 1e-5),
    ModelSpec("checkpoint_copilot_4", "checkpoints/checkpoint_copilot_4", 4, (40, 80, 160, 256), "deeper", 0.995, 0.005, 0.0, 0.0, 1e-5),
    ModelSpec("checkpoint_copilot_5", "checkpoints/checkpoint_copilot_5", 4, (44, 88, 176, 256), "deeper", 0.995, 0.005, 0.001, 0.0, 1e-5),
    ModelSpec(
        "sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010",
        "checkpoints/sweep_20260621_104613_r001_u4_h40-74-138-256_lp0p000_tv0p010",
        4,
        (40, 74, 138, 256),
        "deeper",
        0.995,
        0.005,
        0.0,
        0.0,
        5e-5,
    ),
    ModelSpec(
        "sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010",
        "checkpoints/sweep_20260622_085428_r011_u4_h32-64-128-256_lp0p000_tv0p010",
        4,
        (32, 64, 128, 256),
        "baseline",
        0.990,
        0.000,
        0.000,
        0.010,
        5e-5,
    ),
    ModelSpec(
        "sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001",
        "checkpoints/sweep_20260620_035309_r004_u3_h40-80-160_lp0p000_tv0p001",
        3,
        (40, 80, 160),
        "deeper",
        0.995,
        0.005,
        0.001,
        0.0,
        5e-5,
    ),
    ModelSpec(
        "sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001",
        "checkpoints/sweep_20260620_080306_r006_u3_h32-64-128_lp0p000_tv0p001",
        3,
        (32, 64, 128),
        "deeper",
        0.995,
        0.005,
        0.0,
        0.0,
        5e-5,
    ),
    ModelSpec(
        "sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000",
        "checkpoints/sweep_20260619_170551_r001_u4_h40-74-138-256_lp0p000_tv0p000",
        4,
        (40, 74, 138, 256),
        "deeper",
        0.995,
        0.005,
        0.001,
        0.0,
        5e-5,
    ),
    ModelSpec(
        "sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001",
        "checkpoints/sweep_20260621_232252_r007_u4_h40-74-138-256_lp0p000_tv0p001",
        4,
        (40, 74, 138, 256),
        "deepest",
        0.999,
        0.0,
        0.0,
        0.001,
        5e-5,
    ),
)


def parse_schedule(text: str) -> list[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("schedule must include at least one integer")
    if any(v <= 0 for v in vals):
        raise ValueError("schedule values must be positive")
    return vals


def cumulative_targets(increments: list[int]) -> list[int]:
    total = 0
    out: list[int] = []
    for inc in increments:
        total += int(inc)
        out.append(total)
    return out


def _default_state_path() -> Path:
    return Path("checkpoints") / "top10_retrain_loop_state.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _expand(path_like: str) -> str:
    return str(Path(path_like).expanduser())


def _run_streaming(command: list[str], log_path: Path, dry_run: bool) -> int:
    printable = " ".join(command)
    print(f"$ {printable}")
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n$ {printable}\n")
        f.flush()

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        proc.wait()
        return int(proc.returncode)


def _run_with_retries(
    command: list[str],
    log_path: Path,
    max_retries: int,
    retry_delay_sec: int,
    dry_run: bool,
) -> None:
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        rc = _run_streaming(command, log_path, dry_run=dry_run)
        if rc == 0:
            return
        if attempt >= attempts:
            raise RuntimeError(f"Command failed after {attempts} attempt(s): {' '.join(command)}")
        print(f"WARNING: command failed (attempt {attempt}/{attempts}); retrying in {retry_delay_sec}s")
        if not dry_run:
            time.sleep(max(0, int(retry_delay_sec)))


def _delete_generated_datasets(
    *,
    data_folder: str,
    start_index: int,
    count: int,
    dry_run: bool,
) -> int:
    """Delete previously generated synthetic dataset folders for a run-index range."""
    data_root = Path(data_folder)
    if not data_root.exists():
        print(f"WARNING: data folder does not exist for cleanup: {data_root}")
        return 0

    deleted = 0
    for run_idx in range(int(start_index), int(start_index) + int(count)):
        run_suffix = f"synthoseis_run_{run_idx:04d}"
        pattern = f"seismic__*__{run_suffix}"
        for ds_dir in sorted(data_root.glob(pattern)):
            if not ds_dir.is_dir():
                continue
            if dry_run:
                print(f"DRY-RUN: would delete previous dataset folder: {ds_dir}")
            else:
                print(f"Deleting previous dataset folder: {ds_dir}")
                shutil.rmtree(ds_dir, ignore_errors=False)
            deleted += 1

    if deleted == 0:
        print(
            "No previous generated dataset folders matched for cleanup: "
            f"start_index={start_index}, count={count}"
        )
    return deleted


def _cleanup_partial_generation_outputs(
    *,
    data_folder: str,
    start_index: int,
    count: int,
    dry_run: bool,
) -> int:
    """Remove partial synthoseis outputs for a run-index range.

    This targets both finalized dataset folders and temporary generation folders
    that include `synthoseis_run_XXXX` in their names.
    """
    data_root = Path(data_folder)
    if not data_root.exists():
        return 0

    removed = 0
    for run_idx in range(int(start_index), int(start_index) + int(count)):
        run_suffix = f"synthoseis_run_{run_idx:04d}"
        for candidate in sorted(data_root.glob(f"*{run_suffix}*")):
            if not candidate.is_dir():
                continue
            if dry_run:
                print(f"DRY-RUN: would remove partial generation folder: {candidate}")
            else:
                print(f"Removing partial generation folder: {candidate}")
                shutil.rmtree(candidate, ignore_errors=True)
            removed += 1

    return removed


def _backup_generated_datasets(
    *,
    data_folder: str,
    backup_dir: str,
    start_index: int,
    count: int,
    dry_run: bool,
) -> int:
    """Copy freshly generated dataset folders to the backup root before training.

    The loop backs up finalized SynthoSeis folders directly so the training
    entrypoint does not need to rediscover them later.
    """
    data_root = Path(data_folder)
    backup_root = Path(backup_dir)
    if not data_root.exists():
        print(f"WARNING: data folder does not exist for backup: {data_root}")
        return 0

    data_root_resolved = data_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    backed_up_run_indices: set[int] = set()

    print(
        "Backup copy starting: "
        f"data_root={data_root_resolved} backup_root={backup_root.resolve()} "
        f"start_index={start_index} count={count}"
    )

    for run_idx in range(int(start_index), int(start_index) + int(count)):
        run_suffix = f"synthoseis_run_{run_idx:04d}"
        candidate_dirs = [
            candidate
            for candidate in sorted(data_root.glob(f"seismic__*__{run_suffix}"))
            if candidate.is_dir()
        ]
        if not candidate_dirs:
            print(
                "WARNING: no finalized generated dataset folders found for backup: "
                f"run_index={run_idx}, pattern=seismic__*__{run_suffix}"
            )
            continue

        for src_dataset_dir in candidate_dirs:
            try:
                rel_dataset_dir = src_dataset_dir.resolve().relative_to(data_root_resolved)
            except Exception:
                print(
                    "WARNING: skipping backup for dataset outside --data-folder: "
                    f"{src_dataset_dir}"
                )
                continue

            dst_dataset_dir = backup_root / rel_dataset_dir
            try:
                print(f"Copying dataset: run_index={run_idx} {src_dataset_dir} -> {dst_dataset_dir}")
                if dst_dataset_dir.exists():
                    shutil.rmtree(dst_dataset_dir)
                dst_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_dataset_dir, dst_dataset_dir, copy_function=shutil.copy2)
                shutil.copystat(src_dataset_dir, dst_dataset_dir)
                print(f"Backed up dataset: run_index={run_idx} {src_dataset_dir} -> {dst_dataset_dir}")
                backed_up_run_indices.add(run_idx)
            except Exception as exc:
                print(f"WARNING: failed to back up dataset run_index={run_idx} {src_dataset_dir}: {exc}")

    print(
        "Backup status: "
        f"backed_up_run_indices={len(backed_up_run_indices)}/{count}"
    )
    return len(backed_up_run_indices)


def _build_train_command(
    spec: ModelSpec,
    *,
    epoch_target: int,
    data_folder: str,
    real_train_paths: str,
    real_test_paths: str,
    epoch_samples: int,
    real_epoch_samples: int,
    test_batches_per_epoch: int,
    train_batches_per_epoch: int,
    val_batches_per_epoch: int,
    batch_size: int,
    val_split_ratio: float,
    lr_min: float,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "train_cli.py",
        "--loss",
        "multi_component",
        "--mc_lpips_net",
        "alex",
        "--output_dir",
        spec.output_dir,
        "--data_folder",
        data_folder,
        "--real_train_paths",
        real_train_paths,
        "--real_test_paths",
        real_test_paths,
        "--epoch_samples",
        str(epoch_samples),
        "--real_epoch_samples",
        str(real_epoch_samples),
        "--test_batches_per_epoch",
        str(test_batches_per_epoch),
        "--epochs",
        str(epoch_target),
        "--val_split_ratio",
        str(val_split_ratio),
        "--batch_size",
        str(batch_size),
        "--train_batches_per_epoch",
        str(train_batches_per_epoch),
        "--val_batches_per_epoch",
        str(val_batches_per_epoch),
        "--unet_levels",
        str(spec.unet_levels),
        "--hidden_dims",
        *[str(v) for v in spec.hidden_dims],
        "--encoder_depth_profile",
        spec.encoder_depth_profile,
        "--mc_mse_weight",
        "0",
        "--mc_pmse_weight",
        "0",
        "--mc_mae_weight",
        str(spec.mc_mae_weight),
        "--mc_lpips_weight",
        str(spec.mc_lpips_weight),
        "--mc_tv_weight",
        str(spec.mc_tv_weight),
        "--mc_gdl_weight",
        str(spec.mc_gdl_weight),
        "--lr_min",
        str(lr_min),
        "--lr",
        str(spec.lr),
    ]

    resume_path = Path(spec.output_dir) / "checkpoint_final_model.pt"
    if resume_path.exists():
        cmd.extend(["--resume", str(resume_path)])
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unattended iterative retraining for top-10 checkpoint configurations.")
    parser.add_argument("--schedule", default="2,2,3,3,5,5,5,5", help="Per-pass epoch increments, comma-separated.")
    parser.add_argument("--start-index", type=int, default=1000, help="Initial start index for generate_datasets.sh.")
    parser.add_argument("--datasets-per-pass", type=int, default=10, help="Number of generated datasets per pass.")
    parser.add_argument("--synthoseis-dir", default="~/synthoseis/synthoseis")
    parser.add_argument("--data-folder", default="/Users/donaldpg/synthoseis/fake_data")
    parser.add_argument(
        "--backup-dir",
        default="/Volumes/Crucial X9/fake_data",
        help="Backup root forwarded to train_cli.py --backup_dir (default: /Volumes/Crucial X9/fake_data)",
    )
    parser.add_argument("--real-train-paths", default="/Users/donaldpg/synthoseis/real_data")
    parser.add_argument("--real-test-paths", default="/Users/donaldpg/synthoseis/fake_data/test")
    parser.add_argument("--epoch-samples", type=int, default=200000)
    parser.add_argument("--real-epoch-samples", type=int, default=50000)
    parser.add_argument("--test-batches-per-epoch", type=int, default=60)
    parser.add_argument("--train-batches-per-epoch", type=int, default=100)
    parser.add_argument("--val-batches-per-epoch", type=int, default=40)
    parser.add_argument("--val-split-ratio", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr-min", type=float, default=5e-6)
    parser.add_argument("--generate-script", default="~/synthoseis-pre-train/generate_datasets.sh")
    parser.add_argument("--prune-script", default="studies/prune_pt_in_best_val_folders.py")
    parser.add_argument("--state-path", type=Path, default=_default_state_path())
    parser.add_argument("--log-file", type=Path, default=Path("logs/top10_retrain_loop.log"))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay-sec", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-prune", action="store_true")
    parser.add_argument(
        "--skip-delete-previous-generated",
        action="store_true",
        help="Do not delete the previous pass's generated synthetic datasets before creating new ones.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    increments = parse_schedule(args.schedule)
    targets = cumulative_targets(increments)

    state = _load_state(args.state_path)
    pass_idx = int(state.get("pass_idx", 0))
    model_idx = int(state.get("model_idx", 0))
    datasets_generated = bool(state.get("datasets_generated", False))
    start_index_next = int(state.get("start_index_next", int(args.start_index)))

    if pass_idx >= len(targets):
        print("All passes already complete according to saved state.")
        return 0

    generate_script = _expand(args.generate_script)
    synthoseis_dir = _expand(args.synthoseis_dir)

    while pass_idx < len(targets):
        epoch_target = targets[pass_idx]
        print("\n" + "=" * 80)
        print(
            f"Pass {pass_idx + 1}/{len(targets)} | increment={increments[pass_idx]} | "
            f"target_epochs={epoch_target} | start_index={start_index_next}"
        )
        print("=" * 80)

        if not datasets_generated:
            if not args.skip_delete_previous_generated:
                prev_start = state.get("last_generated_start_index")
                prev_count = state.get("last_generated_count")
                if prev_start is not None and prev_count is not None:
                    _delete_generated_datasets(
                        data_folder=args.data_folder,
                        start_index=int(prev_start),
                        count=int(prev_count),
                        dry_run=args.dry_run,
                    )

            gen_cmd = [
                generate_script,
                "-n",
                str(args.datasets_per_pass),
                "--synthoseis-dir",
                synthoseis_dir,
                "-d",
                args.data_folder,
                "--start-index",
                str(start_index_next),
            ]
            gen_attempts = max(1, int(args.max_retries))
            for attempt in range(1, gen_attempts + 1):
                rc = _run_streaming(gen_cmd, args.log_file, dry_run=args.dry_run)
                if rc == 0:
                    break
                if attempt >= gen_attempts:
                    raise RuntimeError(
                        f"Command failed after {gen_attempts} attempt(s): {' '.join(gen_cmd)}"
                    )
                _cleanup_partial_generation_outputs(
                    data_folder=args.data_folder,
                    start_index=int(start_index_next),
                    count=int(args.datasets_per_pass),
                    dry_run=args.dry_run,
                )
                print(
                    f"WARNING: generation command failed (attempt {attempt}/{gen_attempts}); "
                    f"retrying in {args.retry_delay_sec}s"
                )
                if not args.dry_run:
                    time.sleep(max(0, int(args.retry_delay_sec)))

            if args.backup_dir:
                backed_up_count = _backup_generated_datasets(
                    data_folder=args.data_folder,
                    backup_dir=args.backup_dir,
                    start_index=int(start_index_next),
                    count=int(args.datasets_per_pass),
                    dry_run=args.dry_run,
                )
                if not args.dry_run and backed_up_count != int(args.datasets_per_pass):
                    raise RuntimeError(
                        "Backup copy did not cover all newly generated datasets; "
                        f"expected {int(args.datasets_per_pass)}, copied {backed_up_count}."
                    )
            else:
                print("Backup status: backup_dir is None; skipping dataset backup")

            datasets_generated = True
            state.update(
                {
                    "pass_idx": pass_idx,
                    "model_idx": model_idx,
                    "datasets_generated": datasets_generated,
                    "start_index_next": start_index_next,
                    "last_generated_start_index": int(start_index_next),
                    "last_generated_count": int(args.datasets_per_pass),
                }
            )
            _save_state(args.state_path, state)

        while model_idx < len(TOP_10_MODEL_SPECS):
            spec = TOP_10_MODEL_SPECS[model_idx]
            print(f"\n--- Training model {model_idx + 1}/{len(TOP_10_MODEL_SPECS)}: {spec.name} ---")
            cmd = _build_train_command(
                spec,
                epoch_target=epoch_target,
                data_folder=args.data_folder,
                real_train_paths=args.real_train_paths,
                real_test_paths=args.real_test_paths,
                epoch_samples=args.epoch_samples,
                real_epoch_samples=args.real_epoch_samples,
                test_batches_per_epoch=args.test_batches_per_epoch,
                train_batches_per_epoch=args.train_batches_per_epoch,
                val_batches_per_epoch=args.val_batches_per_epoch,
                batch_size=args.batch_size,
                val_split_ratio=args.val_split_ratio,
                lr_min=args.lr_min,
            )
            _run_with_retries(
                cmd,
                args.log_file,
                max_retries=args.max_retries,
                retry_delay_sec=args.retry_delay_sec,
                dry_run=args.dry_run,
            )

            model_idx += 1
            state.update(
                {
                    "pass_idx": pass_idx,
                    "model_idx": model_idx,
                    "datasets_generated": datasets_generated,
                    "start_index_next": start_index_next,
                    "last_generated_start_index": state.get("last_generated_start_index"),
                    "last_generated_count": state.get("last_generated_count"),
                }
            )
            _save_state(args.state_path, state)

        if not args.skip_prune:
            prune_cmd = ["uv", "run", "python", args.prune_script, "--apply"]
            _run_with_retries(
                prune_cmd,
                args.log_file,
                max_retries=args.max_retries,
                retry_delay_sec=args.retry_delay_sec,
                dry_run=args.dry_run,
            )

        pass_idx += 1
        model_idx = 0
        datasets_generated = False
        start_index_next += int(args.datasets_per_pass)
        state.update(
            {
                "pass_idx": pass_idx,
                "model_idx": model_idx,
                "datasets_generated": datasets_generated,
                "start_index_next": start_index_next,
                "last_generated_start_index": state.get("last_generated_start_index"),
                "last_generated_count": state.get("last_generated_count"),
            }
        )
        _save_state(args.state_path, state)

    print("\nAll passes complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
