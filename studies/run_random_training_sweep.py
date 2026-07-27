#!/usr/bin/env python3
"""Run randomized training sweeps and prune checkpoint artifacts by performance.

Features implemented for this project request:
- Launch up to N train_cli.py runs with random sampled hyperparameters.
- Create unique, identifiable output folders per run (includes run_id and weights).
- Pass --tb_image_epochs 1 5 8 9 10 so TensorBoard images are sparse while
  scalar metrics still log every epoch.
- After 5 completed runs, rank all known train_cli runs using
    score = 2*val_mae + val_mse + val_lpips, with each run scored by its best epoch.
- Pruning policy under storage budget:
    - top-5 runs: keep everything
    - non-top5: keep only final_model.pt and best-epoch checkpoint_epoch_XXXX.pt
    - if still over budget: delete worst non-top5 run directories first
        (top-5 are never deleted by automated pruning).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from synthoseis_pre_train.models import _resolve_encoder_stage_blocks


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = Path("/Volumes/Crucial X9/pretrain_v2_checkpoints")
METRICS_CSV = CHECKPOINTS_ROOT / "epoch_component_metrics.csv"
DEFAULT_DATA_FOLDER = Path("/Users/donaldpg/synthoseis/fake_data")


def _weighted_choice(rng: random.Random, choices: Sequence, probs: Sequence[float]):
    if len(choices) != len(probs):
        raise ValueError("choices and probs length mismatch")
    total = float(sum(probs))
    if total <= 0:
        raise ValueError("probabilities must sum to > 0")
    normalized = [float(p) / total for p in probs]
    return rng.choices(list(choices), weights=normalized, k=1)[0]


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    unet_levels: int
    encoder_depth_profile: str
    hidden_dims: tuple[int, ...]
    mse_weight: float
    pmse_weight: float
    mae_weight: float
    lpips_weight: float
    gdl_weight: float
    tv_weight: float
    output_dir: Path


def _encoder_depth_profile_value(encoder_depth_profile: str, encoder_stage_blocks, unet_levels: int) -> str:
    encoder_stage_blocks = _resolve_encoder_stage_blocks(
        int(unet_levels),
        tuple(encoder_stage_blocks) if encoder_stage_blocks is not None else None,
        str(encoder_depth_profile),
    )
    return f"{str(encoder_depth_profile)} - {' '.join(str(int(v)) for v in encoder_stage_blocks)}"


def _run_config_fingerprint(cfg: RunConfig, args) -> tuple[str, ...]:
    # Intentionally exclude LR/LR-min from dedup so schedule/backoff policy
    # changes do not create duplicate hyperparameter configurations.
    hidden_dims = " ".join(str(v) for v in cfg.hidden_dims)
    kernel_schedule = " ".join(["3"] * len(cfg.hidden_dims))
    encoder_depth_profile = _encoder_depth_profile_value(
        cfg.encoder_depth_profile,
        getattr(args, "encoder_stage_blocks", None),
        cfg.unet_levels,
    )
    return (
        str(int(cfg.unet_levels)),
        hidden_dims,
        kernel_schedule,
        encoder_depth_profile,
        f"{cfg.mse_weight:.6f}",
        f"{cfg.pmse_weight:.6f}",
        f"{cfg.mae_weight:.6f}",
        f"{cfg.lpips_weight:.6f}",
        f"{cfg.tv_weight:.6f}",
        f"{cfg.gdl_weight:.6f}",
    )


def _run_config_fingerprint_from_row(row: dict[str, str]) -> tuple[str, ...] | None:
    try:
        # Intentionally exclude CSV "lr" column from dedup fingerprint.
        return (
            str(int(float(row["unet_levels"]))),
            str(row["hidden_dims (as a space-delimited string)"]).strip(),
            str(row["kernel_schedule (as a space-delimited string)"]).strip(),
            str(row.get("encoder_depth_profile (profile - stage blocks)", "")).strip(),
            f"{float(row['mse_weight']):.6f}",
            f"{float(row['pmse_weight']):.6f}",
            f"{float(row['mae_weight']):.6f}",
            f"{float(row['lpips_weight']):.6f}",
            f"{float(row['tv_weight']):.6f}",
            f"{float(row['gdl_weight']):.6f}",
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_used_run_config_fingerprints(csv_path: Path) -> set[tuple[str, ...]]:
    if not csv_path.exists():
        return set()

    fingerprints: set[tuple[str, ...]] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fingerprint = _run_config_fingerprint_from_row(row)
            if fingerprint is not None:
                fingerprints.add(fingerprint)
    return fingerprints


def sample_run_config(rng: random.Random, run_index: int, output_base: Path) -> RunConfig:
    unet_levels = int(_weighted_choice(rng, [3, 4], [0.5, 0.5]))

    profile_probs = {
        "baseline": 0.20,
        "deeper": 0.50,
        "deepest": 0.30,
    }
    valid_profiles = ["baseline", "deeper", "deepest"] if unet_levels == 4 else ["baseline", "deeper"]
    encoder_depth_profile = str(
        _weighted_choice(rng, valid_profiles, [profile_probs[name] for name in valid_profiles])
    )

    if unet_levels == 3:
        hidden_dims = tuple(
            _weighted_choice(
                rng,
                [(32, 64, 128), (40, 80, 160), (48, 96, 192)],
                [0.333, 0.333, 0.333],
            )
        )
    else:
        hidden_dims = tuple(
            _weighted_choice(
                rng,
                [
                    (32, 64, 128, 256),
                    (40, 74, 138, 256),
                    (48, 76, 121, 192),
                    (48, 84, 146, 256),
                ],
                [0.4, 0.3, 0.2, 0.1],
            )
        )

    mse_weight = float(_weighted_choice(rng, [0.00], [1.0]))
    pmse_weight = float(_weighted_choice(rng, [0.00], [1.0]))
    lpips_weight = float(_weighted_choice(rng, [0.00, 0.01, 0.05, 0.10], [0.5, 0.3, 0.1, 0.1]))
    gdl_weight = float(_weighted_choice(rng, [0.00], [1.0]))
    tv_weight = float(_weighted_choice(rng, [0.00, 0.001, 0.010], [0.10, 0.20, 0.70]))

    mae_weight = 1.0 - (lpips_weight + gdl_weight + tv_weight)
    if mae_weight < 0:
        raise RuntimeError(
            "Sampled weights produced negative mae_weight; check sampling distributions"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = f"sweep_{ts}_r{run_index:03d}_u{unet_levels}_h{'-'.join(str(v) for v in hidden_dims)}_lp{lpips_weight:.3f}_tv{tv_weight:.3f}"
    safe_uid = uid.replace(".", "p")
    output_dir = output_base / safe_uid

    return RunConfig(
        run_id=safe_uid,
        unet_levels=unet_levels,
        encoder_depth_profile=encoder_depth_profile,
        hidden_dims=hidden_dims,
        mse_weight=mse_weight,
        pmse_weight=pmse_weight,
        mae_weight=mae_weight,
        lpips_weight=lpips_weight,
        gdl_weight=gdl_weight,
        tv_weight=tv_weight,
        output_dir=output_dir,
    )


def sample_unique_run_config(
    rng: random.Random,
    run_index: int,
    output_base: Path,
    args,
    used_fingerprints: set[tuple[str, ...]],
    max_attempts: int = 1000,
) -> RunConfig:
    for attempt in range(1, max_attempts + 1):
        cfg = sample_run_config(rng, run_index, output_base)
        fingerprint = _run_config_fingerprint(cfg, args)
        if fingerprint not in used_fingerprints:
            used_fingerprints.add(fingerprint)
            return cfg
        print(
            f"Skipping duplicate sampled config already present in {METRICS_CSV.name}; resampling (attempt {attempt})"
        )
    raise RuntimeError(
        f"Unable to sample a unique run configuration after {max_attempts} attempts; "
        "the search space may be exhausted or the CSV may contain all current combinations."
    )


def _build_cli_command(cfg: RunConfig, args) -> list[str]:
    lr = float(args.lr)
    lr_min = float(args.lr_min)
    if float(cfg.lpips_weight) > 0.0:
        lr = 1.0e-5
        lr_min = 5.0e-6

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "train_cli.py",
        "--epochs",
        str(args.epochs),
        "--val_split_ratio",
        str(args.val_split_ratio),
        "--train_batches_per_epoch",
        str(args.train_batches_per_epoch),
        "--val_batches_per_epoch",
        str(args.val_batches_per_epoch),
        "--loss",
        "multi_component",
        "--mc_mse_weight",
        f"{cfg.mse_weight:.6f}",
        "--mc_pmse_weight",
        f"{cfg.pmse_weight:.6f}",
        "--mc_mae_weight",
        f"{cfg.mae_weight:.6f}",
        "--mc_lpips_weight",
        f"{cfg.lpips_weight:.6f}",
        "--mc_gdl_weight",
        f"{cfg.gdl_weight:.6f}",
        "--mc_tv_weight",
        f"{cfg.tv_weight:.6f}",
        "--mc_lpips_net",
        str(args.mc_lpips_net),
        "--input_extrema_prob",
        str(args.input_extrema_prob),
        "--input_sparse_keep_prob",
        str(args.input_sparse_keep_prob),
        "--input_decimate_trilinear_prob",
        str(args.input_decimate_trilinear_prob),
        "--encoder_depth_profile",
        str(cfg.encoder_depth_profile),
        "--unet_levels",
        str(cfg.unet_levels),
        "--hidden_dims",
        *[str(v) for v in cfg.hidden_dims],
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--grad_clip_norm",
        str(args.grad_clip_norm),
        "--ema_decay",
        str(args.ema_decay),
        "--ema_update_every",
        str(args.ema_update_every),
        "--lr_schedule",
        str(args.lr_schedule),
        "--lr_warmup_epochs",
        str(args.lr_warmup_epochs),
        "--lr_warmup_start_factor",
        str(args.lr_warmup_start_factor),
        "--lr_poly_power",
        str(args.lr_poly_power),
        "--lr_min",
        str(lr_min),
        "--lr",
        str(lr),
        "--output_dir",
        str(cfg.output_dir),
        "--data_folder",
        str(args.data_folder),
        "--tb_image_epochs",
        "1",
        "5",
        "8",
        "9",
        "10",
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    return cmd


def _folder_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


@dataclass
class RankedRun:
    tb_folder: Path
    score: float


@dataclass
class RankedTrainRun:
    run_dir: Path
    tb_folder: Path
    score: float
    best_epoch: int | None


def _parse_metrics_best_scores(csv_path: Path) -> dict[str, float]:
    """Return best (minimum) combined val score per tensorboard folder path."""
    best: dict[str, float] = {}
    if not csv_path.exists():
        return best

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tb = (row.get("tensorboard folder") or "").strip()
            if not tb:
                continue
            try:
                val_mae = float(row["validation mae/L1"])
                val_mse = float(row["validation mse"])
                val_lpips = float(row["validation LPIPS"])
            except (KeyError, TypeError, ValueError):
                continue
            score = 2.0 * val_mae + val_mse + val_lpips
            prev = best.get(tb)
            if prev is None or score < prev:
                best[tb] = score
    return best


def _parse_metrics_best_scores_and_epochs(csv_path: Path) -> dict[str, tuple[float, int | None]]:
    """Return best (minimum) heuristic score and epoch per tensorboard folder path."""
    best: dict[str, tuple[float, int | None]] = {}
    if not csv_path.exists():
        return best

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tb = (row.get("tensorboard folder") or "").strip()
            if not tb:
                continue
            try:
                val_mae = float(row["validation mae/L1"])
                val_mse = float(row["validation mse"])
                val_lpips = float(row["validation LPIPS"])
                epoch = int(float(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            score = 2.0 * val_mae + val_mse + val_lpips
            prev = best.get(tb)
            if prev is None or score < prev[0]:
                best[tb] = (score, epoch)
    return best


def _discover_tb_folders(checkpoints_root: Path) -> list[Path]:
    if not checkpoints_root.exists():
        return []
    return sorted(
        [p for p in checkpoints_root.rglob("runs") if p.is_dir()],
        key=lambda p: str(p),
    )


def _tb_path_candidates(runs_dir: Path, checkpoints_root: Path) -> list[str]:
    candidates = [str(runs_dir)]
    try:
        rel = runs_dir.relative_to(ROOT)
        candidates.append(str(rel))
    except ValueError:
        pass
    try:
        rel_ckpt = runs_dir.relative_to(checkpoints_root)
        candidates.append(str(Path("checkpoints") / rel_ckpt))
    except ValueError:
        pass
    return candidates


def rank_tensorboard_folders(csv_path: Path, checkpoints_root: Path) -> list[RankedRun]:
    best_scores = _parse_metrics_best_scores(csv_path)
    ranked: list[RankedRun] = []

    for runs_dir in _discover_tb_folders(checkpoints_root):
        score = None
        for key in _tb_path_candidates(runs_dir, checkpoints_root):
            score = best_scores.get(key)
            if score is not None:
                break
        if score is None:
            # No metrics found for this folder yet; rank as worst.
            score = float("inf")
        ranked.append(RankedRun(tb_folder=runs_dir, score=score))

    ranked.sort(key=lambda r: (r.score, str(r.tb_folder)))
    return ranked


def rank_train_runs(csv_path: Path, checkpoints_root: Path) -> list[RankedTrainRun]:
    best_by_tb = _parse_metrics_best_scores_and_epochs(csv_path)
    ranked: list[RankedTrainRun] = []

    for runs_dir in _discover_tb_folders(checkpoints_root):
        score_epoch = None
        for key in _tb_path_candidates(runs_dir, checkpoints_root):
            score_epoch = best_by_tb.get(key)
            if score_epoch is not None:
                break
        if score_epoch is None:
            score, best_epoch = float("inf"), None
        else:
            score, best_epoch = score_epoch
        ranked.append(
            RankedTrainRun(
                run_dir=runs_dir.parent,
                tb_folder=runs_dir,
                score=score,
                best_epoch=best_epoch,
            )
        )

    ranked.sort(key=lambda r: (r.score, str(r.run_dir)))
    return ranked


def _prune_run_dir_to_keep_best_only(run: RankedTrainRun) -> tuple[int, int]:
    """Prune one run dir to keep only final_model.pt and best-epoch checkpoint.

    Returns:
        (files_deleted, dirs_deleted)
    """
    keep_files = {"final_model.pt"}
    if run.best_epoch is not None:
        keep_files.add(f"checkpoint_epoch_{run.best_epoch:04d}.pt")

    files_deleted = 0
    dirs_deleted = 0

    if not run.run_dir.exists() or not run.run_dir.is_dir():
        return (0, 0)

    for child in sorted(run.run_dir.iterdir(), key=lambda p: p.name):
        if child.is_file():
            if child.name not in keep_files:
                try:
                    child.unlink()
                    files_deleted += 1
                except OSError:
                    pass
        else:
            shutil.rmtree(child, ignore_errors=True)
            dirs_deleted += 1

    return (files_deleted, dirs_deleted)


def enforce_checkpoint_storage_budget(
    csv_path: Path,
    checkpoints_root: Path,
    max_bytes: int,
    top_k_keep_all: int = 5,
) -> dict[str, int | float]:
    """Apply checkpoint pruning policy under a storage budget.

    Policy:
    - top-k ranked runs (best heuristic): keep all files
    - non-topk runs: keep only final_model.pt + best-epoch checkpoint
    - if still above budget: delete worst non-topk run dirs first
    """
    ranked = rank_train_runs(csv_path, checkpoints_root)
    if not ranked:
        return {
            "ranked_runs": 0,
            "topk_kept_all": 0,
            "runs_pruned_to_best": 0,
            "runs_deleted_entirely": 0,
            "files_deleted": 0,
            "dirs_deleted": 0,
            "bytes_after": 0,
        }

    top_k = max(0, min(int(top_k_keep_all), len(ranked)))
    top_runs = ranked[:top_k]
    non_top_runs = ranked[top_k:]

    files_deleted = 0
    dirs_deleted = 0
    runs_pruned_to_best = 0
    runs_deleted_entirely = 0

    for run in non_top_runs:
        f_del, d_del = _prune_run_dir_to_keep_best_only(run)
        files_deleted += f_del
        dirs_deleted += d_del
        runs_pruned_to_best += 1

    current = _folder_size_bytes(checkpoints_root)
    if current > max_bytes:
        # Escalation: delete worst non-top runs entirely until budget is met.
        for run in reversed(non_top_runs):
            if current <= max_bytes:
                break
            run_sz = _folder_size_bytes(run.run_dir)
            if run.run_dir.exists() and run.run_dir.is_dir():
                shutil.rmtree(run.run_dir, ignore_errors=True)
                dirs_deleted += 1
                runs_deleted_entirely += 1
                current = max(0, current - run_sz)

    bytes_after = _folder_size_bytes(checkpoints_root)
    return {
        "ranked_runs": len(ranked),
        "topk_kept_all": len(top_runs),
        "runs_pruned_to_best": runs_pruned_to_best,
        "runs_deleted_entirely": runs_deleted_entirely,
        "files_deleted": files_deleted,
        "dirs_deleted": dirs_deleted,
        "bytes_after": bytes_after,
    }


def enforce_tb_budget(
    csv_path: Path,
    checkpoints_root: Path,
    max_tb_bytes: int,
) -> tuple[int, int, int]:
    """Delete worst-ranked TensorBoard folders until size <= budget.

    Returns:
        (kept_count, deleted_count, bytes_after)
    """
    ranked = rank_tensorboard_folders(csv_path, checkpoints_root)
    if not ranked:
        return (0, 0, 0)

    sizes = {r.tb_folder: _folder_size_bytes(r.tb_folder) for r in ranked}
    current = sum(sizes.values())
    if current <= max_tb_bytes:
        return (len(ranked), 0, current)

    deleted = 0
    # Delete worst first -> reverse sorted list.
    for rr in reversed(ranked):
        if current <= max_tb_bytes:
            break
        tb_folder = rr.tb_folder
        parent = tb_folder.parent
        sz = sizes.get(tb_folder, 0)
        if tb_folder.exists():
            shutil.rmtree(tb_folder, ignore_errors=True)
        # Remove empty run directory after deleting runs.
        if parent.exists() and parent.is_dir():
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass
        current = max(0, current - sz)
        deleted += 1

    kept_count = max(0, len(ranked) - deleted)
    return (kept_count, deleted, current)


def run_one_training(cfg: RunConfig, args, env: dict[str, str]) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cli_command(cfg, args)

    print("\n" + "=" * 120)
    print(f"Run {cfg.run_id}")
    print(f"  output_dir: {cfg.output_dir}")
    print(
        "  sampled: "
        f"unet_levels={cfg.unet_levels}, encoder_depth_profile={cfg.encoder_depth_profile}, hidden_dims={list(cfg.hidden_dims)}, "
        f"weights(mse={cfg.mse_weight:.3f}, pmse={cfg.pmse_weight:.3f}, mae={cfg.mae_weight:.3f}, "
        f"lpips={cfg.lpips_weight:.3f}, gdl={cfg.gdl_weight:.3f}, tv={cfg.tv_weight:.3f})"
    )
    print("  command:")
    print("    " + " ".join(cmd))

    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - start
    print(f"  exit_code={proc.returncode} elapsed_sec={elapsed:.1f}")
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run randomized train_cli.py sweep with TensorBoard pruning")
    p.add_argument("--num_runs", type=int, default=100, help="Maximum number of training runs to launch")
    p.add_argument("--seed", type=int, default=20260618, help="Random seed")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val_split_ratio", type=float, default=0.3)
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional explicit batch-size override. If omitted, train_cli.py uses the model-size default.",
    )
    p.add_argument("--train_batches_per_epoch", type=int, default=100)
    p.add_argument("--val_batches_per_epoch", type=int, default=40)

    p.add_argument("--mc_lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--input_extrema_prob", type=float, default=0.6)
    p.add_argument("--input_sparse_keep_prob", type=float, default=0.3)
    p.add_argument("--input_decimate_trilinear_prob", type=float, default=0.1)
    p.add_argument("--encoder_depth_profile", type=str, default="deeper")

    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--ema_update_every", type=int, default=1)

    p.add_argument("--lr_schedule", type=str, default="poly")
    p.add_argument("--lr_warmup_epochs", type=int, default=3)
    p.add_argument("--lr_warmup_start_factor", type=float, default=0.025)
    p.add_argument("--lr_poly_power", type=float, default=0.9)
    p.add_argument("--lr_min", type=float, default=2.5e-5)
    p.add_argument("--lr", type=float, default=5.0e-5)

    p.add_argument("--data_folder", type=Path, default=DEFAULT_DATA_FOLDER)
    p.add_argument(
        "--output_base",
        type=Path,
        default=CHECKPOINTS_ROOT / "sweeps_randomized",
        help="Parent folder under checkpoints for sweep run output directories",
    )

    p.add_argument(
        "--tb_budget_gb",
        type=float,
        default=80.0,
        help=(
            "Storage budget in GiB for checkpoints pruning policy. "
            "Top-5 runs keep all; others keep final_model.pt + best checkpoint."
        ),
    )
    p.add_argument(
        "--prune_start_after",
        type=int,
        default=5,
        help="Start pruning after this many completed runs",
    )
    p.add_argument(
        "--stop_on_failure",
        action="store_true",
        help="Stop sweep immediately if a run exits non-zero",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    args.output_base.mkdir(parents=True, exist_ok=True)
    max_tb_bytes = int(args.tb_budget_gb * (1024 ** 3))
    used_fingerprints = _load_used_run_config_fingerprints(METRICS_CSV)

    env = dict(os.environ)

    completed_runs = 0
    for run_idx in range(1, int(args.num_runs) + 1):
        cfg = sample_unique_run_config(rng, run_idx, args.output_base, args, used_fingerprints)
        rc = run_one_training(cfg, args, env)

        if rc != 0:
            print(f"Run failed: {cfg.run_id} (exit_code={rc})")
            if args.stop_on_failure:
                return rc
            continue

        completed_runs += 1

        if completed_runs >= int(args.prune_start_after):
            summary = enforce_checkpoint_storage_budget(
                csv_path=METRICS_CSV,
                checkpoints_root=CHECKPOINTS_ROOT,
                max_bytes=max_tb_bytes,
                top_k_keep_all=5,
            )
            print(
                "Checkpoint prune summary: "
                f"ranked_runs={summary['ranked_runs']}, "
                f"top5_kept_all={summary['topk_kept_all']}, "
                f"runs_pruned_to_best={summary['runs_pruned_to_best']}, "
                f"runs_deleted_entirely={summary['runs_deleted_entirely']}, "
                f"files_deleted={summary['files_deleted']}, "
                f"dirs_deleted={summary['dirs_deleted']}, "
                f"size_after_gb={float(summary['bytes_after']) / (1024 ** 3):.2f}"
            )

    print(f"Sweep complete. completed_runs={completed_runs}, requested={args.num_runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
