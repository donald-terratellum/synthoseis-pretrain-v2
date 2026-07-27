#!/usr/bin/env python3
"""Recreate existing SynthoSeis datasets with --zarr-out segmentation.

Default mode is dry-run. Use --apply to execute regeneration.

Workflow per dataset folder:
1) Read model_config_2026*.json from existing folder.
2) Replay SynthoSeis with same run tag but force --zarr-out segmentation.
3) Verify new folder against old folder:
   - model_data.zarr dataset key list must match
   - model_parameters_*.txt checks: number_faults, number_onlap_episodes,
     number_layers, closure count (when available)
4) If verified, delete old folder (apply mode only).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import zarr


DEFAULT_DATA_ROOT = Path("/Volumes/Crucial X9/fake_data")
DEFAULT_SYNTHOSEIS_DIR = Path("/Users/donaldpg/synthoseis/synthoseis")
DEFAULT_STAGING_ROOT = Path("/Users/donaldpg/synthoseis/fake_data_staging")
DATASET_GLOB = "seismic__2026.*__synthoseis_run_*"
RUN_TAG_RE = re.compile(r"synthoseis_run_(\d{4})$")


@dataclass
class DatasetItem:
    path: Path
    run_tag: str
    run_idx: int
    config_path: Path
    params_txt_path: Path


@dataclass
class VerifyResult:
    ok: bool
    reasons: list[str]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Recreate existing SynthoSeis datasets with --zarr-out segmentation, "
            "verify, then replace old folders."
        )
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--synthoseis-dir", type=Path, default=DEFAULT_SYNTHOSEIS_DIR)
    p.add_argument(
        "--staging-root",
        type=Path,
        default=DEFAULT_STAGING_ROOT,
        help=(
            "Local APFS staging folder used for synthoseis writes before verification "
            "and move into --data-root."
        ),
    )
    p.add_argument(
        "--dataset-glob",
        default=DATASET_GLOB,
        help=f"Dataset folder glob under --data-root (default: {DATASET_GLOB})",
    )
    p.add_argument(
        "--zarr-out",
        default="segmentation",
        choices=["segmentation"],
        help="Forced zarr output mode (CLI-supported value: segmentation)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Execute regeneration and deletions. Default is dry-run.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of datasets to process.",
    )
    p.add_argument(
        "--run-tags",
        nargs="*",
        default=None,
        help="Optional specific run tags (e.g. synthoseis_run_1040 synthoseis_run_1041).",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining datasets when one fails.",
    )
    return p.parse_args()


def _find_single(path_iter: Iterable[Path], description: str, folder: Path) -> Path:
    items = sorted(path_iter)
    if len(items) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} in {folder}, found {len(items)}"
        )
    return items[0]


def _discover_datasets(root: Path, dataset_glob: str, run_tags: set[str] | None) -> list[DatasetItem]:
    dataset_dirs = sorted([p for p in root.glob(dataset_glob) if p.is_dir()])
    items: list[DatasetItem] = []

    for ds_dir in dataset_dirs:
        m = RUN_TAG_RE.search(ds_dir.name)
        if m is None:
            continue
        run_idx = int(m.group(1))
        run_tag = f"synthoseis_run_{run_idx:04d}"
        if run_tags is not None and run_tag not in run_tags:
            continue

        # Only process complete dataset folders.
        if not (ds_dir / "model_data.zarr").is_dir():
            continue
        config_candidates = sorted(ds_dir.glob("model_config_2026*.json"))
        params_candidates = sorted(ds_dir.glob("model_parameters_2026*.txt"))
        if len(config_candidates) != 1 or len(params_candidates) != 1:
            continue

        config_path = _find_single(
            config_candidates,
            "model_config_2026*.json",
            ds_dir,
        )
        params_txt_path = _find_single(
            params_candidates,
            "model_parameters_2026*.txt",
            ds_dir,
        )

        items.append(
            DatasetItem(
                path=ds_dir,
                run_tag=run_tag,
                run_idx=run_idx,
                config_path=config_path,
                params_txt_path=params_txt_path,
            )
        )

    return items


def _dataset_keys(zarr_store_dir: Path) -> list[str]:
    if not zarr_store_dir.exists():
        raise RuntimeError(f"Missing zarr store: {zarr_store_dir}")

    group = zarr.open_group(str(zarr_store_dir), mode="r")
    keys: list[str] = []

    # zarr>=3 exposes Group.members(max_depth=...) instead of visit/visititems.
    if hasattr(group, "members"):
        for name, obj in group.members(max_depth=None):
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                keys.append(name)
    else:
        def _visit(name: str, obj) -> None:
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                keys.append(name)

        group.visititems(_visit)

    return sorted(keys)


def _parse_model_parameters_text(txt_path: Path) -> dict[str, int]:
    text = txt_path.read_text(encoding="utf-8", errors="replace")

    patterns = {
        "number_faults": r"^number_faults:\s*(\d+)\s*$",
        "number_onlap_episodes": r"^number_onlap_episodes:\s*(\d+)\s*$",
        "number_layers": r"^number_layers:\s*(\d+)\s*$",
        # Interpreted as closure count when present in this summary line.
        "channel_closure_bodies": r"^Channel-levee closure statistics\s+[\u2014-]\s+(\d+)\s+body/bodies",
        # Fallback closure proxy if count line is absent.
        "total_hc_voxels": r"^Closures3D\s+[\u2014-].*total\s+HC\s+voxels:\s*(\d+)\s*$",
    }

    out: dict[str, int] = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.MULTILINE)
        if m is not None:
            out[key] = int(m.group(1))
    return out


def _write_overridden_config(src_config: Path, dst_config: Path, data_root: Path) -> None:
    payload = json.loads(src_config.read_text(encoding="utf-8"))
    payload["project_folder"] = str(data_root)
    payload["work_folder"] = str(data_root)
    dst_config.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cleanup_partial_staging_outputs(staging_root: Path, run_tag: str) -> None:
    for candidate in sorted(staging_root.glob(f"*{run_tag}*")):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


def _promote_staged_folder_to_data_root(staged_dir: Path, data_root: Path) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    target = data_root / staged_dir.name
    if target.exists():
        raise RuntimeError(f"Cannot promote staged folder; target already exists: {target}")
    shutil.move(str(staged_dir), str(target))
    return target


def _delete_tree_tolerating_missing(path: Path) -> None:
    """Delete a directory tree while ignoring transient missing-file races.

    On macOS/external volumes, sidecar files (for example `._*`) may disappear
    between directory scan and unlink during rmtree. Treat those as benign.
    """

    if not path.exists():
        return

    def _onerror(_func, _path, exc_info):
        err = exc_info[1]
        if isinstance(err, FileNotFoundError):
            return
        raise err

    shutil.rmtree(path, onerror=_onerror)

    # Best-effort fallback if any residue remains.
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

    if path.exists():
        raise RuntimeError(f"Failed to fully remove directory: {path}")


def _find_new_folder_for_run(data_root: Path, run_tag: str, before: set[Path]) -> Path:
    after = {
        p
        for p in data_root.glob(f"seismic__*__{run_tag}")
        if p.is_dir()
    }
    new_dirs = sorted(after - before)
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)

    raise RuntimeError(
        f"Could not find a newly created dataset folder for run tag {run_tag}."
    )


def _verify(old_item: DatasetItem, new_dir: Path, *, zarr_out: str) -> VerifyResult:
    reasons: list[str] = []

    old_keys = _dataset_keys(old_item.path / "model_data.zarr")
    new_keys = _dataset_keys(new_dir / "model_data.zarr")
    if old_keys != new_keys:
        old_set = set(old_keys)
        new_set = set(new_keys)
        if zarr_out == "segmentation" and old_set.issubset(new_set):
            # Expected for essential -> segmentation reruns: keep old key coverage,
            # add segmentation labels.
            pass
        else:
            reasons.append(
                "model_data.zarr dataset key list mismatch\n"
                f"  old: {old_keys}\n"
                f"  new: {new_keys}"
            )

    new_params_txt = _find_single(
        new_dir.glob("model_parameters_2026*.txt"),
        "model_parameters_2026*.txt",
        new_dir,
    )

    old_metrics = _parse_model_parameters_text(old_item.params_txt_path)
    new_metrics = _parse_model_parameters_text(new_params_txt)

    required = ["number_faults", "number_onlap_episodes", "number_layers"]
    for key in required:
        if key not in old_metrics or key not in new_metrics:
            reasons.append(
                f"Missing metric '{key}' in old/new model_parameters text "
                f"(old_has={key in old_metrics}, new_has={key in new_metrics})"
            )
            continue
        if old_metrics[key] != new_metrics[key]:
            reasons.append(
                f"Mismatch for {key}: old={old_metrics[key]} new={new_metrics[key]}"
            )

    # Closure-count verification (preferred + fallback proxy).
    if "channel_closure_bodies" in old_metrics and "channel_closure_bodies" in new_metrics:
        if old_metrics["channel_closure_bodies"] != new_metrics["channel_closure_bodies"]:
            reasons.append(
                "Mismatch for channel_closure_bodies: "
                f"old={old_metrics['channel_closure_bodies']} "
                f"new={new_metrics['channel_closure_bodies']}"
            )
    elif "total_hc_voxels" in old_metrics and "total_hc_voxels" in new_metrics:
        if old_metrics["total_hc_voxels"] != new_metrics["total_hc_voxels"]:
            reasons.append(
                "Mismatch for total_hc_voxels (closure proxy): "
                f"old={old_metrics['total_hc_voxels']} new={new_metrics['total_hc_voxels']}"
            )
    else:
        reasons.append(
            "Could not verify closure count/proxy (neither channel_closure_bodies nor total_hc_voxels present in both)."
        )

    return VerifyResult(ok=(len(reasons) == 0), reasons=reasons)


def _run_recreate_for_item(
    item: DatasetItem,
    *,
    data_root: Path,
    staging_root: Path,
    synthoseis_dir: Path,
    zarr_out: str,
    apply: bool,
) -> tuple[bool, str]:
    print(f"\n=== {item.path.name} ===")
    print(f"run_tag: {item.run_tag}")
    print(f"config:  {item.config_path}")

    if not apply:
        print("DRY-RUN: would regenerate with:")
        print(
            "  uv run python -u main.py "
            f"-n 1 -r _{item.run_tag} -c <temp_overridden_config.json> "
            f"--telemetry --zarr-out {zarr_out}"
        )
        print(f"DRY-RUN: would stage output in {staging_root}")
        print("DRY-RUN: would verify zarr dataset keys + model_parameters metrics and then delete old folder on success.")
        return True, "dry-run"

    staging_root.mkdir(parents=True, exist_ok=True)
    _cleanup_partial_staging_outputs(staging_root, item.run_tag)

    before = {
        p for p in staging_root.glob(f"seismic__*__{item.run_tag}") if p.is_dir()
    }

    with tempfile.TemporaryDirectory(prefix=f"recreate_{item.run_tag}_") as tmp:
        tmp_config = Path(tmp) / "replay_config.json"
        _write_overridden_config(item.config_path, tmp_config, staging_root)

        cmd = [
            "uv",
            "run",
            "python",
            "-u",
            "main.py",
            "-n",
            "1",
            "-r",
            f"_{item.run_tag}",
            "-c",
            str(tmp_config),
            "--telemetry",
            "--zarr-out",
            zarr_out,
        ]

        print("Running:")
        print("  " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(synthoseis_dir))
        if proc.returncode != 0:
            print(
                "Hint: If failure contains '[Errno 45] Operation not supported' during zarr writes, "
                "ensure --staging-root is on a local APFS filesystem (not exFAT)."
            )
            return False, f"synthoseis failed with exit code {proc.returncode}"

    staged_dir = _find_new_folder_for_run(staging_root, item.run_tag, before)
    print(f"staged folder: {staged_dir}")

    verify = _verify(item, staged_dir, zarr_out=zarr_out)
    if not verify.ok:
        print("VERIFY FAILED:")
        for reason in verify.reasons:
            print("  - " + reason.replace("\n", "\n    "))
        print("Keeping old folder unchanged; staged folder retained for inspection:")
        print(f"  {staged_dir}")
        return False, "verification failed"

    print("VERIFY OK: zarr keys and model_parameters checks matched")

    new_dir = _promote_staged_folder_to_data_root(staged_dir, data_root)
    print(f"promoted new folder: {new_dir}")

    print(f"Deleting old folder: {item.path}")
    _delete_tree_tolerating_missing(item.path)
    return True, f"replaced with {new_dir.name}"


def main() -> int:
    args = _parse_args()

    data_root = args.data_root
    staging_root = args.staging_root
    synthoseis_dir = args.synthoseis_dir
    run_tags = set(args.run_tags) if args.run_tags else None

    if not data_root.exists() or not data_root.is_dir():
        print(f"ERROR: data root does not exist or is not a directory: {data_root}")
        return 2

    if data_root.resolve() == staging_root.resolve():
        print("ERROR: --staging-root must be different from --data-root")
        return 2

    main_py = synthoseis_dir / "main.py"
    if not main_py.exists():
        print(f"ERROR: synthoseis main.py not found at: {main_py}")
        return 2

    items = _discover_datasets(data_root, args.dataset_glob, run_tags)
    if args.limit is not None:
        items = items[: max(0, int(args.limit))]

    if not items:
        print("No matching dataset folders found.")
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Data root: {data_root}")
    print(f"Staging root: {staging_root}")
    print(f"SynthoSeis dir: {synthoseis_dir}")
    print(f"Datasets matched: {len(items)}")

    ok_count = 0
    fail_count = 0

    for item in items:
        try:
            ok, message = _run_recreate_for_item(
                item,
                data_root=data_root,
                staging_root=staging_root,
                synthoseis_dir=synthoseis_dir,
                zarr_out=args.zarr_out,
                apply=bool(args.apply),
            )
        except Exception as exc:
            ok = False
            message = str(exc)

        if ok:
            ok_count += 1
            print(f"RESULT: OK - {message}")
        else:
            fail_count += 1
            print(f"RESULT: FAIL - {message}")
            if not args.continue_on_error:
                print("Stopping on first failure (use --continue-on-error to keep going).")
                break

    print("\nSummary:")
    print(f"  ok   : {ok_count}")
    print(f"  fail : {fail_count}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
