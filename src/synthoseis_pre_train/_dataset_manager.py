"""Dataset discovery, split management, and loader construction helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

from torch.utils.data import ConcatDataset, DataLoader

from synthoseis_pre_train.dataloader import create_dataloader


def _discover_zarr_paths(data_folder: str, dataset_glob: str) -> list:
    """Return zarr paths matching dataset_glob under data_folder, sorted oldest-first.

    Sorting is by parent dataset folder mtime (consistent with
    generate_datasets.sh ls -1dtr). Datasets whose companion temp_folder__
    sibling exists are treated as in-progress and excluded.
    """
    paths = list(Path(data_folder).glob(dataset_glob))
    complete = []
    for p in paths:
        ds_folder = p.parent
        temp_companion = ds_folder.parent / ds_folder.name.replace("seismic__", "temp_folder__", 1)
        if temp_companion.exists():
            continue
        complete.append(p)
    complete.sort(key=lambda p: p.parent.stat().st_mtime)
    return [str(p) for p in complete]


def _prune_oldest_to_target(
    data_folder: str,
    dataset_glob: str,
    discovered: list,
    keep_total: int,
) -> list:
    """Prune oldest complete datasets on disk so only newest keep_total remain."""
    keep_total = int(keep_total)
    if keep_total < 2:
        print(
            f"Epoch prune: safety guard engaged (target keep_total={keep_total} < 2); "
            "skipping pruning."
        )
        return discovered

    if len(discovered) <= keep_total:
        return discovered

    n_delete = len(discovered) - keep_total
    delete_candidates = discovered[:n_delete]
    removed = []

    print(
        f"Epoch prune: {len(discovered)} complete dataset(s) on disk; "
        f"keeping newest {keep_total}, deleting oldest {n_delete}."
    )
    for p in delete_candidates:
        ds_dir = Path(p).parent
        if not ds_dir.name.startswith("seismic__"):
            print(f"  WARNING: refusing to delete unexpected folder: {ds_dir}")
            continue
        try:
            shutil.rmtree(ds_dir)
            removed.append(ds_dir.name)
        except Exception as exc:
            print(f"  WARNING: failed to delete {ds_dir.name}: {exc}")

    if removed:
        print(f"  Removed oldest dataset(s): {removed}")

    return _discover_zarr_paths(data_folder, dataset_glob)


def _update_split(discovered: list, train_paths: list, val_paths: list,
                  num_train: int, num_val: int) -> tuple:
    """Maintain train/val assignment with permanent side exclusivity."""
    discovered_set = set(discovered)

    active_train = [p for p in train_paths if p in discovered_set]
    active_val = [p for p in val_paths if p in discovered_set]

    known = set(train_paths) | set(val_paths)

    new_paths = [p for p in discovered if p not in known]
    added_train, added_val = [], []
    for p in new_paths:
        val_need = num_val - len(active_val) - len(added_val)
        train_need = num_train - len(active_train) - len(added_train)
        if val_need > 0:
            added_val.append(p)
        elif train_need > 0:
            added_train.append(p)
        else:
            break

    if added_train or added_val:
        print(f"Epoch split: {len(added_train) + len(added_val)} new dataset(s) assigned, "
              f"{len(added_train)} to train, {len(added_val)} to val:")
        for p in added_train:
            print(f"  train: {Path(p).parent.name}")
        for p in added_val:
            print(f"    val: {Path(p).parent.name}")
    else:
        n_t = min(len(active_train), num_train)
        n_v = min(len(active_val), num_val)
        missing = max(0, num_train - len(active_train)) + max(0, num_val - len(active_val))
        if missing:
            print(f"Epoch split: {missing} slot(s) below target "
                  f"({n_t}/{num_train} train, {n_v}/{num_val} val) — waiting for new datasets.")
        else:
            print(f"Epoch split: no changes ({n_t} train, {n_v} val active).")

    return train_paths + added_train, val_paths + added_val


def _active_paths(historical: list, n: int, discovered_set: set) -> list:
    """Return the newest n paths from historical that are currently on disk."""

    def _mtime(p: str) -> float:
        return Path(p).parent.stat().st_mtime if Path(p).parent.exists() else 0.0

    on_disk = [p for p in historical if p in discovered_set]
    on_disk.sort(key=_mtime)
    return on_disk[-n:] if on_disk else []


def _resolve_target_counts(
    total_datasets: int,
    val_split_ratio: float,
) -> tuple[int, int]:
    """Resolve train/val target counts from validation split ratio."""
    if total_datasets <= 0:
        return 0, 0

    if total_datasets == 1:
        return 1, 0

    ratio = max(0.0, min(1.0, float(val_split_ratio)))
    n_val = int(round(total_datasets * ratio))
    n_val = max(1, min(n_val, total_datasets - 1))
    n_train = total_datasets - n_val
    return n_train, n_val


def _build_loaders(
    train_paths: list,
    val_paths: list,
    loader_kwargs: dict,
    train_batches_per_epoch: int | None = None,
    val_batches_per_epoch: int | None = None,
) -> tuple[DataLoader | None, list[tuple[str, DataLoader]]]:
    """Build one merged train DataLoader and per-dataset val DataLoaders."""

    def _mtime(p: str) -> float:
        parent = Path(p).parent
        return parent.stat().st_mtime if parent.exists() else 0.0

    train_per_ds: list[tuple[str, DataLoader]] = []

    def _dataset_len(loader: DataLoader) -> int:
        try:
            return len(cast(Any, loader.dataset))
        except Exception:
            return 0

    print("  Loading train datasets...")
    for path in sorted(train_paths, key=_mtime):
        name = Path(path).parent.name
        try:
            loader = cast(DataLoader, create_dataloader(path, augment=True, **loader_kwargs))
            print(f"    {name}: {_dataset_len(loader)} samples, {len(loader)} batches")
            train_per_ds.append((name, loader))
        except Exception as e:
            print(f"    WARNING: skipping {name} (train) — {e}")

    if train_per_ds:
        merged_dataset = ConcatDataset([ldr.dataset for _, ldr in train_per_ds])
        base = train_per_ds[0][1]
        train_loader: DataLoader | None = DataLoader(
            merged_dataset,
            batch_size=int(base.batch_size) if base.batch_size is not None else 1,
            shuffle=True,
            num_workers=base.num_workers,
            pin_memory=base.pin_memory,
        )
    else:
        train_loader = None

    val_loaders: list[tuple[str, DataLoader]] = []
    if val_paths:
        print("  Loading val datasets...")
        for path in sorted(val_paths, key=_mtime):
            name = Path(path).parent.name
            try:
                loader = cast(DataLoader, create_dataloader(path, augment=False, **loader_kwargs))
                print(f"    {name}: {_dataset_len(loader)} samples, {len(loader)} batches")
                val_loaders.append((name, loader))
            except Exception as e:
                print(f"    WARNING: skipping {name} (val) — {e}")

    def _safe_loader_len(loader: DataLoader | None) -> int:
        if loader is None:
            return 0
        try:
            return len(loader)
        except Exception as e:
            print(f"  WARNING: loader length unavailable; treating as 0 batches — {e}")
            return 0

    natural_train = _safe_loader_len(train_loader)
    natural_val = sum(_safe_loader_len(l) for _, l in val_loaders)
    shown_train = natural_train if train_batches_per_epoch is None else train_batches_per_epoch
    shown_val = natural_val if val_batches_per_epoch is None else val_batches_per_epoch
    if shown_train == natural_train and shown_val == natural_val:
        print(f"  Batches this epoch: {shown_train} train, {shown_val} val")
    else:
        print(
            f"  Batches this epoch: {shown_train} train, {shown_val} val "
            f"(natural loader sizes: {natural_train} train, {natural_val} val)"
        )
    return train_loader, val_loaders
