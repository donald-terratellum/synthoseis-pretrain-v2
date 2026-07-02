"""Dataset discovery, split management, and loader construction helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from synthoseis_pre_train.dataloader import create_dataloader


class _SourceTaggedDataset(Dataset):
    """Wrap a dataset and append a source tag to each sample tuple."""

    def __init__(self, base_dataset: Any, source_tag: str) -> None:
        self.base_dataset = base_dataset
        self.source_tag = str(source_tag)

    def __len__(self) -> int:
        return len(cast(Any, self.base_dataset))

    def __getitem__(self, idx: int):
        inp, tgt, msk = self.base_dataset[idx]
        return inp, tgt, msk, self.source_tag

    def __getattr__(self, name: str) -> Any:
        # Preserve compatibility for code paths that inspect dataset metadata
        # (for example, data_path used in per-dataset figure logging).
        return getattr(self.base_dataset, name)


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


def _voxel_count_for_zarr(path: str) -> int:
    """Return the voxel count of the first 3-D array in a zarr store.

    Only the shape of one array key is used — multiple keys for the same physical
    volume do not multiply the weight.
    """
    try:
        import zarr
        from synthoseis_pre_train.dataloader import DEFAULT_GEOLOGIC_SCORE_KEYS
        store = zarr.open(str(path), mode="r")
        for key in store.array_keys():
            arr = store[key]
            if len(arr.shape) == 3 and key not in DEFAULT_GEOLOGIC_SCORE_KEYS:
                z, x, y = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
                return z * x * y
    except Exception:
        pass
    return 1


def _build_loaders(
    train_paths: list,
    val_paths: list,
    loader_kwargs: dict,
    train_batches_per_epoch: int | None = None,
    val_batches_per_epoch: int | None = None,
    test_batches_per_epoch: int | None = None,
    real_train_paths: list[str] | None = None,
    real_test_paths: list[str] | None = None,
    real_epoch_samples: int | None = None,
) -> tuple[DataLoader | None, list[tuple[str, DataLoader]], list[tuple[str, DataLoader]]]:
    """Build merged train loader, per-dataset val loaders, and per-dataset test loaders.

    Real datasets from ``real_train_paths`` can be either .npy volumes or zarr
    stores, and are merged into the same ConcatDataset as synthetic zarr datasets.
    Sampling weights are proportional to the volume's voxel count so that larger
    datasets are sampled more often in the same ratio as physical volume sizes.

    Returns a 3-tuple: (train_loader, val_loaders, test_loaders).
    """
    from synthoseis_pre_train._npy_dataset import NpySeismicDataset

    def _mtime(p: str) -> float:
        parent = Path(p).parent
        return parent.stat().st_mtime if parent.exists() else 0.0

    def _dataset_len(loader: DataLoader) -> int:
        try:
            return len(cast(Any, loader.dataset))
        except Exception:
            return 0

    def _collate_mixed_samples(
        batch: list[tuple[Any, Any, Any, str] | tuple[Any, Any, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """Collate mixed numpy/tensor samples from synthetic + real datasets.

        Normalizes each sample component to 3D (D,H,W) before stacking, so
        mixed contracts like (D,H,W) and (1,D,H,W) do not break collation.
        """
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        source_tags: list[str] = []

        for sample in batch:
            if len(sample) == 4:
                inp, tgt, msk, src = sample
            else:
                inp, tgt, msk = sample
                src = "?"
            inp_t = torch.as_tensor(inp)
            tgt_t = torch.as_tensor(tgt)
            msk_t = torch.as_tensor(msk)

            if inp_t.ndim == 4 and int(inp_t.shape[0]) == 1:
                inp_t = inp_t.squeeze(0)
            if tgt_t.ndim == 4 and int(tgt_t.shape[0]) == 1:
                tgt_t = tgt_t.squeeze(0)
            if msk_t.ndim == 4 and int(msk_t.shape[0]) == 1:
                msk_t = msk_t.squeeze(0)

            if inp_t.ndim != 3 or tgt_t.ndim != 3 or msk_t.ndim != 3:
                raise ValueError(
                    "Unexpected sample rank in mixed collate: "
                    f"inp={tuple(inp_t.shape)}, tgt={tuple(tgt_t.shape)}, mask={tuple(msk_t.shape)}"
                )

            inputs.append(inp_t.float())
            targets.append(tgt_t.float())
            masks.append(msk_t.bool())
            source_tags.append(str(src))

        return torch.stack(inputs, dim=0), torch.stack(targets, dim=0), torch.stack(masks, dim=0), source_tags

    # ------------------------------------------------------------------ train
    # Collect (dataset, voxel_count) pairs so we can build weighted sampler.
    train_datasets: list[Any] = []
    train_voxel_counts: list[int] = []

    print("  Loading train datasets (zarr)...")
    for path in sorted(train_paths, key=_mtime):
        name = Path(path).parent.name
        try:
            loader = cast(DataLoader, create_dataloader(path, augment=True, shuffle=False, **loader_kwargs))
            ds = loader.dataset
            n_samples = len(cast(Any, ds))
            voxels = _voxel_count_for_zarr(path)
            print(f"    {name}: {n_samples} samples, voxels={voxels:,}")
            train_datasets.append(_SourceTaggedDataset(ds, "S"))
            train_voxel_counts.append(voxels)
        except Exception as e:
            print(f"    WARNING: skipping {name} (train) — {e}")

    if real_train_paths:
        # Build npy_loader_kwargs: subset of loader_kwargs relevant to NpySeismicDataset.
        npy_kw: dict[str, Any] = dict(
            sample_shape=loader_kwargs.get("sample_shape", (128, 128, 128)),
            augment=True,
            trace_mask_ratio=loader_kwargs.get("trace_mask_ratio", 0.07),
            cluster_prob=loader_kwargs.get("cluster_prob", 0.8),
            input_extrema_prob=loader_kwargs.get("input_extrema_prob", 1.0),
            input_sparse_keep_prob=loader_kwargs.get("input_sparse_keep_prob", 0.0),
            input_decimate_trilinear_prob=loader_kwargs.get("input_decimate_trilinear_prob", 0.0),
            epoch_samples=real_epoch_samples,
        )
        print("  Loading real train datasets (.npy/.zarr)...")
        for real_path in real_train_paths:
            path_obj = Path(real_path)
            name = path_obj.stem if path_obj.suffix in (".npy", ".zarr") else path_obj.name
            try:
                if path_obj.suffix == ".npy":
                    ds = NpySeismicDataset(str(path_obj), **npy_kw)
                    n_samples = len(ds)
                    z, x, y = ds.volume_shape
                    voxels = z * x * y
                    print(f"    {name}: {n_samples} samples, shape={ds.volume_shape}, voxels={voxels:,}")
                    train_datasets.append(_SourceTaggedDataset(ds, "R"))
                    train_voxel_counts.append(voxels)
                elif path_obj.suffix == ".zarr":
                    loader = cast(DataLoader, create_dataloader(str(path_obj), augment=True, shuffle=False, **loader_kwargs))
                    ds = loader.dataset
                    n_samples = len(cast(Any, ds))
                    voxels = _voxel_count_for_zarr(str(path_obj))
                    print(f"    {name}: {n_samples} samples, voxels={voxels:,}")
                    train_datasets.append(_SourceTaggedDataset(ds, "R"))
                    train_voxel_counts.append(voxels)
                else:
                    print(f"    WARNING: skipping {name} (real train) — unsupported path type: {path_obj}")
            except Exception as e:
                print(f"    WARNING: skipping {name} (real train) — {e}")

    train_loader: DataLoader | None = None
    if train_datasets:
        merged_dataset = ConcatDataset(train_datasets)
        base_loader = cast(
            DataLoader,
            create_dataloader(train_paths[0] if train_paths else real_train_paths[0],
                              augment=True, shuffle=False, **loader_kwargs)
            if (train_paths or real_train_paths) else None,
        )
        batch_size = int(loader_kwargs.get("batch_size", 1))
        num_workers = int(loader_kwargs.get("num_workers", 0))
        pin_memory = bool(loader_kwargs.get("pin_memory", False))

        # Build per-sample weights: every sample in a dataset gets weight
        # proportional to that dataset's voxel count.
        weights_list: list[float] = []
        for ds, voxels in zip(train_datasets, train_voxel_counts):
            n = len(cast(Any, ds))
            w = float(voxels) / max(1, n)
            weights_list.extend([w] * n)

        weights_np = np.array(weights_list, dtype=np.float64)
        weights_np /= weights_np.sum()  # normalise to probability

        sampler = WeightedRandomSampler(
            weights=weights_np.tolist(),
            num_samples=len(merged_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            merged_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_mixed_samples,
        )

    # ------------------------------------------------------------------ val
    val_loaders: list[tuple[str, DataLoader]] = []
    if val_paths:
        print("  Loading val datasets...")
        for path in sorted(val_paths, key=_mtime):
            name = Path(path).parent.name
            try:
                loader = cast(DataLoader, create_dataloader(path, augment=False, shuffle=False, **loader_kwargs))
                print(f"    {name}: {_dataset_len(loader)} samples, {len(loader)} batches")
                val_loaders.append((name, loader))
            except Exception as e:
                print(f"    WARNING: skipping {name} (val) — {e}")

    # ------------------------------------------------------------------ test (.npy/.zarr)
    test_loaders: list[tuple[str, DataLoader]] = []
    if real_test_paths:
        npy_test_kw: dict[str, Any] = dict(
            sample_shape=loader_kwargs.get("sample_shape", (128, 128, 128)),
            augment=False,
            trace_mask_ratio=loader_kwargs.get("trace_mask_ratio", 0.07),
            cluster_prob=loader_kwargs.get("cluster_prob", 0.8),
            input_extrema_prob=loader_kwargs.get("input_extrema_prob", 1.0),
            input_sparse_keep_prob=loader_kwargs.get("input_sparse_keep_prob", 0.0),
            input_decimate_trilinear_prob=loader_kwargs.get("input_decimate_trilinear_prob", 0.0),
            epoch_samples=real_epoch_samples,
        )
        batch_size = int(loader_kwargs.get("batch_size", 1))
        num_workers = int(loader_kwargs.get("num_workers", 0))
        pin_memory = bool(loader_kwargs.get("pin_memory", False))
        print("  Loading real test datasets (.npy/.zarr)...")
        for real_path in real_test_paths:
            path_obj = Path(real_path)
            name = path_obj.stem if path_obj.suffix in (".npy", ".zarr") else path_obj.name
            try:
                if path_obj.suffix == ".npy":
                    ds = NpySeismicDataset(str(path_obj), **npy_test_kw)
                    loader = DataLoader(
                        ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )
                    print(f"    {name}: {len(ds)} samples, {len(loader)} batches")
                    test_loaders.append((name, loader))
                elif path_obj.suffix == ".zarr":
                    loader = cast(DataLoader, create_dataloader(str(path_obj), augment=False, shuffle=False, **loader_kwargs))
                    print(f"    {name}: {_dataset_len(loader)} samples, {len(loader)} batches")
                    test_loaders.append((name, loader))
                else:
                    print(f"    WARNING: skipping {name} (real test) — unsupported path type: {path_obj}")
            except Exception as e:
                print(f"    WARNING: skipping {name} (real test) — {e}")

    # ------------------------------------------------------------------ summary
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
    natural_test = sum(_safe_loader_len(l) for _, l in test_loaders)
    shown_train = natural_train if train_batches_per_epoch is None else train_batches_per_epoch
    shown_val = natural_val if val_batches_per_epoch is None else val_batches_per_epoch
    shown_test = natural_test if test_batches_per_epoch is None else test_batches_per_epoch
    parts = [f"{shown_train} train", f"{shown_val} val"]
    if test_loaders:
        parts.append(f"{shown_test} test")
    suffix = ""
    if shown_train != natural_train or shown_val != natural_val or (test_loaders and shown_test != natural_test):
        suffix_parts = [f"{natural_train} train", f"{natural_val} val"]
        if test_loaders:
            suffix_parts.append(f"{natural_test} test")
        suffix = f" (natural: {', '.join(suffix_parts)})"
    print(f"  Batches this epoch: {', '.join(parts)}{suffix}")

    return train_loader, val_loaders, test_loaders
