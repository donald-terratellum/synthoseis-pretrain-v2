"""Helpers for fetching train batches with restart-on-exhaustion behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainBatchFetchResult:
    loader_iter: Iterator
    input_data: torch.Tensor | None
    target: torch.Tensor | None
    mask: torch.Tensor | None
    source_tags: list[str] | None
    reload_requested: bool
    should_break: bool


def _ensure_5d_batch(t: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize batch tensors to (B, C, D, H, W).

    Legacy dataset code can yield 4D (B, D, H, W), while current datasets yield
    5D (B, 1, D, H, W). This helper supports both and rejects unexpected ranks.
    """
    if t.ndim == 4:
        t = t.unsqueeze(1)
    elif t.ndim != 5:
        raise ValueError(f"Expected 4D/5D tensor for {name}, got shape={tuple(t.shape)}")
    return t


def _fetch_train_batch(
    loader_iter: Iterator,
    train_loader: DataLoader,
    device: torch.device,
    batch_idx: int,
) -> TrainBatchFetchResult:
    """Fetch next training batch, restarting iterator once on StopIteration."""
    try:
        batch = next(loader_iter)
        if len(batch) == 4:
            input_data, target, mask, source_tags = batch
            source_tags = [str(v) for v in source_tags]
        else:
            input_data, target, mask = batch
            source_tags = None
        input_data = _ensure_5d_batch(input_data, "input_data").float().to(device, non_blocking=True)
        target = _ensure_5d_batch(target, "target").float().to(device, non_blocking=True)
        mask = _ensure_5d_batch(mask, "mask").to(device, non_blocking=True)
        return TrainBatchFetchResult(loader_iter, input_data, target, mask, source_tags, False, False)
    except StopIteration:
        loader_iter = iter(train_loader)
        try:
            batch = next(loader_iter)
            if len(batch) == 4:
                input_data, target, mask, source_tags = batch
                source_tags = [str(v) for v in source_tags]
            else:
                input_data, target, mask = batch
                source_tags = None
            input_data = _ensure_5d_batch(input_data, "input_data").float().to(device, non_blocking=True)
            target = _ensure_5d_batch(target, "target").float().to(device, non_blocking=True)
            mask = _ensure_5d_batch(mask, "mask").to(device, non_blocking=True)
            return TrainBatchFetchResult(loader_iter, input_data, target, mask, source_tags, False, False)
        except Exception as exc:
            print(f"    WARNING: loader exhausted/unavailable at batch {batch_idx} — {exc}")
            return TrainBatchFetchResult(loader_iter, None, None, None, None, True, True)
    except Exception as exc:
        print(f"    WARNING: skipping batch {batch_idx} — {exc}")
        return TrainBatchFetchResult(loader_iter, None, None, None, None, True, True)
