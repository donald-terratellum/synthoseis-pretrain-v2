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
    reload_requested: bool
    should_break: bool


def _fetch_train_batch(
    loader_iter: Iterator,
    train_loader: DataLoader,
    device: torch.device,
    batch_idx: int,
) -> TrainBatchFetchResult:
    """Fetch next training batch, restarting iterator once on StopIteration."""
    try:
        input_data, target, mask = next(loader_iter)
        input_data = input_data.unsqueeze(1).float().to(device, non_blocking=True)
        target = target.unsqueeze(1).float().to(device, non_blocking=True)
        mask = mask.unsqueeze(1).to(device, non_blocking=True)
        return TrainBatchFetchResult(loader_iter, input_data, target, mask, False, False)
    except StopIteration:
        loader_iter = iter(train_loader)
        try:
            input_data, target, mask = next(loader_iter)
            input_data = input_data.unsqueeze(1).float().to(device, non_blocking=True)
            target = target.unsqueeze(1).float().to(device, non_blocking=True)
            mask = mask.unsqueeze(1).to(device, non_blocking=True)
            return TrainBatchFetchResult(loader_iter, input_data, target, mask, False, False)
        except Exception as exc:
            print(f"    WARNING: loader exhausted/unavailable at batch {batch_idx} — {exc}")
            return TrainBatchFetchResult(loader_iter, None, None, None, True, True)
    except Exception as exc:
        print(f"    WARNING: skipping batch {batch_idx} — {exc}")
        return TrainBatchFetchResult(loader_iter, None, None, None, True, True)
