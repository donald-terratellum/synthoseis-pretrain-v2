"""Validation batch scheduling helpers."""

from __future__ import annotations


def _compute_per_loader_targets(
    max_batches: int | None,
    n_loaders: int,
) -> list[int | None]:
    """Distribute max validation batches across loaders.

    When max_batches is None, each loader uses its natural length.
    Otherwise batches are split as evenly as possible, with remainder
    allocated to lower-index loaders first.
    """
    if max_batches is None:
        return [None] * n_loaders

    remaining_batches = max(1, int(max_batches))
    n = max(1, int(n_loaders))
    base = remaining_batches // n
    remainder = remaining_batches % n
    return [
        base + (1 if idx < remainder else 0)
        for idx in range(n_loaders)
    ]
