"""Per-dataset visualization helpers for training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from synthoseis_pre_train.plotting import make_4panel_figure


def _log_per_dataset_figures(
    model: nn.Module,
    merged_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    epoch_loss: float,
) -> None:
    """Log one 4-panel cross-section figure per source dataset to TensorBoard.

    Runs a single index-0 inference sample per sub-dataset in eval mode.
    Called once at the end of each training epoch; cost is negligible relative
    to the epoch itself.
    """

    def _get_live_example(requested_ds, all_datasets):
        candidates = [requested_ds] + [ds for ds in all_datasets if ds is not requested_ds]
        for candidate in candidates:
            try:
                inp, tgt, mask = candidate[0]
                return candidate, inp, tgt, mask
            except RuntimeError as exc:
                if "All array keys unavailable in zarr store" not in str(exc):
                    raise
        return None

    if not isinstance(merged_loader.dataset, ConcatDataset):
        import warnings
        warnings.warn(
            "_log_per_dataset_figures: merged_loader.dataset is not a ConcatDataset; "
            "skipping per-dataset figures.",
            stacklevel=2,
        )
        return

    model.eval()
    try:
        with torch.no_grad():
            import warnings
            all_datasets = cast(list[Any], list(merged_loader.dataset.datasets))
            for ds in all_datasets:
                ds_data_path = getattr(ds, "data_path", "unknown_dataset/model_data.zarr")
                ds_name = Path(ds_data_path).parent.name
                sample = _get_live_example(ds, all_datasets)
                if sample is None:
                    warnings.warn(
                        "_log_per_dataset_figures: no live zarr datasets remained at epoch end; "
                        "skipping remaining per-dataset figures.",
                        stacklevel=2,
                    )
                    break
                sample_ds, inp, tgt, _ = sample
                inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
                out_t = model(inp_t)
                tgt_t = torch.from_numpy(tgt).unsqueeze(0)
                sample_ds_data_path = getattr(sample_ds, "data_path", "unknown_dataset/model_data.zarr")
                sample_ds_name = Path(sample_ds_data_path).parent.name
                title = (
                    f"{ds_name}  |  epoch {epoch + 1}  |  loss {epoch_loss:.4f}"
                )
                if sample_ds is not ds:
                    title = f"{title}  |  example from {sample_ds_name}"
                fig = make_4panel_figure(
                    inp_t[0].cpu(), out_t[0].cpu(), tgt_t.cpu(), title
                )
                writer.add_figure(f"train/{ds_name}", fig, global_step=epoch + 1)
                plt.close(fig)
    finally:
        model.train()
