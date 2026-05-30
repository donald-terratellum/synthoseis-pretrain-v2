"""Validation visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from synthoseis_pre_train.plotting import make_crosssection_figure


def _log_validation_crosssections(
    writer: SummaryWriter,
    ds_name: str,
    first_input: torch.Tensor,
    first_output: torch.Tensor,
    first_target: torch.Tensor,
    epoch: int,
    avg_ds_loss: float,
) -> None:
    """Log per-dataset validation cross-sections to TensorBoard."""
    title_base = (
        f"{ds_name}  |  epoch {epoch + 1}  |  val loss {avg_ds_loss:.4f}"
    )
    for axis in ("x", "y"):
        for kind, vol in (("input", first_input), ("output", first_output), ("label", first_target)):
            title = f"{title_base}  |  center-{axis.upper()}  |  {kind}"
            fig = make_crosssection_figure(vol, title, axis=axis)
            tag = f"val_center{axis.upper()}/{kind}/{ds_name}"
            writer.add_figure(tag, fig, global_step=epoch + 1)
            plt.close(fig)
