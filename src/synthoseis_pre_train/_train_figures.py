"""Training visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from synthoseis_pre_train.plotting import make_4panel_figure


def _log_train_merged_figure(
    writer: SummaryWriter,
    last_input: torch.Tensor,
    last_output: torch.Tensor,
    last_target: torch.Tensor,
    epoch: int,
    avg_epoch_loss: float,
) -> None:
    """Log merged train prediction figure to TensorBoard."""
    title = f"merged-train  |  epoch {epoch + 1}  |  loss {avg_epoch_loss:.4f}"
    fig = make_4panel_figure(last_input, last_output, last_target, title)
    writer.add_figure("train/merged", fig, global_step=epoch + 1)
    plt.close(fig)
