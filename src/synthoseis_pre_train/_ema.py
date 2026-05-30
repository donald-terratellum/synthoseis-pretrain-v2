"""Exponential moving average helper for model weights."""

from __future__ import annotations

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self.backup = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                shadow_tensor = self.shadow[name]
                if torch.is_floating_point(shadow_tensor):
                    shadow_tensor.mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
                else:
                    shadow_tensor.copy_(tensor)

    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        for name, tensor in self.shadow.items():
            if name in shadow:
                self.shadow[name].copy_(shadow[name].to(device=tensor.device, dtype=tensor.dtype))
