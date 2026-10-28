"""Placeholder model. Replace with your architecture."""

from __future__ import annotations

import torch
from torch import nn

from adapters.torch import TorchModel

from .config import ModelConfig


def build_model(config: ModelConfig, device: str | torch.device = "cpu") -> TorchModel:
    if config.name == "mlp":
        module = nn.Sequential(
            nn.Linear(1, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, 1),
        )
    elif config.name == "linear":
        module = nn.Linear(1, 1)
    else:
        raise ValueError(f"unknown model.name: {config.name!r}")

    return TorchModel(module, lambda module, batch: module(batch["x"]), device=device)
