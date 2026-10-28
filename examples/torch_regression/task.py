from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from adapters.torch import (
    TorchLoss,
    TorchMetric,
    TorchModel,
    TorchOptimizer,
    TorchScheduler,
    TorchTask,
)
from dlef.contracts import LossOutput

from .config import TorchConfig


def mse_loss(output, batch) -> LossOutput:
    total = F.mse_loss(output.predictions, batch["y"])
    return LossOutput(total=total, terms={"mse": total})


def build_task(config: TorchConfig, device: str | torch.device = "cpu") -> TorchTask:
    module = nn.Sequential(
        nn.Linear(1, config.hidden),
        nn.ReLU(),
        nn.Linear(config.hidden, 1),
    )
    model = TorchModel(module, lambda module, batch: module(batch["x"]), device=device)
    loss = TorchLoss(mse_loss)
    optimizer_fn = lambda model: TorchOptimizer(  # noqa: E731
        torch.optim.Adam(model.parameters(), lr=config.lr)
    )
    scheduler_fn = lambda optimizer: TorchScheduler(  # noqa: E731
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.optimizer, T_max=config.epochs
        ),
        step_unit="epoch",
    )
    metrics = {
        "mae": TorchMetric(lambda preds, targets: {"mae": (preds - targets).abs().mean()}),
    }
    return TorchTask(
        model=model,
        loss=loss,
        optimizer_fn=optimizer_fn,
        metrics=metrics,
        scheduler_fn=scheduler_fn,
        device=device,
    )
