"""Wire model / loss / optimizer / scheduler / metrics into a TorchTask."""

from __future__ import annotations

import torch

from adapters.torch import TorchOptimizer, TorchScheduler, TorchTask

from .config import ProjectConfig
from .loss import build_loss
from .metrics import build_metrics
from .model import build_model


def build_task(config: ProjectConfig, device: str | torch.device = "cpu") -> TorchTask:
    model = build_model(config.model, device=device)
    loss = build_loss(config)
    metrics = build_metrics(config)

    def optimizer_fn(model):
        return TorchOptimizer(
            torch.optim.Adam(
                model.parameters(),
                lr=config.optim.lr,
                weight_decay=config.optim.weight_decay,
            )
        )

    def scheduler_fn(optimizer):
        return TorchScheduler(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.optimizer, T_max=config.epochs
            ),
            step_unit="epoch",
        )

    return TorchTask(
        model=model,
        loss=loss,
        optimizer_fn=optimizer_fn,
        metrics=metrics,
        scheduler_fn=scheduler_fn,
        device=device,
    )
