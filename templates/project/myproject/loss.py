"""Placeholder losses. Replace with your objectives and named terms.

Switch with ``--set loss.name=huber`` or a config field. A loss is a new run,
never an edit to an old run.
"""

from __future__ import annotations

import torch.nn.functional as F

from adapters.torch import TorchLoss
from dlef.contracts import LossOutput

from .config import ProjectConfig


def build_loss(config: ProjectConfig) -> TorchLoss:
    name = config.loss.name

    if name == "mse":
        def loss_fn(output, batch) -> LossOutput:
            total = F.mse_loss(output.predictions, batch["y"])
            return LossOutput(total=total, terms={"mse": total})

    elif name == "huber":
        def loss_fn(output, batch) -> LossOutput:
            total = F.smooth_l1_loss(
                output.predictions,
                batch["y"],
                beta=config.loss.huber_beta,
            )
            return LossOutput(total=total, terms={"huber": total})

    else:
        raise ValueError(f"unknown loss.name: {name!r}")

    return TorchLoss(loss_fn)
