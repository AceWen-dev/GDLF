from __future__ import annotations

import random
from typing import Mapping, Sequence

from dlef.contracts import (
    Batch,
    Callback,
    Metric,
    Model,
    Optimizer,
    Scheduler,
    StepOutput,
)

from .components import LinearModel, LossValue, MAEMetric, MSELoss, SGD
from .config import ToyConfig


class ToyTask:
    """The only domain-aware object in the toy example."""

    def __init__(self, config: ToyConfig) -> None:
        self.config = config
        rng = random.Random(config.seed)
        self.model = LinearModel(rng)
        self.loss_fn = MSELoss()

    def get_model(self) -> Model:
        return self.model

    def train_step(self, batch: Batch) -> StepOutput:
        output = self.model(batch)
        loss_out = self.loss_fn(output, batch)
        preds = output.predictions
        targets = batch["y"]
        inputs = batch["x"]
        n = len(preds)
        diffs = [pred - target for pred, target in zip(preds, targets)]

        def backward() -> None:
            scale = 2.0 / n
            self.model.w.grad = sum(scale * diff * x for diff, x in zip(diffs, inputs))
            self.model.b.grad = sum(scale * diff for diff in diffs)

        return StepOutput(
            loss=LossValue(loss_out.total, backward),
            terms=dict(loss_out.terms),
            predictions=preds,
        )

    def val_step(self, batch: Batch) -> StepOutput:
        output = self.model(batch)
        return StepOutput(loss=None, predictions=output.predictions)

    def configure_optimizer(self, model: Model) -> Optimizer:
        return SGD(model.parameters(), lr=self.config.lr)

    def configure_scheduler(self, optimizer: Optimizer) -> Scheduler | None:
        return None

    def configure_metrics(self) -> Mapping[str, Metric]:
        return {"mae": MAEMetric()}

    def configure_callbacks(self) -> Sequence[Callback]:
        return []
