from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from dlef.contracts import (
    Batch,
    Callback,
    DataSpec,
    LossOutput,
    Metric,
    MetricResult,
    ModelOutput,
    Optimizer,
    Scheduler,
    StepOutput,
)


def move_batch(batch: Batch, device: str | torch.device) -> dict[str, Any]:
    """Move every tensor value in a batch to ``device``; leave others alone."""

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class TorchDataLoader:
    """Adapter from ``torch.utils.data.DataLoader`` to the core Dataset contract."""

    def __init__(self, loader: DataLoader, spec: DataSpec | None = None) -> None:
        self.loader = loader
        self.spec = spec

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch to a DistributedSampler when present."""
        sampler = getattr(self.loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


class TorchModel:
    """Wrap an ``nn.Module`` plus an explicit forward function.

    The forward function receives ``(module, batch)`` and returns predictions.
    Inputs are expected to be on the module's device; ``TorchTask`` moves them.
    """

    def __init__(
        self,
        module: nn.Module,
        forward_fn: Callable[[nn.Module, Batch], Any],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.module = module.to(device)
        self.device = torch.device(device)
        self.forward_fn = forward_fn

    def __call__(self, batch: Batch) -> ModelOutput:
        return ModelOutput(predictions=self.forward_fn(self.module, batch))

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def _inner_module(self):
        return self.module.module if hasattr(self.module, "module") else self.module

    def state_dict(self):
        return self._inner_module().state_dict()

    def load_state_dict(self, state) -> None:
        self._inner_module().load_state_dict(state)


class TorchLoss:
    """Wrap a callable returning ``LossOutput`` (or a scalar tensor)."""

    def __init__(self, loss_fn: Callable[[ModelOutput, Batch], LossOutput]) -> None:
        self.loss_fn = loss_fn

    def __call__(self, output: ModelOutput, batch: Batch) -> LossOutput:
        result = self.loss_fn(output, batch)
        if isinstance(result, LossOutput):
            return result
        if torch.is_tensor(result):
            return LossOutput(total=result, terms={"loss": result})
        raise TypeError("TorchLoss function must return LossOutput or a tensor")


class TorchOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state) -> None:
        self.optimizer.load_state_dict(state)


class TorchScheduler:
    def __init__(self, scheduler, *, step_unit: str = "epoch") -> None:
        if step_unit not in {"epoch", "step"}:
            raise ValueError("step_unit must be 'epoch' or 'step'")
        self.scheduler = scheduler
        self.step_unit = step_unit

    def step(self) -> None:
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state) -> None:
        self.scheduler.load_state_dict(state)


class TorchMetric:
    """Accumulate predictions/targets, then call a user compute function.

    ``compute_fn(preds, targets)`` returns a mapping of name -> scalar.
    """

    def __init__(
        self,
        compute_fn: Callable[[torch.Tensor, torch.Tensor], Mapping[str, Any]],
        *,
        target_key: str = "y",
    ) -> None:
        self.compute_fn = compute_fn
        self.target_key = target_key
        self.reset()

    def reset(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, output: ModelOutput, batch: Batch) -> None:
        preds = output.predictions
        targets = batch[self.target_key]
        if not torch.is_tensor(preds) or not torch.is_tensor(targets):
            raise TypeError("TorchMetric expects tensor predictions and tensor targets")
        self._preds.append(preds.detach().cpu().reshape(-1))
        self._targets.append(targets.detach().cpu().reshape(-1))

    def compute(self) -> MetricResult:
        if not self._preds:
            return MetricResult(values={})
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        values = self.compute_fn(preds, targets)
        return MetricResult(values={key: float(value) for key, value in values.items()})


class TorchTask:
    """Supervised-style Task built from adapter components.

    It owns device placement for the batch. The loss and metric decide what
    the predictions/targets mean; the core does not.
    """

    def __init__(
        self,
        *,
        model: TorchModel,
        loss: TorchLoss,
        optimizer_fn: Callable[[TorchModel], TorchOptimizer],
        metrics: Mapping[str, Metric],
        scheduler_fn: Callable[[TorchOptimizer], TorchScheduler | None] | None = None,
        callbacks: Sequence[Callback] = (),
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer_fn = optimizer_fn
        self.metrics = dict(metrics)
        self.scheduler_fn = scheduler_fn
        self.callbacks = list(callbacks)
        self.device = torch.device(device) if device is not None else model.device

    def get_model(self) -> TorchModel:
        return self.model

    def train_step(self, batch: Batch) -> StepOutput:
        batch = move_batch(batch, self.device)
        output = self.model(batch)
        loss = self.loss(output, batch)
        return StepOutput(loss=loss.total, terms=loss.terms, predictions=output.predictions)

    def val_step(self, batch: Batch) -> StepOutput:
        batch = move_batch(batch, self.device)
        output = self.model(batch)
        return StepOutput(loss=None, predictions=output.predictions)

    def configure_optimizer(self, model: TorchModel) -> Optimizer:
        return self.optimizer_fn(model)

    def configure_scheduler(self, optimizer: Optimizer) -> Scheduler | None:
        return self.scheduler_fn(optimizer) if self.scheduler_fn is not None else None

    def configure_metrics(self) -> Mapping[str, Metric]:
        return self.metrics

    def configure_callbacks(self) -> Sequence[Callback]:
        return self.callbacks
