"""Toy components: pure stdlib, no torch/numpy required.

They exist only to prove the contracts can be wired end-to-end. They are not
part of the core and are not meant to be a good model/loss/metric library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from dlef.contracts import (
    DataSpec,
    LossOutput,
    MetricResult,
    ModelOutput,
    StepOutput,
)


@dataclass
class Param:
    value: float
    grad: float = 0.0


class LinearModel:
    """y = w*x + b, with manual gradients supplied by the task."""

    def __init__(self, rng: random.Random) -> None:
        self.w = Param(rng.uniform(-1.0, 1.0))
        self.b = Param(rng.uniform(-1.0, 1.0))

    def __call__(self, batch):
        return ModelOutput(predictions=[self.w.value * x + self.b.value for x in batch["x"]])

    def parameters(self) -> list[Param]:
        return [self.w, self.b]

    def state_dict(self):
        return {"w": self.w.value, "b": self.b.value}

    def load_state_dict(self, state):
        self.w.value = float(state["w"])
        self.b.value = float(state["b"])


class ToyDataset:
    """y = w_true*x + b_true + noise, yielded in batches."""

    def __init__(
        self,
        n: int,
        batch_size: int,
        noise: float,
        seed: int,
        w_true: float = 2.0,
        b_true: float = 1.0,
    ) -> None:
        rng = random.Random(seed)
        self.batch_size = batch_size
        self.samples = []
        for _ in range(n):
            x = rng.uniform(-1.0, 1.0)
            y = w_true * x + b_true + rng.gauss(0.0, noise)
            self.samples.append((x, y))
        self.spec = DataSpec(
            name="toy-linear",
            version=f"n{n}-noise{noise}-seed{seed}",
            schema={"x": "list[float]", "y": "list[float]"},
            splits={"train": "generated", "val": "generated"},
            transform="none",
        )

    def __iter__(self):
        for start in range(0, len(self.samples), self.batch_size):
            chunk = self.samples[start : start + self.batch_size]
            yield {"x": [item[0] for item in chunk], "y": [item[1] for item in chunk]}

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)


class LossValue:
    """Scalar loss with a backward callback.

    The core only requires that a loss object has an optional ``backward``.
    A PyTorch tensor satisfies this; the toy supplies a closure.
    """

    def __init__(self, value: float, backward_fn) -> None:
        self.value = float(value)
        self._backward_fn = backward_fn

    def backward(self) -> None:
        self._backward_fn()

    def item(self) -> float:
        return self.value

    def __float__(self) -> float:
        return self.value


class MSELoss:
    def __call__(self, output: ModelOutput, batch) -> LossOutput:
        preds = output.predictions
        targets = batch["y"]
        n = len(preds)
        diffs = [pred - target for pred, target in zip(preds, targets)]
        total = sum(diff * diff for diff in diffs) / n
        return LossOutput(total=total, terms={"mse": total})


class SGD:
    def __init__(self, parameters, lr: float) -> None:
        self.parameters = list(parameters)
        self.lr = float(lr)

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.value -= self.lr * parameter.grad

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = 0.0

    def state_dict(self):
        return {"lr": self.lr, "values": [parameter.value for parameter in self.parameters]}

    def load_state_dict(self, state) -> None:
        self.lr = float(state["lr"])
        for parameter, value in zip(self.parameters, state["values"]):
            parameter.value = float(value)


class MAEMetric:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, output: ModelOutput, batch) -> None:
        for pred, target in zip(output.predictions, batch["y"]):
            self.total += abs(pred - target)
            self.count += 1

    def compute(self) -> MetricResult:
        value = self.total / self.count if self.count else float("nan")
        return MetricResult(values={"mae": value})
