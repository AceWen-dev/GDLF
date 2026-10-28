"""Training policies: the seam for AMP, gradient accumulation, clipping, DDP.

The core owns the loop; a policy owns the parts that depend on the training
stack. The default ``NoOpTrainingPolicy`` reproduces plain single-process
training. The torch adapter supplies ``TorchTrainingPolicy``.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import Optimizer, Task


@runtime_checkable
class TrainingPolicy(Protocol):
    @property
    def is_main_process(self) -> bool: ...

    def setup(self) -> None: ...

    def on_fit_start(self, task: Task, optimizer: Optimizer, config: Any) -> None: ...

    def forward_context(self): ...

    def scale_loss(self, loss: Any) -> Any: ...

    def backward(self, loss: Any) -> None: ...

    def should_step(self, step: int) -> bool: ...

    def should_flush(self, step: int) -> bool: ...

    def optimizer_step(self, task: Task, optimizer: Optimizer) -> None: ...

    def zero_grad(self, optimizer: Optimizer) -> None: ...

    def on_fit_end(self, task: Task, optimizer: Optimizer) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class NoOpTrainingPolicy:
    """Plain training: backward every batch, step every batch, no AMP."""

    is_main_process = True

    def setup(self) -> None:
        pass

    def on_fit_start(self, task: Task, optimizer: Optimizer, config: Any) -> None:
        pass

    def forward_context(self):
        return nullcontext()

    def scale_loss(self, loss: Any) -> Any:
        return loss

    def backward(self, loss: Any) -> None:
        loss.backward()

    def should_step(self, step: int) -> bool:
        return True

    def should_flush(self, step: int) -> bool:
        return False

    def optimizer_step(self, task: Task, optimizer: Optimizer) -> None:
        optimizer.step()

    def zero_grad(self, optimizer: Optimizer) -> None:
        optimizer.zero_grad()

    def on_fit_end(self, task: Task, optimizer: Optimizer) -> None:
        pass

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass
