"""Domain-agnostic contracts for DLEF.

Hard rule: this module contains no domain words (no image/text/audio/video,
no bbox/depth/mm, no classification/detection/segmentation, no RL/LLM).
A batch is an opaque mapping; the core never interprets its semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

Batch = Mapping[str, Any]
Tensor = Any


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSpec:
    """Immutable description of a dataset.

    ``schema`` describes batch keys/dtypes/shapes. ``splits`` maps a split name
    to a manifest path or digest. ``transform`` is the transform version.
    """

    name: str
    version: str
    schema: Mapping[str, Any] = field(default_factory=dict)
    splits: Mapping[str, str] = field(default_factory=dict)
    transform: str = ""
    stats: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Dataset(Protocol):
    """An iterable of batches.

    Batching, shuffling, padding and collation are adapter responsibilities.
    This keeps streaming datasets and non-tensor data first-class.
    """

    def __iter__(self) -> Iterator[Batch]: ...


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelOutput:
    predictions: Any
    aux: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Model(Protocol):
    """A pure forward function plus its state.

    Models must not do I/O, read config, move devices, or compute losses.
    Device placement is a Runner concern.
    """

    def __call__(self, batch: Batch) -> ModelOutput: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    terms: Mapping[str, Tensor] = field(default_factory=dict)


@runtime_checkable
class Loss(Protocol):
    def __call__(self, output: ModelOutput, batch: Batch) -> LossOutput: ...


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------


@runtime_checkable
class Optimizer(Protocol):
    def step(self) -> None: ...

    def zero_grad(self) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class Scheduler(Protocol):
    def step(self) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    values: Mapping[str, float]
    meta: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    def update(self, output: ModelOutput, batch: Batch) -> None: ...

    def compute(self) -> MetricResult: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    run_id: str
    config: Any
    manifest: Any
    epoch: int = 0
    step: int = 0
    device: str = "cpu"
    metrics: Mapping[str, float] = field(default_factory=dict)


class CallbackBase:
    """Convenience base class with no-op hooks.

    Callbacks may implement any subset; the Runner uses ``getattr`` so
    inheritance is not required.
    """

    def on_run_start(self, ctx: RunContext) -> None:
        pass

    def on_epoch_start(self, ctx: RunContext) -> None:
        pass

    def on_step_end(self, ctx: RunContext, step: "StepOutput") -> None:
        pass

    def on_epoch_end(self, ctx: RunContext) -> None:
        pass

    def on_run_end(self, ctx: RunContext) -> None:
        pass

    def on_exception(self, ctx: RunContext, error: BaseException) -> None:
        pass


@runtime_checkable
class Callback(Protocol):
    def on_run_start(self, ctx: RunContext) -> None: ...

    def on_epoch_start(self, ctx: RunContext) -> None: ...

    def on_step_end(self, ctx: RunContext, step: "StepOutput") -> None: ...

    def on_epoch_end(self, ctx: RunContext) -> None: ...

    def on_run_end(self, ctx: RunContext) -> None: ...

    def on_exception(self, ctx: RunContext, error: BaseException) -> None: ...


# ---------------------------------------------------------------------------
# Task / step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOutput:
    loss: Tensor | None
    terms: Mapping[str, Tensor] = field(default_factory=dict)
    predictions: Any | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Task(Protocol):
    """The only place where domain components are assembled.

    Standard path: ``train_step`` returns a loss; the Runner calls
    ``backward``/``step``. Escape hatch: implement ``custom_train_step`` and
    the Runner will call it instead, leaving lifecycle management to the Runner.
    """

    def get_model(self) -> Model: ...

    def train_step(self, batch: Batch) -> StepOutput: ...

    def val_step(self, batch: Batch) -> StepOutput: ...

    def configure_optimizer(self, model: Model) -> Optimizer: ...

    def configure_scheduler(self, optimizer: Optimizer) -> Scheduler | None: ...

    def configure_metrics(self) -> Mapping[str, Metric]: ...

    def configure_callbacks(self) -> Sequence[Callback]: ...


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    model: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any] | None
    rng_state: Any
    epoch: int
    step: int
    manifest: Mapping[str, Any]
    data_position: Mapping[str, Any] | None = None
    policy: Mapping[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_task(task: Any) -> None:
    """Validate the Task contract without relying on Protocol data members."""

    required = (
        "get_model",
        "train_step",
        "val_step",
        "configure_optimizer",
        "configure_scheduler",
        "configure_metrics",
        "configure_callbacks",
    )
    missing = [name for name in required if not callable(getattr(task, name, None))]
    if missing:
        raise TypeError(f"Task is missing required methods: {', '.join(missing)}")


def to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
