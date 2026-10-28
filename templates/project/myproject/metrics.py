"""Placeholder metrics. Replace with your task's metrics."""

from __future__ import annotations

from adapters.torch import TorchMetric
from dlef.contracts import Metric

from .config import ProjectConfig


def build_metrics(config: ProjectConfig) -> dict[str, Metric]:
    return {
        "mae": TorchMetric(
            lambda preds, targets: {"mae": (preds - targets).abs().mean()}
        ),
    }
