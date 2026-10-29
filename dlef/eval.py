"""Evaluation spec and record: the link between a checkpoint and a number."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .config import config_digest


@dataclass(frozen=True)
class EvalSpec:
    checkpoint: str
    split: str
    metrics: Sequence[str] = ()
    aggregation: str = "mean"
    ci_method: str | None = None
    test_access_policy: str = "once_per_claim"
    benchmark_id: str = ""
    test_set_id: str = ""


@dataclass(frozen=True)
class EvalRecord:
    eval_id: str
    split: str
    checkpoint_digest: str
    data_version: str
    eval_spec_digest: str
    metrics: Mapping[str, Any]
    predictions: str | None = None
    created_at: str = ""
    test_access: int = 0
    benchmark_id: str = ""
    test_set_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalRecord":
        return cls(**payload)


def eval_spec_digest(spec: EvalSpec) -> str:
    return config_digest(asdict(spec))
