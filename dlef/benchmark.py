"""Benchmark identity: test set + evaluation protocol.

A number is comparable only within one benchmark. The benchmark is split into
two identities:

- ``test_set_id``: dataset + split membership. Used for leakage/access
  accounting; changing labels, the evaluation transform or the metric does not
  change it.
- ``benchmark_id``: ``test_set_id`` + label version + eval transform + metric
  spec + aggregation + evaluator version. Used for comparison; any of these
  changing creates a new benchmark.

The core only owns the identity contract. Building membership, checking
duplicates/leakage and sealing labels belong to the data layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Stable ``sha256:`` digest of a JSON-compatible mapping."""
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def make_test_set_id(
    *,
    dataset: str,
    split: str,
    membership_digest: str,
) -> str:
    return canonical_digest(
        {
            "dataset": dataset,
            "split": split,
            "membership_digest": membership_digest,
        }
    )


@dataclass(frozen=True)
class BenchmarkSpec:
    """Immutable benchmark manifest.

    ``metrics`` maps metric name to its implementation/spec version, not just
    the metric name. ``thresholds`` records any fixed decision threshold.
    """

    name: str
    dataset: str
    dataset_version: str
    split: str
    membership_digest: str
    label_version: str = ""
    transform: str = ""
    metrics: Mapping[str, str] = field(default_factory=dict)
    aggregation: str = "mean"
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    evaluator_version: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "dataset",
            "dataset_version",
            "split",
            "membership_digest",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"benchmark.{field_name} must not be empty")
        if not self.metrics:
            raise ValueError("benchmark.metrics must not be empty")
        for name, version in self.metrics.items():
            if not str(name).strip() or not str(version).strip():
                raise ValueError("benchmark.metrics keys/values must not be empty")

    @property
    def test_set_id(self) -> str:
        return make_test_set_id(
            dataset=self.dataset,
            split=self.split,
            membership_digest=self.membership_digest,
        )

    @property
    def benchmark_id(self) -> str:
        return canonical_digest(
            {
                "test_set_id": self.test_set_id,
                "label_version": self.label_version,
                "transform": self.transform,
                "metrics": dict(self.metrics),
                "aggregation": self.aggregation,
                "thresholds": dict(self.thresholds),
                "evaluator_version": self.evaluator_version,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "membership_digest": self.membership_digest,
            "test_set_id": self.test_set_id,
            "label_version": self.label_version,
            "transform": self.transform,
            "metrics": dict(self.metrics),
            "aggregation": self.aggregation,
            "thresholds": dict(self.thresholds),
            "evaluator_version": self.evaluator_version,
            "benchmark_id": self.benchmark_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkSpec":
        return cls(
            name=str(payload["name"]),
            dataset=str(payload["dataset"]),
            dataset_version=str(payload["dataset_version"]),
            split=str(payload["split"]),
            membership_digest=str(payload["membership_digest"]),
            label_version=str(payload.get("label_version", "")),
            transform=str(payload.get("transform", "")),
            metrics={str(key): str(value) for key, value in payload.get("metrics", {}).items()},
            aggregation=str(payload.get("aggregation", "mean")),
            thresholds=dict(payload.get("thresholds", {})),
            evaluator_version=str(payload.get("evaluator_version", "")),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSpec":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = cls.from_dict(payload)
        expected_benchmark = payload.get("benchmark_id")
        if expected_benchmark and expected_benchmark != spec.benchmark_id:
            raise ValueError(
                f"benchmark manifest id mismatch in {path}: "
                f"stored {expected_benchmark}, computed {spec.benchmark_id}"
            )
        expected_test_set = payload.get("test_set_id")
        if expected_test_set and expected_test_set != spec.test_set_id:
            raise ValueError(
                f"benchmark test set id mismatch in {path}: "
                f"stored {expected_test_set}, computed {spec.test_set_id}"
            )
        return spec
