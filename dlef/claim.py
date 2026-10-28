"""Claims and test-access accounting.

A Claim is a reported number/table/figure linked to one or more EvalRecords.
The test-access log records every test-split evaluation per data version, so
the report can flag reuse instead of silently treating test numbers as fresh.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .run import utc_now


class ClaimError(ValueError):
    pass


@dataclass
class Claim:
    claim_id: str
    text: str
    eval_ids: tuple[str, ...]
    metric: str = ""
    value: float | None = None
    unit: str = ""
    notes: str = ""
    tags: Mapping[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eval_ids"] = list(self.eval_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Claim":
        return cls(
            claim_id=str(payload["claim_id"]),
            text=str(payload.get("text", "")),
            eval_ids=tuple(str(item) for item in payload.get("eval_ids", ())),
            metric=str(payload.get("metric", "")),
            value=payload.get("value"),
            unit=str(payload.get("unit", "")),
            notes=str(payload.get("notes", "")),
            tags=dict(payload.get("tags", {})),
            created_at=str(payload.get("created_at", "")),
        )


class ClaimStore:
    """One JSON file per claim: ``<root>/<claim_id>.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, claim: Claim) -> Claim:
        _safe_id(claim.claim_id)
        if not claim.eval_ids:
            raise ClaimError("a claim must reference at least one eval id")
        path = self.root / f"{claim.claim_id}.json"
        if path.exists():
            raise ClaimError(f"claim already exists: {claim.claim_id}")
        path.write_text(
            json.dumps(claim.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return claim

    def get(self, claim_id: str) -> Claim:
        _safe_id(claim_id)
        path = self.root / f"{claim_id}.json"
        return Claim.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[Claim]:
        return [
            Claim.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.root.glob("*.json"))
        ]

    def remove(self, claim_id: str) -> bool:
        _safe_id(claim_id)
        path = self.root / f"{claim_id}.json"
        if not path.is_file():
            return False
        path.unlink()
        return True


class TestAccessLog:
    """Append-only test-split access log, grouped by data version."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, data_version: str) -> Path:
        digest = hashlib.sha256(data_version.encode("utf-8")).hexdigest()[:16]
        return self.root / f"{digest}.jsonl"

    def count(self, data_version: str, split: str = "test") -> int:
        path = self._path(data_version)
        if not path.is_file():
            return 0
        total = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("data_version") == data_version and payload.get("split") == split:
                total += 1
        return total

    def record(self, data_version: str, eval_id: str, split: str = "test") -> int:
        with self._lock:
            count = self.count(data_version, split) + 1
            payload = {
                "data_version": data_version,
                "eval_id": eval_id,
                "split": split,
                "n": count,
                "at": utc_now(),
            }
            with self._path(data_version).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return count

    def entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries


def _safe_id(value: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ClaimError(f"invalid claim id: {value!r}")
