"""Candidates: the shortlist between raw runs and reported claims.

A Candidate is a human judgment that one or more runs are worth keeping and
inspecting. It is not a measurement and not a reported number — claims keep
that role. A candidate is per run-set (normally one configuration across
seeds), records *why* it was shortlisted and *what to look at next*, and moves
through a small status lifecycle.

It is also the layer that protects promising runs from retention before they
become claims. ``RetentionService`` keeps runs referenced by active candidates.

Candidates are an **owner-approved registry**: a run set enters the shortlist,
gets promoted or rejected only with the owner's explicit approval, recorded in
``approved_by``. Agents propose candidates; they never approve their own.

Files: ``<root>/<candidate_id>.json``, one per candidate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .run import utc_now

CANDIDATE_STATUSES = ("candidate", "promoted", "rejected", "archived")

# Active candidates are protected from gc; rejected/archived are not.
CANDIDATE_ACTIVE_STATUSES = ("candidate", "promoted")


class CandidateError(ValueError):
    pass


@dataclass
class Candidate:
    """A named, reasoned pointer to a set of runs."""

    candidate_id: str
    title: str
    run_ids: tuple[str, ...]
    status: str = "candidate"
    reason: str = ""
    next: str = ""
    metric: str = ""
    experiment: str = ""
    claim_ids: tuple[str, ...] = ()
    priority: int = 0
    tags: Mapping[str, str] = field(default_factory=dict)
    # Owner approval is part of the record, not of the chat log: a candidate
    # enters the shortlist only when the owner says so, and the record says
    # who approved it. Agents must ask, never self-approve.
    approved_by: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        # Duplicates never add information; keep the first occurrence so a
        # copy-pasted run list does not fail validation.
        self.run_ids = tuple(dict.fromkeys(self.run_ids))
        self.claim_ids = tuple(dict.fromkeys(self.claim_ids))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_ids"] = list(self.run_ids)
        payload["claim_ids"] = list(self.claim_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Candidate":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            title=str(payload.get("title", "")),
            run_ids=tuple(str(item) for item in payload.get("run_ids", ())),
            status=str(payload.get("status", "candidate")),
            reason=str(payload.get("reason", "")),
            next=str(payload.get("next", "")),
            metric=str(payload.get("metric", "")),
            experiment=str(payload.get("experiment", "")),
            claim_ids=tuple(str(item) for item in payload.get("claim_ids", ())),
            priority=int(payload.get("priority", 0) or 0),
            tags=dict(payload.get("tags", {})),
            approved_by=str(payload.get("approved_by", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )

    def validate(self) -> None:
        """Enforce the invariants that make a candidate meaningful."""

        if not self.candidate_id.strip():
            raise CandidateError("candidate_id must not be empty")
        if not self.title.strip():
            raise CandidateError("a candidate needs a title")
        if not self.run_ids:
            raise CandidateError("a candidate must reference at least one run id")
        if self.status not in CANDIDATE_STATUSES:
            raise CandidateError(
                f"invalid candidate status {self.status!r}; "
                f"expected one of {list(CANDIDATE_STATUSES)}"
            )
        if self.status in {"candidate", "rejected"} and not self.reason.strip():
            raise CandidateError(
                f"a {self.status} candidate needs a reason: "
                "why is it kept or rejected?"
            )
        if self.status == "promoted" and not self.claim_ids:
            raise CandidateError(
                "a promoted candidate must reference at least one claim id"
            )
        if not self.approved_by.strip():
            raise CandidateError(
                "a candidate needs approved_by: the owner's approval that these "
                "runs enter the shortlist. Agents propose; the owner approves."
            )


class CandidateStore:
    """One JSON file per candidate: ``<root>/<candidate_id>.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, candidate_id: str) -> Path:
        _safe_id(candidate_id)
        return self.root / f"{candidate_id}.json"

    def _write(self, candidate: Candidate) -> None:
        self._path(candidate.candidate_id).write_text(
            json.dumps(
                candidate.to_dict(), indent=2, ensure_ascii=False, default=str
            )
            + "\n",
            encoding="utf-8",
        )

    def add(self, candidate: Candidate) -> Candidate:
        candidate.validate()
        path = self._path(candidate.candidate_id)
        if path.exists():
            raise CandidateError(f"candidate already exists: {candidate.candidate_id}")
        self._write(candidate)
        return candidate

    def get(self, candidate_id: str) -> Candidate:
        path = self._path(candidate_id)
        if not path.is_file():
            raise CandidateError(f"candidate not found: {candidate_id}")
        return Candidate.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, status: str | Iterable[str] | None = None) -> list[Candidate]:
        candidates = [
            Candidate.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.root.glob("*.json"))
        ]
        if status is None:
            return candidates
        wanted = {status} if isinstance(status, str) else set(status)
        return [candidate for candidate in candidates if candidate.status in wanted]

    def update(self, identifier: str, **changes: Any) -> Candidate:
        """Apply explicit field changes and bump ``updated_at``.

        ``None`` values are ignored, which makes CLI call sites simple.
        Unknown field names are rejected instead of silently ignored.
        """

        candidate = self.get(identifier)
        payload = candidate.to_dict()
        unknown = set(changes) - set(payload)
        if unknown:
            raise CandidateError(
                f"unknown candidate fields: {', '.join(sorted(unknown))}"
            )
        if "candidate_id" in changes and changes["candidate_id"] != identifier:
            raise CandidateError("candidate_id cannot be changed; create a new candidate")
        for key, value in changes.items():
            if value is not None:
                payload[key] = value
        payload["updated_at"] = utc_now()
        updated = Candidate.from_dict(payload)
        updated.validate()
        self._write(updated)
        return updated

    def remove(self, candidate_id: str) -> bool:
        path = self._path(candidate_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def protected_run_ids(
        self, statuses: Iterable[str] = CANDIDATE_ACTIVE_STATUSES
    ) -> dict[str, list[str]]:
        """Map run id -> candidate ids that keep this run from being deleted."""

        wanted = set(statuses)
        protected: dict[str, list[str]] = {}
        for candidate in self.list():
            if candidate.status not in wanted:
                continue
            for run_id in candidate.run_ids:
                protected.setdefault(run_id, []).append(candidate.candidate_id)
        return protected


def _safe_id(value: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise CandidateError(f"invalid candidate id: {value!r}")
