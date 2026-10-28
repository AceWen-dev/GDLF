"""Reference-checked retention and garbage collection.

A plan is always computed before anything is deleted. The default action is a
dry-run; ``apply(..., dry_run=False)`` is explicit.

What is protected:

- runs kept by recency / best metric / every-N;
- runs with protected tags or statuses;
- young runs (``min_age_days``);
- parents of kept runs (lineage closure);
- runs referenced by sweep records (unless disabled);
- artifacts referenced by any kept run, or younger than ``artifact_ttl_days``.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .run import ArtifactStore, Run, RunStore
from .sweep import SweepStore


@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int = 5
    keep_best: int = 0
    keep_every: int = 0
    metric_key: str = ""
    metric_mode: str = "min"  # min | max
    keep_tags: frozenset[str] = frozenset({"reported"})
    protect_status: frozenset[str] = frozenset({"running"})
    min_age_days: float = 0.0
    artifact_ttl_days: float = 30.0
    protect_sweep_runs: bool = True
    sweep_ttl_days: float | None = None

    def validate(self) -> None:
        if self.keep_last < 0:
            raise ValueError("keep_last must be >= 0")
        if self.keep_best < 0:
            raise ValueError("keep_best must be >= 0")
        if self.keep_every < 0:
            raise ValueError("keep_every must be >= 0")
        if self.keep_best > 0 and not self.metric_key:
            raise ValueError("metric_key is required when keep_best > 0")
        if self.metric_mode not in {"min", "max"}:
            raise ValueError("metric_mode must be 'min' or 'max'")
        if self.min_age_days < 0:
            raise ValueError("min_age_days must be >= 0")
        if self.artifact_ttl_days < 0:
            raise ValueError("artifact_ttl_days must be >= 0")
        if self.sweep_ttl_days is not None and self.sweep_ttl_days < 0:
            raise ValueError("sweep_ttl_days must be >= 0 or None")


@dataclass(frozen=True)
class RunAction:
    run_id: str
    project: str
    experiment: str
    action: str  # keep | delete
    reason: str
    size_bytes: int
    path: str
    started_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project": self.project,
            "experiment": self.experiment,
            "action": self.action,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "path": self.path,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class ArtifactAction:
    digest: str
    name: str
    action: str  # keep | delete
    reason: str
    size_bytes: int
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "name": self.name,
            "action": self.action,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "path": self.path,
        }


@dataclass(frozen=True)
class GcPlan:
    runs: tuple[RunAction, ...] = ()
    artifacts: tuple[ArtifactAction, ...] = ()

    @property
    def delete_runs(self) -> tuple[RunAction, ...]:
        return tuple(action for action in self.runs if action.action == "delete")

    @property
    def delete_artifacts(self) -> tuple[ArtifactAction, ...]:
        return tuple(action for action in self.artifacts if action.action == "delete")

    @property
    def bytes_reclaimable(self) -> int:
        return sum(action.size_bytes for action in self.delete_runs) + sum(
            action.size_bytes for action in self.delete_artifacts
        )

    def summary(self, *, include_kept: bool = False) -> str:
        rows: list[tuple[str, str, str, str, str]] = []
        for action in self.runs:
            if include_kept or action.action == "delete":
                rows.append(
                    ("run", action.run_id, action.action, action.reason, _format_bytes(action.size_bytes))
                )
        for action in self.artifacts:
            if include_kept or action.action == "delete":
                rows.append(
                    (
                        "artifact",
                        f"{action.digest[:8]}/{action.name}",
                        action.action,
                        action.reason,
                        _format_bytes(action.size_bytes),
                    )
                )
        if not rows:
            kept_runs = len(self.runs) - len(self.delete_runs)
            kept_artifacts = len(self.artifacts) - len(self.delete_artifacts)
            return f"(no deletions; kept {kept_runs} runs, {kept_artifacts} artifacts)"
        rows.sort(key=lambda row: (0 if row[2] == "delete" else 1, row[0], row[1]))
        lines = [
            "| kind | id | action | reason | size |",
            "|---|---|---|---|---|",
        ]
        for kind, identifier, action, reason, size in rows:
            lines.append(f"| {kind} | {identifier} | {action} | {reason} | {size} |")
        lines.append("")
        lines.append(
            f"reclaimable: {_format_bytes(self.bytes_reclaimable)} "
            f"({len(self.delete_runs)} runs, {len(self.delete_artifacts)} artifacts)"
        )
        return "\n".join(lines)


@dataclass
class GcResult:
    dry_run: bool
    deleted_runs: list[str] = field(default_factory=list)
    deleted_artifacts: list[str] = field(default_factory=list)
    bytes_freed: int = 0
    bytes_reclaimable: int = 0
    errors: list[str] = field(default_factory=list)
    planned_runs: list[str] = field(default_factory=list)
    planned_artifacts: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.dry_run:
            text = (
                f"dry-run: would delete {len(self.planned_runs)} run(s), "
                f"{len(self.planned_artifacts)} artifact(s); "
                f"reclaimable {_format_bytes(self.bytes_reclaimable)}"
            )
        else:
            text = (
                f"deleted {len(self.deleted_runs)} run(s), "
                f"{len(self.deleted_artifacts)} artifact(s); "
                f"freed {_format_bytes(self.bytes_freed)}"
            )
        if self.errors:
            text += f"; errors: {len(self.errors)}"
        return text


class RetentionService:
    def __init__(
        self,
        run_store: RunStore,
        artifact_store: ArtifactStore | None = None,
        sweep_store: SweepStore | None = None,
    ) -> None:
        self.run_store = run_store
        self.artifact_store = artifact_store or run_store.artifact_store
        self.sweep_store = sweep_store

    def plan(self, policy: RetentionPolicy) -> GcPlan:
        policy.validate()
        runs = self.run_store.iter_runs()
        by_id = {run.run_id: run for run in runs}
        keep: dict[str, set[str]] = defaultdict(set)

        groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
        for run in runs:
            groups[(run.manifest.project, run.manifest.experiment)].append(run)

        for group in groups.values():
            group.sort(key=_sort_key)
            if policy.keep_last > 0:
                for run in group[-policy.keep_last :]:
                    keep[run.run_id].add("keep_last")
            if policy.keep_best > 0:
                scored = [
                    (value, run)
                    for run in group
                    if (value := _metric_value(run, policy.metric_key, policy.metric_mode)) is not None
                ]
                scored.sort(key=lambda item: item[0], reverse=policy.metric_mode == "max")
                for _, run in scored[: policy.keep_best]:
                    keep[run.run_id].add("keep_best")
            if policy.keep_every > 0:
                for index, run in enumerate(group):
                    if index % policy.keep_every == 0:
                        keep[run.run_id].add("keep_every")
            for run in group:
                for tag in run.manifest.tags:
                    if tag in policy.keep_tags:
                        keep[run.run_id].add(f"tag:{tag}")
                if run.manifest.status in policy.protect_status:
                    keep[run.run_id].add(f"status:{run.manifest.status}")
                age = _age_days(run.manifest.started_at)
                if age is None:
                    keep[run.run_id].add("unknown-age")
                elif age < policy.min_age_days:
                    keep[run.run_id].add("young")

        if policy.protect_sweep_runs and self.sweep_store is not None:
            for sweep in self.sweep_store.list():
                if policy.sweep_ttl_days is not None:
                    age = _age_days(sweep.created_at)
                    if age is not None and age > policy.sweep_ttl_days:
                        continue
                for result in sweep.results:
                    run_id = result.get("run_id") if isinstance(result, Mapping) else getattr(result, "run_id", None)
                    if run_id and run_id in by_id:
                        keep[run_id].add(f"sweep:{sweep.sweep_id}")

        # Lineage closure: a kept run keeps its parents.
        changed = True
        while changed:
            changed = False
            for run_id in list(keep):
                run = by_id.get(run_id)
                if run is None:
                    continue
                parent = run.manifest.parent_run_id
                if parent and parent in by_id and parent not in keep:
                    keep[parent].add(f"parent-of:{run_id}")
                    changed = True

        run_actions = tuple(
            RunAction(
                run_id=run.run_id,
                project=run.manifest.project,
                experiment=run.manifest.experiment,
                action="keep" if run.run_id in keep else "delete",
                reason=", ".join(sorted(keep[run.run_id])) if run.run_id in keep else "not retained",
                size_bytes=run.size_bytes(),
                path=str(run.dir),
                started_at=run.manifest.started_at,
            )
            for run in runs
        )

        kept_digests = {
            digest
            for run in runs
            if run.run_id in keep
            for digest in run.manifest.artifacts.values()
        }
        now = time.time()
        artifact_actions_list: list[ArtifactAction] = []
        for entry in self.artifact_store.list():
            if entry.digest in kept_digests:
                action, reason = "keep", "referenced"
            elif _age_days_from_mtime(entry.mtime, now) < policy.artifact_ttl_days:
                action, reason = "keep", "young"
            else:
                action, reason = "delete", "unreferenced"
            artifact_actions_list.append(
                ArtifactAction(
                    digest=entry.digest,
                    name=entry.name,
                    action=action,
                    reason=reason,
                    size_bytes=entry.size_bytes,
                    path=str(entry.path),
                )
            )

        return GcPlan(runs=run_actions, artifacts=tuple(artifact_actions_list))

    def apply(self, plan: GcPlan, *, dry_run: bool = True) -> GcResult:
        result = GcResult(
            dry_run=dry_run,
            bytes_reclaimable=plan.bytes_reclaimable,
            planned_runs=[action.run_id for action in plan.delete_runs],
            planned_artifacts=[f"{action.digest}/{action.name}" for action in plan.delete_artifacts],
        )
        if dry_run:
            return result

        for action in plan.delete_runs:
            try:
                shutil.rmtree(action.path)
                result.deleted_runs.append(action.run_id)
                result.bytes_freed += action.size_bytes
            except FileNotFoundError:
                continue
            except OSError as exc:
                result.errors.append(f"{action.run_id}: {exc}")

        for action in plan.delete_artifacts:
            try:
                if self.artifact_store.delete(action.digest, action.name):
                    result.deleted_artifacts.append(action.digest)
                    result.bytes_freed += action.size_bytes
            except OSError as exc:
                result.errors.append(f"{action.digest}/{action.name}: {exc}")

        return result

    def gc(self, policy: RetentionPolicy, *, dry_run: bool = True) -> GcResult:
        return self.apply(self.plan(policy), dry_run=dry_run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sort_key(run: Run) -> tuple[str, str]:
    return (run.manifest.started_at or "", run.run_id)


def _age_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds() / 86400.0


def _age_days_from_mtime(mtime: float, now: float) -> float:
    return (now - mtime) / 86400.0


def _metric_value(run: Run, key: str, mode: str) -> float | None:
    values: list[float] = []
    for path in run.eval_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics", {})
        if key in metrics:
            try:
                values.append(float(metrics[key]))
            except (TypeError, ValueError):
                continue
    if not values:
        return None
    return min(values) if mode == "min" else max(values)


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"
