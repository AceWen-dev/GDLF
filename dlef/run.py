"""Run identity, local run store, and content-addressed artifacts.

The Run Store is the source of truth. A tracker, if any, is only an index.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_version(cwd: str | Path | None = None) -> str:
    """Return ``<short-sha>[-dirty]`` or ``no-git``."""

    cwd = Path(cwd or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not commit:
            return "no-git"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001 - code version must not block a run
        return "no-git"


def env_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


@dataclass
class RunManifest:
    run_id: str
    project: str
    experiment: str
    config_digest: str
    code_version: str
    env: Mapping[str, Any]
    seed: int
    data_versions: Mapping[str, str] = field(default_factory=dict)
    status: str = "running"
    artifacts: Mapping[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    parent_run_id: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    error: str | None = None
    config_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        return cls(**payload)


@dataclass(frozen=True)
class ArtifactEntry:
    digest: str
    name: str
    size_bytes: int
    mtime: float
    path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


class ArtifactStore:
    """Content-addressed file store.

    Layout: ``<root>/<sha[:2]>/<sha>/<name>``. Directories are not supported
    in v0; adapters can zip or tar before storing.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, path: str | Path, name: str | None = None) -> str:
        path = Path(path)
        if not path.is_file():
            raise IsADirectoryError(f"artifact must be a file in v0: {path}")
        digest = sha256_file(path)
        target_dir = self.root / digest[:2] / digest
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (name or path.name)
        if not target.exists():
            shutil.copy2(path, target)
        return digest

    def get(self, digest: str, name: str) -> Path:
        target = self.root / digest[:2] / digest / name
        if not target.is_file():
            raise FileNotFoundError(target)
        return target

    def exists(self, digest: str, name: str) -> bool:
        return (self.root / digest[:2] / digest / name).is_file()

    def list(self) -> list[ArtifactEntry]:
        entries: list[ArtifactEntry] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            digest = path.parent.name
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                continue
            stat = path.stat()
            entries.append(
                ArtifactEntry(
                    digest=digest,
                    name=path.name,
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                    path=path,
                )
            )
        return entries

    def delete(self, digest: str, name: str) -> bool:
        path = self.root / digest[:2] / digest / name
        if not path.is_file():
            return False
        path.unlink()
        for candidate in (path.parent, path.parent.parent):
            try:
                candidate.rmdir()
            except OSError:
                break
        return True


class Run:
    """A single immutable run directory."""

    def __init__(self, directory: Path, manifest: RunManifest, artifact_store: ArtifactStore) -> None:
        self.directory = Path(directory)
        self.manifest = manifest
        self.artifact_store = artifact_store
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return self.directory

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.jsonl"

    @property
    def config_path(self) -> Path:
        return self.directory / "config.resolved.json"

    @property
    def checkpoint_path(self) -> Path:
        candidates = sorted(
            (self.directory / "checkpoints").glob("last.*"),
            key=lambda path: path.stat().st_mtime,
        )
        return candidates[-1] if candidates else self.directory / "checkpoints" / "last.json"

    @property
    def log_path(self) -> Path:
        return self.directory / "logs" / "train.log"

    def log(self, message: str) -> None:
        """Append readable, timestamped lines to this run's training log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
        lines = str(message).splitlines() or [""]
        with self.log_path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{timestamp} - dlef - INFO - {line}\n")

    def write_manifest(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def log_metrics(self, record: Mapping[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")

    def save_json(self, relative_path: str, payload: Any) -> Path:
        if dataclasses.is_dataclass(payload):
            payload = dataclasses.asdict(payload)
        path = self.directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def save_artifact(self, path: str | Path, name: str) -> str:
        digest = self.artifact_store.put(path, name)
        artifacts = dict(self.manifest.artifacts)
        artifacts[name] = digest
        self.manifest.artifacts = artifacts
        self.write_manifest()
        return digest

    def finalize(self, status: str = "completed", *, error: str | None = None) -> None:
        if status not in {"completed", "failed", "killed"}:
            raise ValueError(f"invalid final status: {status}")
        self.manifest.status = status
        self.manifest.ended_at = utc_now()
        self.manifest.error = error
        self.write_manifest()

    def size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def eval_paths(self) -> list[Path]:
        return sorted((self.directory / "evals").glob("*.json"))

    def __repr__(self) -> str:
        return f"Run({self.run_id!r}, status={self.manifest.status!r})"


class RunStore:
    """Local-first run store: ``<root>/<project>/<experiment>/<run_id>/``."""

    MANIFEST = "manifest.json"

    def __init__(
        self,
        root: str | Path,
        artifact_store: ArtifactStore | None = None,
        code_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_store = artifact_store or ArtifactStore(self.root.parent / "artifacts")
        self.code_root = Path(code_root) if code_root is not None else Path.cwd()

    def create(
        self,
        *,
        project: str,
        experiment: str,
        config_digest: str,
        seed: int,
        data_versions: Mapping[str, str] | None = None,
        tags: Mapping[str, str] | None = None,
        parent_run_id: str | None = None,
        config_source: str = "",
    ) -> Run:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}_{config_digest[:8]}_s{seed}_{uuid.uuid4().hex[:6]}"
        directory = self.root / _safe(project) / _safe(experiment) / run_id
        manifest = RunManifest(
            run_id=run_id,
            project=project,
            experiment=experiment,
            config_digest=config_digest,
            code_version=git_version(self.code_root),
            env=env_snapshot(),
            seed=seed,
            data_versions=dict(data_versions or {}),
            tags=dict(tags or {}),
            parent_run_id=parent_run_id,
            config_source=config_source,
        )
        run = Run(directory, manifest, self.artifact_store)
        run.write_manifest()
        return run

    def open(self, path: str | Path) -> Run:
        directory = Path(path)
        if directory.name == self.MANIFEST:
            directory = directory.parent
        manifest_path = directory / self.MANIFEST
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return Run(directory, RunManifest.from_dict(payload), self.artifact_store)

    def iter_runs(self) -> list[Run]:
        return [self.open(path) for path in sorted(self.root.rglob(self.MANIFEST))]

    def iter_evals(self) -> list[tuple[Run, Any]]:
        from .eval import EvalRecord

        results: list[tuple[Run, EvalRecord]] = []
        for run in self.iter_runs():
            for path in run.eval_paths():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    record = EvalRecord.from_dict(payload)
                except (OSError, ValueError, TypeError):
                    continue
                results.append((run, record))
        return results

    def find_eval(self, eval_id: str) -> tuple[Run, Any] | None:
        for run, record in self.iter_evals():
            if record.eval_id == eval_id:
                return run, record
        return None

    def list(
        self,
        *,
        project: str | None = None,
        experiment: str | None = None,
    ) -> list[RunManifest]:
        manifests: list[RunManifest] = []
        for run in self.iter_runs():
            manifest = run.manifest
            if project is not None and manifest.project != project:
                continue
            if experiment is not None and manifest.experiment != experiment:
                continue
            manifests.append(manifest)
        return manifests


def _safe(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid path segment: {name!r}")
    return name
