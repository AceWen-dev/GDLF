"""Sweep specification + local GPU-aware queue.

A sweep is a set of trials; one trial = one run. The core only schedules
trials and records results. What a trial builds and how it trains belongs to
the adapter (the ``trial_fn`` passed to ``LocalSweep.run``).

No domain words here: parameters are arbitrary keys, values are opaque.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

from .config import config_digest
from .run import utc_now


# ---------------------------------------------------------------------------
# Parameter spaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Choice.values must not be empty")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def grid_values(self) -> tuple[Any, ...]:
        return self.values

    def to_dict(self) -> dict[str, Any]:
        return {"type": "choice", "values": list(self.values)}


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("Uniform.low must be <= high")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def grid_values(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class LogUniform:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0 or self.low > self.high:
            raise ValueError("LogUniform requires 0 < low <= high")

    def sample(self, rng: random.Random) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))

    def grid_values(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "log_uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class IntUniform:
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("IntUniform.low must be <= high")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)

    def grid_values(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "int_uniform", "low": self.low, "high": self.high}


ParamSpace = Union[Choice, Uniform, LogUniform, IntUniform]


def param_space_from_dict(payload: Mapping[str, Any]) -> ParamSpace:
    kind = payload.get("type")
    if kind == "choice":
        return Choice(tuple(payload["values"]))
    if kind == "uniform":
        return Uniform(float(payload["low"]), float(payload["high"]))
    if kind == "log_uniform":
        return LogUniform(float(payload["low"]), float(payload["high"]))
    if kind == "int_uniform":
        return IntUniform(int(payload["low"]), int(payload["high"]))
    raise ValueError(f"unknown parameter space type: {kind!r}")


# ---------------------------------------------------------------------------
# Sweep / trial specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepSpec:
    name: str
    params: Mapping[str, ParamSpace]
    method: str = "grid"  # "grid" | "random"
    max_trials: int | None = None
    seed: int = 0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SweepSpec.name must not be empty")
        if not self.params:
            raise ValueError("SweepSpec.params must not be empty")
        if self.method not in {"grid", "random"}:
            raise ValueError("method must be 'grid' or 'random'")
        if self.method == "random" and not self.max_trials:
            raise ValueError("random sweeps require max_trials")
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError("max_trials must be >= 1 when set")
        if self.method == "grid":
            for key, space in self.params.items():
                if not isinstance(space, Choice):
                    raise ValueError(f"grid sweep parameter {key!r} must be a Choice")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "max_trials": self.max_trials,
            "seed": self.seed,
            "params": {key: value.to_dict() for key, value in self.params.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepSpec":
        params = {
            key: param_space_from_dict(value)
            for key, value in payload["params"].items()
        }
        return cls(
            name=str(payload["name"]),
            params=params,
            method=str(payload.get("method", "grid")),
            max_trials=payload.get("max_trials"),
            seed=int(payload.get("seed", 0)),
        )

    def digest(self) -> str:
        return config_digest(self.to_dict())

    def trials(self, sweep_id: str) -> list["TrialSpec"]:
        self.validate()
        values: list[dict[str, Any]] = []

        if self.method == "grid":
            keys = list(self.params)
            grids = [self.params[key].grid_values() for key in keys]
            for index, combination in enumerate(itertools.product(*grids)):
                if self.max_trials is not None and index >= self.max_trials:
                    break
                values.append(dict(zip(keys, combination)))
        else:
            rng = random.Random(self.seed)
            for _ in range(self.max_trials or 0):
                values.append({key: space.sample(rng) for key, space in self.params.items()})

        return [
            TrialSpec(
                trial_id=f"{sweep_id}-t{index:03d}",
                index=index,
                params=params,
            )
            for index, params in enumerate(values)
        ]


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    index: int
    params: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialSpec":
        return cls(
            trial_id=str(payload["trial_id"]),
            index=int(payload["index"]),
            params=dict(payload["params"]),
        )


@dataclass
class TrialContext:
    sweep_id: str
    trial_id: str
    index: int
    gpu_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialResult:
    trial_id: str
    status: str = "completed"  # completed | failed | killed
    run_id: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: str = ""
    ended_at: str = ""
    gpu_ids: tuple[int, ...] = ()
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialResult":
        return cls(**payload)


@dataclass(frozen=True)
class TrialCommand:
    """A command to run for one trial in a subprocess."""

    argv: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_s: float | None = None


@dataclass
class SweepRecord:
    sweep_id: str
    spec: Mapping[str, Any]
    status: str = "running"  # running | completed | partial | failed
    trials: list[Mapping[str, Any]] = field(default_factory=list)
    results: list[Mapping[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepRecord":
        return cls(**payload)


# ---------------------------------------------------------------------------
# Sweep store
# ---------------------------------------------------------------------------


class SweepStore:
    """Local sweep store: ``<root>/<sweep_id>/sweep.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, spec: SweepSpec) -> SweepRecord:
        spec.validate()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        sweep_id = f"{timestamp}_{spec.digest()[:8]}_{uuid.uuid4().hex[:6]}"
        record = SweepRecord(sweep_id=sweep_id, spec=spec.to_dict())
        self.save(record)
        return record

    def save(self, record: SweepRecord) -> Path:
        directory = self.root / record.sweep_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "sweep.json"
        path.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, sweep_id: str) -> SweepRecord:
        path = self.root / sweep_id / "sweep.json"
        return SweepRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[SweepRecord]:
        return [
            SweepRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.root.glob("*/sweep.json"))
        ]

    def delete(self, sweep_id: str) -> bool:
        """Remove one sweep directory; return False when it is already gone."""

        path = self.root / sweep_id
        if not path.is_dir():
            return False
        shutil.rmtree(path)
        return True


# ---------------------------------------------------------------------------
# Local GPU-aware queue
# ---------------------------------------------------------------------------


class GpuPool:
    """Thread-safe GPU slot allocator.

    ``acquire(n)`` blocks until ``n`` slots are free. The returned tuple is the
    set of GPU ids the trial should use.
    """

    def __init__(self, gpu_ids: Sequence[int] = (0,)) -> None:
        self._all = tuple(int(item) for item in gpu_ids)
        if not self._all:
            raise ValueError("gpu_ids must not be empty")
        if len(set(self._all)) != len(self._all):
            raise ValueError("gpu_ids must be unique")
        self._free = list(self._all)
        self._condition = threading.Condition()

    @property
    def all(self) -> tuple[int, ...]:
        return self._all

    def acquire(self, count: int = 1) -> tuple[int, ...]:
        if count < 1 or count > len(self._all):
            raise ValueError(f"cannot acquire {count} GPU(s) from {self._all}")
        with self._condition:
            while len(self._free) < count:
                self._condition.wait()
            assigned = tuple(self._free[:count])
            del self._free[:count]
            return assigned

    def release(self, gpu_ids: Sequence[int]) -> None:
        with self._condition:
            self._free.extend(int(item) for item in gpu_ids)
            self._free.sort()
            self._condition.notify_all()


class LocalQueue:
    """Thread-based queue for one machine.

    v0 uses threads so the interface is simple and testable. Real GPU training
    should set ``device`` from ``TrialContext.gpu_ids`` inside the trial_fn, or
    wrap each trial in a subprocess. The scheduler contract stays the same.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int] = (0,),
        *,
        max_parallel: int = 1,
        gpus_per_trial: int = 1,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if gpus_per_trial < 1:
            raise ValueError("gpus_per_trial must be >= 1")
        self.pool = GpuPool(gpu_ids)
        if gpus_per_trial > len(self.pool.all):
            raise ValueError("gpus_per_trial exceeds available gpu_ids")
        self.max_parallel = max_parallel
        self.gpus_per_trial = gpus_per_trial

    def run(
        self,
        sweep_id: str,
        trials: Sequence[TrialSpec],
        trial_fn: Callable[[TrialSpec, TrialContext], TrialResult],
    ) -> list[TrialResult]:
        results: list[TrialResult | None] = [None] * len(trials)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(self._run_one, sweep_id, trial, trial_fn): index
                for index, trial in enumerate(trials)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return [result for result in results if result is not None]

    def _run_one(
        self,
        sweep_id: str,
        trial: TrialSpec,
        trial_fn: Callable[[TrialSpec, TrialContext], TrialResult],
    ) -> TrialResult:
        gpu_ids = self.pool.acquire(self.gpus_per_trial)
        context = TrialContext(
            sweep_id=sweep_id,
            trial_id=trial.trial_id,
            index=trial.index,
            gpu_ids=gpu_ids,
        )
        started = utc_now()
        try:
            result = trial_fn(trial, context)
            if not isinstance(result, TrialResult):
                raise TypeError("trial_fn must return TrialResult")
            result.started_at = result.started_at or started
            result.ended_at = utc_now()
            result.gpu_ids = gpu_ids
            return result
        except BaseException as exc:  # noqa: BLE001 - isolate trial failures
            return TrialResult(
                trial_id=trial.trial_id,
                status="failed",
                error=repr(exc),
                started_at=started,
                ended_at=utc_now(),
                gpu_ids=gpu_ids,
            )
        finally:
            self.pool.release(gpu_ids)


class LocalSweep:
    """Create a sweep, generate trials, run them through ``LocalQueue``."""

    def __init__(
        self,
        store: SweepStore,
        *,
        gpu_ids: Sequence[int] = (0,),
        max_parallel: int = 1,
        gpus_per_trial: int = 1,
    ) -> None:
        self.store = store
        self.queue = LocalQueue(
            gpu_ids,
            max_parallel=max_parallel,
            gpus_per_trial=gpus_per_trial,
        )

    def run(
        self,
        spec: SweepSpec,
        trial_fn: Callable[[TrialSpec, TrialContext], TrialResult],
    ) -> SweepRecord:
        record = self.store.create(spec)
        trials = spec.trials(record.sweep_id)
        record.trials = [trial.to_dict() for trial in trials]
        self.store.save(record)

        try:
            results = self.queue.run(record.sweep_id, trials, trial_fn)
        except BaseException as exc:  # noqa: BLE001 - persist failure
            record.status = "failed"
            record.error = repr(exc)
            record.ended_at = utc_now()
            self.store.save(record)
            raise

        record.results = [result.to_dict() for result in results]
        record.status = (
            "completed" if all(result.status == "completed" for result in results) else "partial"
        )
        record.ended_at = utc_now()
        self.store.save(record)
        return record


class SubprocessQueue:
    """Run one trial per subprocess, with GPU slots and captured logs.

    The child receives:

    - ``CUDA_VISIBLE_DEVICES``: the GPU ids assigned to this trial;
    - ``DLEF_SWEEP_ID`` / ``DLEF_TRIAL_ID`` / ``DLEF_TRIAL_INDEX``;
    - ``DLEF_GPU_IDS``: comma-separated assigned ids;
    - ``DLEF_TRIAL_RESULT``: absolute path where the child should write a
      ``TrialResult`` JSON object.

    If the child writes no result file, exit code 0 means ``completed`` and a
    non-zero exit means ``failed``. A non-zero exit always marks the result as
    failed, even if the child wrote ``completed``.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int] = (0,),
        *,
        max_parallel: int = 1,
        gpus_per_trial: int = 1,
        timeout_s: float | None = None,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if gpus_per_trial < 1:
            raise ValueError("gpus_per_trial must be >= 1")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be > 0 when set")
        self.pool = GpuPool(gpu_ids)
        if gpus_per_trial > len(self.pool.all):
            raise ValueError("gpus_per_trial exceeds available gpu_ids")
        self.max_parallel = max_parallel
        self.gpus_per_trial = gpus_per_trial
        self.timeout_s = timeout_s

    def run(
        self,
        sweep_id: str,
        trials: Sequence[TrialSpec],
        command_fn: Callable[[TrialSpec, TrialContext], TrialCommand | Sequence[str]],
        log_dir: str | Path,
    ) -> list[TrialResult]:
        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        results: list[TrialResult | None] = [None] * len(trials)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(self._run_one, sweep_id, trial, command_fn, log_dir): index
                for index, trial in enumerate(trials)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return [result for result in results if result is not None]

    def _run_one(
        self,
        sweep_id: str,
        trial: TrialSpec,
        command_fn: Callable[[TrialSpec, TrialContext], TrialCommand | Sequence[str]],
        log_dir: Path,
    ) -> TrialResult:
        gpu_ids = self.pool.acquire(self.gpus_per_trial)
        context = TrialContext(
            sweep_id=sweep_id,
            trial_id=trial.trial_id,
            index=trial.index,
            gpu_ids=gpu_ids,
        )
        started = utc_now()
        log_path = log_dir / f"{trial.trial_id}.log"
        result_path = log_dir / f"{trial.trial_id}.result.json"
        effective_timeout = self.timeout_s

        try:
            command = command_fn(trial, context)
            if isinstance(command, TrialCommand):
                argv = list(command.argv)
                env_extra = command.env
                cwd = command.cwd
                timeout = command.timeout_s if command.timeout_s is not None else self.timeout_s
            else:
                argv = list(command)
                env_extra = {}
                cwd = None
                timeout = self.timeout_s
            if not argv:
                raise ValueError("trial command is empty")
            effective_timeout = timeout

            env = dict(os.environ)
            env.update({str(key): str(value) for key, value in env_extra.items()})
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpu_ids)
            env["DLEF_SWEEP_ID"] = sweep_id
            env["DLEF_TRIAL_ID"] = trial.trial_id
            env["DLEF_TRIAL_INDEX"] = str(trial.index)
            env["DLEF_GPU_IDS"] = ",".join(str(item) for item in gpu_ids)
            env["DLEF_TRIAL_RESULT"] = str(result_path)

            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                completed = subprocess.run(
                    argv,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            return _result_from_file(
                result_path,
                trial,
                gpu_ids,
                log_path,
                started,
                completed.returncode,
            )
        except subprocess.TimeoutExpired:
            return TrialResult(
                trial_id=trial.trial_id,
                status="failed",
                error=f"timeout after {effective_timeout}s",
                started_at=started,
                ended_at=utc_now(),
                gpu_ids=gpu_ids,
                log_path=str(log_path),
            )
        except BaseException as exc:  # noqa: BLE001 - isolate trial failures
            return TrialResult(
                trial_id=trial.trial_id,
                status="failed",
                error=repr(exc),
                started_at=started,
                ended_at=utc_now(),
                gpu_ids=gpu_ids,
                log_path=str(log_path),
            )
        finally:
            self.pool.release(gpu_ids)


class SubprocessSweep:
    """Create a sweep and run each trial in its own subprocess."""

    def __init__(
        self,
        store: SweepStore,
        *,
        gpu_ids: Sequence[int] = (0,),
        max_parallel: int = 1,
        gpus_per_trial: int = 1,
        timeout_s: float | None = None,
    ) -> None:
        self.store = store
        self.queue = SubprocessQueue(
            gpu_ids,
            max_parallel=max_parallel,
            gpus_per_trial=gpus_per_trial,
            timeout_s=timeout_s,
        )

    def run(
        self,
        spec: SweepSpec,
        command_fn: Callable[[TrialSpec, TrialContext], TrialCommand | Sequence[str]],
    ) -> SweepRecord:
        record = self.store.create(spec)
        trials = spec.trials(record.sweep_id)
        record.trials = [trial.to_dict() for trial in trials]
        self.store.save(record)
        log_dir = (self.store.root / record.sweep_id / "trials").resolve()

        try:
            results = self.queue.run(record.sweep_id, trials, command_fn, log_dir)
        except BaseException as exc:  # noqa: BLE001 - persist failure
            record.status = "failed"
            record.error = repr(exc)
            record.ended_at = utc_now()
            self.store.save(record)
            raise

        record.results = [result.to_dict() for result in results]
        record.status = (
            "completed" if all(result.status == "completed" for result in results) else "partial"
        )
        record.ended_at = utc_now()
        self.store.save(record)
        return record


def _result_from_file(
    path: Path,
    trial: TrialSpec,
    gpu_ids: tuple[int, ...],
    log_path: Path,
    started: str,
    returncode: int,
) -> TrialResult:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = TrialResult.from_dict(payload)
        except (OSError, ValueError, TypeError) as exc:
            return TrialResult(
                trial_id=trial.trial_id,
                status="failed",
                error=f"invalid trial result: {exc}",
                started_at=started,
                ended_at=utc_now(),
                gpu_ids=gpu_ids,
                log_path=str(log_path),
            )
        result.trial_id = trial.trial_id
        result.gpu_ids = gpu_ids
        result.log_path = str(log_path)
        result.started_at = result.started_at or started
        result.ended_at = utc_now()
        if returncode != 0:
            result.status = "failed"
            note = f"exit code {returncode}"
            result.error = f"{result.error}; {note}" if result.error else note
        elif result.status not in {"completed", "failed", "killed"}:
            result.status = "failed"
            result.error = f"invalid status: {result.status!r}"
        return result

    return TrialResult(
        trial_id=trial.trial_id,
        status="completed" if returncode == 0 else "failed",
        error=None if returncode == 0 else f"exit code {returncode}",
        started_at=started,
        ended_at=utc_now(),
        gpu_ids=gpu_ids,
        log_path=str(log_path),
    )
