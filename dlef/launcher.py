"""Minimal local launcher: same semantics as remote launchers, synchronous."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Launcher(Protocol):
    def submit(self, command: Sequence[str]) -> str: ...

    def status(self, job_id: str) -> Mapping[str, Any]: ...

    def cancel(self, job_id: str) -> None: ...


class LocalLauncher:
    """Runs a command in-process. Useful for v0 and for tests."""

    def submit(self, command: Sequence[str]) -> str:
        subprocess.run(list(command), check=True)
        return "local"

    def status(self, job_id: str) -> Mapping[str, Any]:
        return {"id": job_id, "status": "completed"}

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError("LocalLauncher runs synchronously")
