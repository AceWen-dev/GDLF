"""Checkpoint codecs: the core owns the envelope, the adapter owns the bytes.

A codec only needs to dump/load a ``Checkpoint``. The default is JSON; a
tensor-library adapter can supply ``torch.save``/``torch.load`` without the
core importing that library.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .contracts import Checkpoint


class CheckpointCodec(Protocol):
    suffix: str

    def dump(self, path: str | Path, payload: Checkpoint) -> None: ...

    def load(self, path: str | Path) -> Checkpoint: ...


class JsonCheckpointCodec:
    suffix = ".json"

    def dump(self, path: str | Path, payload: Checkpoint) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> Checkpoint:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return Checkpoint(**payload)
