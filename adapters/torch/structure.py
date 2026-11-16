"""Structure fingerprint: make "which network actually ran" a recorded fact.

A config digest describes the *declared* choices; it cannot see the code. Two
runs can share a config digest and still execute different networks when a
model file was edited in place. This module closes that gap by hashing the
module that was actually built: layer names, module types, parameter shapes and
freeze state.

Uses:
- ``same config digest + different structure digest`` proves in-place drift;
- a run card can answer "what structure was this?" without reading source;
- a refactor can be proven structure-preserving by asserting digests are equal,
  which needs no training, only ``build_model``.

Domain-agnostic: it walks an ``nn.Module`` and knows nothing about the task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch import nn

from dlef.config import config_digest
from dlef.contracts import CallbackBase, RunContext

DEFAULT_FILENAME = "model_structure.json"


def structure_layers(module: nn.Module) -> list[dict[str, Any]]:
    """Record every submodule, including parameterless ones.

    Parameterless modules are kept on purpose: inserting a ``ReLU`` or removing
    a ``Dropout`` changes the network even though no parameter shape does.
    ``named_modules`` order is registration order, so the result is stable
    across processes for the same construction code.
    """

    layers: list[dict[str, Any]] = []
    for name, submodule in module.named_modules():
        params = list(submodule.parameters(recurse=False))
        entry: dict[str, Any] = {
            "name": name or "<root>",
            "type": type(submodule).__name__,
        }
        if params:
            entry["shapes"] = [list(param.shape) for param in params]
            entry["frozen"] = all(not param.requires_grad for param in params)
        layers.append(entry)
    return layers


def structure_fingerprint(module: nn.Module) -> dict[str, Any]:
    """Hash the built module's structure plus its parameter accounting.

    Two digests answer two different questions:

    - ``digest`` covers layer names, types, parameter shapes and freeze state.
      Any change at all, including an in-place edit, moves it.
    - ``topology_digest`` covers names and types only. It is stable across
      width/depth sweeps, so it identifies the architecture family while
      ``digest`` identifies the exact instance.

    ``n_trainable`` is part of the payload, not just a convenience: a freeze
    strategy changes what a run learns even when the layer list is identical,
    and it surfaces dead config such as a per-group learning rate whose group
    ended up empty.
    """

    layers = structure_layers(module)
    topology = [{"name": entry["name"], "type": entry["type"]} for entry in layers]
    params = list(module.parameters())
    payload: dict[str, Any] = {
        "layers": layers,
        "n_layers": len(layers),
        "n_params": sum(param.numel() for param in params),
        "n_trainable": sum(param.numel() for param in params if param.requires_grad),
        "topology_digest": config_digest({"layers": topology}),
    }
    payload["digest"] = config_digest(payload)
    return payload


class StructureCallback(CallbackBase):
    """Write the fingerprint into the run directory before the first step.

    Always on and deliberately without a config knob: it is an audit record
    like ``metrics.jsonl``, not a behavior-affecting choice, and it writes
    inside the run directory only.
    """

    def __init__(self, module: nn.Module, *, filename: str = DEFAULT_FILENAME) -> None:
        self.module = module
        self.filename = filename
        self.fingerprint: dict[str, Any] | None = None

    def on_run_start(self, ctx: RunContext) -> None:
        self.fingerprint = structure_fingerprint(self.module)
        if not ctx.run_dir:
            return
        path = Path(ctx.run_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.fingerprint, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def read_structure(run_dir: str | Path) -> dict[str, Any] | None:
    """Read a run's fingerprint, or None when the run predates this record."""

    path = Path(run_dir) / DEFAULT_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
