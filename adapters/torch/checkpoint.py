from __future__ import annotations

from pathlib import Path

import torch

from dlef.contracts import Checkpoint


class TorchCheckpointCodec:
    """Binary checkpoint codec using torch.save/torch.load.

    The core never imports torch; the Runner accepts any codec with the same
    dump/load/suffix surface.
    """

    suffix = ".pt"

    def dump(self, path: str | Path, payload: Checkpoint) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    def load(self, path: str | Path) -> Checkpoint:
        try:
            return torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:  # older torch without weights_only
            return torch.load(Path(path), map_location="cpu")
