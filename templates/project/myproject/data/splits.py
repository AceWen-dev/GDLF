"""Materialized split strategies.

A split is an explicit mapping ``role -> list[int]``. It is created once and
recorded in ``DataSpec.splits``; loaders never call ``random_split`` at runtime.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import torch

SplitIndices = dict[str, list[int]]


def holdout(n_train: int, n_val: int, seed: int) -> SplitIndices:
    total = n_train + n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator).tolist()
    return {
        "train": perm[:n_train],
        "val": perm[n_train:],
    }


SPLITTERS: dict[str, Callable[[int, int, int], SplitIndices]] = {
    "holdout": holdout,
}


def make_split(name: str, n_train: int, n_val: int, seed: int) -> SplitIndices:
    try:
        splitter = SPLITTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown data.split: {name!r}; available: {sorted(SPLITTERS)}"
        ) from exc
    return splitter(n_train, n_val, seed)


def membership_ids(split_name: str, indices: SplitIndices) -> dict[str, str]:
    payload = json.dumps(indices, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return {role: f"{split_name}@{digest}/{role}" for role in indices}
