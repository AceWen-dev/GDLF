"""Placeholder data pipeline: synthetic 1-D regression.

Replace ``RegressionDataset`` / ``make_loaders`` with your real data. Keep the
two outputs: a train batch iterable and a val batch iterable, both carrying a
``DataSpec``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from adapters.torch import TorchDataLoader
from dlef.contracts import DataSpec

from .config import DataConfig


class RegressionDataset(Dataset):
    def __init__(self, n: int, noise: float, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 1, generator=generator)
        self.y = 2.0 * self.x + 1.0 + noise * torch.randn(n, 1, generator=generator)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return {"x": self.x[index], "y": self.y[index]}


def _spec(split: str, n: int, seed: int) -> DataSpec:
    return DataSpec(
        name="synthetic-regression",
        version=f"{split}-n{n}-seed{seed}",
        schema={"x": "float32[n,1]", "y": "float32[n,1]"},
        splits={split: "generated"},
        transform="none",
    )


def make_loaders(config: DataConfig, seed: int, *, distributed: bool = False):
    train_dataset = RegressionDataset(config.n_train, config.noise, seed)
    val_dataset = RegressionDataset(config.n_val, config.noise, seed + 1)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return (
        TorchDataLoader(train_loader, _spec("train", config.n_train, seed)),
        TorchDataLoader(val_loader, _spec("val", config.n_val, seed + 1)),
    )
