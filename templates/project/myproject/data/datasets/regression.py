from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ...config import DataConfig
from ..spec import DataBundle, make_spec
from ..splits import make_split, membership_ids
from ..transforms.base import Transform


class RegressionDataset(Dataset):
    """Synthetic 1-D regression samples, optionally transformed."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        transform: Transform | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        sample = {"x": self.x[index], "y": self.y[index]}
        return self.transform.fn(sample) if self.transform is not None else sample


def build_regression(
    config: DataConfig,
    role: str,
    seed: int,
    transform: Transform,
) -> DataBundle:
    indices = make_split(config.split, config.n_train, config.n_val, seed)
    if role not in indices:
        raise ValueError(
            f"unknown data role: {role!r}; available: {sorted(indices)}"
        )
    selected = indices[role]

    total = config.n_train + config.n_val
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(total, 1, generator=generator)
    y = 2.0 * x + 1.0 + config.noise * torch.randn(total, 1, generator=generator)

    dataset = RegressionDataset(x[selected], y[selected], transform)
    spec = make_spec(
        config,
        role,
        seed,
        transform.name,
        len(selected),
        membership_ids(config.split, indices),
    )
    return DataBundle(dataset=dataset, spec=spec)
