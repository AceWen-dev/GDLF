from __future__ import annotations

from dataclasses import dataclass

from dlef.config import ConfigError, TrainConfig


@dataclass
class ToyConfig(TrainConfig):
    lr: float = 0.1
    batch_size: int = 16
    n_train: int = 512
    n_val: int = 128
    noise: float = 0.1
    w_true: float = 2.0
    b_true: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.lr <= 0:
            raise ConfigError("lr must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if self.n_train <= 0 or self.n_val <= 0:
            raise ConfigError("n_train/n_val must be > 0")
