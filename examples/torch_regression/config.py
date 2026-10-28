from __future__ import annotations

from dataclasses import dataclass

from dlef.config import ConfigError, TrainConfig


@dataclass
class TorchConfig(TrainConfig):
    lr: float = 1e-2
    batch_size: int = 32
    n_train: int = 512
    n_val: int = 128
    noise: float = 0.1
    hidden: int = 16
    amp: bool = False
    grad_accum_steps: int = 1
    grad_clip: float | None = None

    def validate(self) -> None:
        super().validate()
        if self.lr <= 0:
            raise ConfigError("lr must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if self.n_train <= 0 or self.n_val <= 0:
            raise ConfigError("n_train/n_val must be > 0")
        if self.hidden <= 0:
            raise ConfigError("hidden must be > 0")
        if self.grad_accum_steps < 1:
            raise ConfigError("grad_accum_steps must be >= 1")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ConfigError("grad_clip must be > 0 when set")
