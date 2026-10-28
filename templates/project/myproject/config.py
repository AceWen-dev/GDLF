from __future__ import annotations

from dataclasses import dataclass, field

from dlef.config import ConfigError, ConfigSchema, TrainConfig


@dataclass
class DataConfig(ConfigSchema):
    n_train: int = 512
    n_val: int = 128
    noise: float = 0.1
    batch_size: int = 32

    def validate(self) -> None:
        if self.n_train <= 0 or self.n_val <= 0:
            raise ConfigError("n_train/n_val must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if self.noise < 0:
            raise ConfigError("noise must be >= 0")


@dataclass
class ModelConfig(ConfigSchema):
    name: str = "mlp"
    hidden: int = 16

    def validate(self) -> None:
        if self.name not in {"mlp", "linear"}:
            raise ConfigError(f"unknown model.name: {self.name!r}")
        if self.hidden <= 0:
            raise ConfigError("hidden must be > 0")


@dataclass
class LossConfig(ConfigSchema):
    name: str = "mse"
    huber_beta: float = 1.0

    def validate(self) -> None:
        if self.name not in {"mse", "huber"}:
            raise ConfigError(f"unknown loss.name: {self.name!r}")
        if self.huber_beta <= 0:
            raise ConfigError("huber_beta must be > 0")


@dataclass
class OptimConfig(ConfigSchema):
    lr: float = 1e-2
    weight_decay: float = 0.0

    def validate(self) -> None:
        if self.lr <= 0:
            raise ConfigError("lr must be > 0")
        if self.weight_decay < 0:
            raise ConfigError("weight_decay must be >= 0")


@dataclass
class ProjectConfig(TrainConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    amp: bool = False
    grad_accum_steps: int = 1
    grad_clip: float | None = None

    def validate(self) -> None:
        super().validate()
        if self.grad_accum_steps < 1:
            raise ConfigError("grad_accum_steps must be >= 1")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ConfigError("grad_clip must be > 0 when set")
