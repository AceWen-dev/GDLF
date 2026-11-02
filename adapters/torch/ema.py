"""Model EMA: 权重的指数滑动平均。

为什么需要：小样本训练时验证指标逐轮抖动很大，直接拿某一轮的原始权重去选模，
等于在一串噪声里挑最小值。对权重做 EMA 相当于把最近若干个更新步平均起来，
通常比任何单轮权重都稳，也是本项目 PolypMeasure 侧既有最优配方的一部分。

约定：
- 浮点参数按 ``decay`` 滑动平均，整数 buffer（如 BN 的 num_batches_tracked）直接复制；
- ``copy_to`` 把 EMA 权重写回模型，用于验证/保存；
- 状态可以随 checkpoint 存取，恢复训练时不会从零重新累积。

早期偏差：shadow 从**初始权重**起步，直接按 0.999 做平均意味着前
``1/(1-decay)`` 步里始终混着大量初始权重（随机初始化的 head 尤其有害）。
因此在 ``warmup_steps``（默认取 EMA 视野 ``1/(1-decay)``）内把 decay 从 0
线性升到配置值，先快速跟住当前权重，再转入正常平滑。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


class ModelEMA:
    """Exponential moving average of a model's ``state_dict``."""

    def __init__(
        self, model: Any, decay: float, warmup_steps: int | None = None
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        # 默认 warmup 覆盖一个 EMA 视野：decay=0.999 -> 1000 步。
        if warmup_steps is None:
            warmup_steps = max(1, int(round(1.0 / (1.0 - self.decay)))) if self.decay > 0 else 0
        if warmup_steps < 0:
            raise ValueError(f"ema warmup_steps must be >= 0, got {warmup_steps}")
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self._keys = [
            key for key, value in self.shadow.items() if value.is_floating_point()
        ]

    @torch.no_grad()
    def update(self, model: Any) -> None:
        """Blend the model's current weights into the shadow copy."""
        self.num_updates += 1
        state = model.state_dict()
        decay = self.decay
        if self.warmup_steps > 0:
            decay *= min(1.0, self.num_updates / self.warmup_steps)
        shadows = [self.shadow[key] for key in self._keys]
        values = [state[key].detach() for key in self._keys]
        # foreach 版本把几百个张量的更新合成两批 kernel，避免每步几百次启动开销。
        torch._foreach_mul_(shadows, decay)
        torch._foreach_add_(shadows, values, alpha=1.0 - decay)
        for key, value in state.items():
            if key not in self.shadow:
                continue
            if key not in self._keys:
                self.shadow[key].copy_(value)

    @torch.no_grad()
    def copy_to(self, model: Any) -> None:
        """Load the averaged weights into ``model`` (in place)."""
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        missing = [key for key in self.shadow if key not in state]
        if missing:
            raise KeyError(f"ema state is missing {len(missing)} tensors")
        with torch.no_grad():
            for key in self.shadow:
                self.shadow[key].copy_(state[key])
