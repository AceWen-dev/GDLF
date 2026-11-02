from __future__ import annotations

import unittest

try:
    import torch

    from adapters.torch.ema import ModelEMA

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _model():
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(3))

    return M()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ModelEMATests(unittest.TestCase):
    def test_default_warmup_covers_one_ema_horizon(self) -> None:
        ema = ModelEMA(_model(), decay=0.999)
        self.assertEqual(ema.warmup_steps, 1000)
        ema_zero = ModelEMA(_model(), decay=0.0)
        self.assertEqual(ema_zero.warmup_steps, 0)

    def test_warmup_tracks_current_weights_early(self) -> None:
        model = _model()
        ema = ModelEMA(model, decay=0.999)
        with torch.no_grad():
            model.w.fill_(10.0)
        ema.update(model)
        # 第 1 步 decay≈0.001，shadow 应几乎等于当前权重，而不是初始全零。
        self.assertTrue(
            torch.allclose(ema.shadow["w"], torch.full((3,), 10.0), atol=0.02)
        )

    def test_without_warmup_initial_weights_dominate(self) -> None:
        model = _model()
        ema = ModelEMA(model, decay=0.999, warmup_steps=0)
        with torch.no_grad():
            model.w.fill_(10.0)
        ema.update(model)
        # 旧行为：初始权重占 99.9%，这也正是早期 EMA 指标失真的来源。
        self.assertTrue(
            torch.allclose(ema.shadow["w"], torch.full((3,), 0.01), atol=1e-6)
        )

    def test_decay_reaches_configured_value_after_warmup(self) -> None:
        model = _model()
        ema = ModelEMA(model, decay=0.5, warmup_steps=2)
        with torch.no_grad():
            model.w.fill_(1.0)
        ema.update(model)  # decay = 0.25
        self.assertTrue(
            torch.allclose(ema.shadow["w"], torch.full((3,), 0.75), atol=1e-6)
        )
        ema.update(model)  # decay = 0.5
        self.assertTrue(
            torch.allclose(ema.shadow["w"], torch.full((3,), 0.875), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
