from __future__ import annotations

import unittest

try:
    import torch

    from adapters.torch import TorchModel, TorchOptimizer
    from dlef.contracts import ModelOutput
    from myproject.config import LossConfig, ModelConfig, OptimConfig, SchedulerConfig
    from myproject.losses import build_loss
    from myproject.metrics import build_metrics
    from myproject.models import build_model
    from myproject.optimizers import build_optimizer
    from myproject.schedulers import build_scheduler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ComponentTests(unittest.TestCase):
    def test_models(self) -> None:
        batch = {"x": torch.randn(4, 1)}
        for name in ("mlp", "linear"):
            model = build_model(ModelConfig(name=name, hidden=8), device="cpu")
            self.assertIsInstance(model, TorchModel)
            output = model(batch)
            self.assertEqual(tuple(output.predictions.shape), (4, 1))

    def test_losses(self) -> None:
        output = ModelOutput(predictions=torch.zeros(2, 1))
        batch = {"y": torch.ones(2, 1)}
        mse = build_loss(LossConfig(name="mse"))(output, batch)
        self.assertAlmostEqual(float(mse.total), 1.0, places=6)
        self.assertIn("mse", mse.terms)

        huber = build_loss(LossConfig(name="huber", huber_beta=1.0))(output, batch)
        self.assertTrue(torch.isfinite(huber.total))
        self.assertIn("huber", huber.terms)

    def test_metrics(self) -> None:
        metric = build_metrics(["mae"])["mae"]
        metric.update(
            ModelOutput(predictions=torch.tensor([1.0, 2.0])),
            {"y": torch.tensor([2.0, 4.0])},
        )
        self.assertAlmostEqual(metric.compute().values["mae"], 1.5)

    def test_optimizers_and_schedulers(self) -> None:
        model = build_model(ModelConfig(name="linear"), device="cpu")
        for name in ("adamw", "adam", "sgd"):
            optimizer = build_optimizer(OptimConfig(name=name, lr=0.01), model)
            self.assertIsInstance(optimizer, TorchOptimizer)

        optimizer = build_optimizer(OptimConfig(name="adamw", lr=0.01), model)
        cosine = build_scheduler(SchedulerConfig(name="cosine"), optimizer, epochs=3)
        self.assertIsNotNone(cosine)
        constant = build_scheduler(SchedulerConfig(name="constant"), optimizer, epochs=3)
        self.assertIsNone(constant)

    def test_unknown_names_raise(self) -> None:
        with self.assertRaises(ValueError):
            build_model(ModelConfig(name="nope"), device="cpu")
        with self.assertRaises(ValueError):
            build_loss(LossConfig(name="nope"))


if __name__ == "__main__":
    unittest.main()
