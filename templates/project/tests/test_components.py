from __future__ import annotations

import unittest

try:
    import torch

    from adapters.torch import TorchModel, TorchOptimizer
    from dlef.contracts import ModelOutput
    from myproject.config import (
        DataConfig,
        LossConfig,
        ModelConfig,
        OptimConfig,
        SchedulerConfig,
    )
    from myproject.data import build_dataset, build_transform, make_loaders
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
    def test_data(self) -> None:
        config = DataConfig(n_train=8, n_val=4, batch_size=2)
        train = build_dataset(config, "train", 7, "identity")
        val = build_dataset(config, "val", 7, "identity")

        self.assertEqual(len(train.dataset), 8)
        self.assertEqual(len(val.dataset), 4)
        self.assertEqual(train.spec.splits, val.spec.splits)
        self.assertNotEqual(train.spec.version, val.spec.version)
        self.assertEqual(train.spec.transform, "identity")

        again = build_dataset(config, "train", 7, "identity")
        self.assertEqual(train.spec.version, again.spec.version)
        self.assertTrue(torch.equal(train.dataset.x, again.dataset.x))

        other_seed = build_dataset(config, "train", 8, "identity")
        self.assertNotEqual(train.spec.version, other_seed.spec.version)

        other_version = build_dataset(
            DataConfig(n_train=8, n_val=4, version="v2"),
            "train",
            7,
            "identity",
        )
        self.assertNotEqual(train.spec.version, other_version.spec.version)

        train_loader, _ = make_loaders(config, seed=7)
        batch = next(iter(train_loader))
        self.assertEqual(tuple(batch["x"].shape), (2, 1))

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
            build_dataset(DataConfig(name="nope"), "train", 7, "identity")
        with self.assertRaises(ValueError):
            build_dataset(DataConfig(split="nope"), "train", 7, "identity")
        with self.assertRaises(ValueError):
            build_dataset(DataConfig(), "test", 7, "identity")
        with self.assertRaises(ValueError):
            build_transform("nope")
        with self.assertRaises(ValueError):
            build_model(ModelConfig(name="nope"), device="cpu")
        with self.assertRaises(ValueError):
            build_loss(LossConfig(name="nope"))


if __name__ == "__main__":
    unittest.main()
