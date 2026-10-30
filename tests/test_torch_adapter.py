from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.contracts import LossOutput, ModelOutput
from dlef.eval import EvalSpec
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner

try:
    import torch

    from adapters.torch import (
        TorchCheckpointCodec,
        TorchLoss,
        TorchMetric,
        TorchModel,
        TorchOptimizer,
        TorchTask,
        TorchTrainingPolicy,
    )
    from examples.torch_regression.config import TorchConfig
    from examples.torch_regression.data import make_loaders
    from examples.torch_regression.task import build_task

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchAdapterTests(unittest.TestCase):
    def test_components(self) -> None:
        module = torch.nn.Linear(1, 1)
        model = TorchModel(module, lambda module, batch: module(batch["x"]), device="cpu")
        batch = {"x": torch.randn(4, 1), "y": torch.randn(4, 1)}
        output = model(batch)
        self.assertEqual(tuple(output.predictions.shape), (4, 1))

        aux_model = TorchModel(
            module,
            lambda module, batch: ModelOutput(
                predictions=module(batch["x"]),
                aux={"features": batch["x"]},
            ),
            device="cpu",
        )
        aux_output = aux_model(batch)
        self.assertIn("features", aux_output.aux)
        self.assertTrue(torch.equal(aux_output.aux["features"], batch["x"]))

        task = TorchTask(
            model=aux_model,
            loss=TorchLoss(
                lambda out, b: LossOutput(total=out.predictions.sum() * 0.0)
            ),
            optimizer_fn=lambda model: TorchOptimizer(
                torch.optim.SGD(model.parameters(), lr=0.1)
            ),
            metrics={},
            device="cpu",
        )
        step = task.val_step(batch)
        self.assertIn("features", step.meta)

        loss = TorchLoss(
            lambda out, b: LossOutput(total=((out.predictions - b["y"]) ** 2).mean())
        )(output, batch)
        self.assertTrue(torch.isfinite(loss.total))

        optimizer = TorchOptimizer(torch.optim.SGD(model.parameters(), lr=0.1))
        optimizer.zero_grad()
        loss.total.backward()
        optimizer.step()

        metric = TorchMetric(
            lambda preds, targets: {"mae": (preds - targets).abs().mean()}
        )
        metric.update(output, batch)
        self.assertIn("mae", metric.compute().values)

    def test_end_to_end_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(
                store,
                store.artifact_store,
                checkpoint_codec=TorchCheckpointCodec(),
            )
            config = resolve_config(
                TorchConfig,
                {
                    "seed": 5,
                    "epochs": 20,
                    "lr": 0.02,
                    "batch_size": 16,
                    "n_train": 128,
                    "n_val": 64,
                    "noise": 0.1,
                    "hidden": 16,
                    "project": "torch-test",
                    "experiment": "run",
                    "log_every": 5,
                },
            )
            train_loader, val_loader = make_loaders(config.schema)
            task = build_task(config.schema, device="cpu")
            run = runner.fit(task, train_loader, val_loader, config)

            self.assertEqual(run.manifest.status, "completed")
            self.assertTrue(run.checkpoint_path.is_file())
            self.assertEqual(run.checkpoint_path.suffix, ".pt")
            self.assertIn("checkpoint:last", run.manifest.artifacts)

            record = runner.evaluate(
                task,
                val_loader,
                EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
                run,
            )
            self.assertIn("mae", record.metrics)
            self.assertIsInstance(record.metrics["mae"], float)

            losses = [
                json.loads(line)["loss"]
                for line in run.metrics_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("phase") == "train"
            ]
            self.assertTrue(losses)
            self.assertLess(losses[-1], losses[0])

            task2 = build_task(config.schema, device="cpu")
            optimizer2 = task2.configure_optimizer(task2.get_model())
            runner.load_checkpoint(run.checkpoint_path, task2, optimizer2)
            for (name1, value1), (name2, value2) in zip(
                task.get_model().state_dict().items(),
                task2.get_model().state_dict().items(),
            ):
                self.assertEqual(name1, name2)
                self.assertTrue(torch.equal(value1.cpu(), value2.cpu()))

    def test_policy_accumulation_clip(self) -> None:
        policy = TorchTrainingPolicy(
            amp=False,
            grad_accum_steps=2,
            grad_clip=1.0,
        )
        self.assertTrue(policy.is_main_process)
        self.assertTrue(policy.should_step(2))
        self.assertFalse(policy.should_step(3))
        self.assertTrue(policy.should_flush(3))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(
                store,
                store.artifact_store,
                checkpoint_codec=TorchCheckpointCodec(),
                training_policy=policy,
            )
            config = resolve_config(
                TorchConfig,
                {
                    "seed": 2,
                    "epochs": 3,
                    "lr": 0.02,
                    "batch_size": 16,
                    "n_train": 64,
                    "n_val": 32,
                    "amp": False,
                    "grad_accum_steps": 2,
                    "grad_clip": 1.0,
                    "project": "torch-policy",
                    "experiment": "accum",
                    "log_every": 0,
                },
            )
            train_loader, val_loader = make_loaders(config.schema)
            task = build_task(config.schema, device="cpu")
            run = runner.fit(task, train_loader, val_loader, config)
            self.assertIsNotNone(run)
            self.assertEqual(run.manifest.status, "completed")
            record = runner.evaluate(
                task,
                val_loader,
                EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
                run,
            )
            self.assertTrue(math.isfinite(record.metrics["mae"]))
            self.assertEqual(policy.state_dict()["grad_accum_steps"], 2)

    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "cuda not available")
    def test_amp_cuda(self) -> None:
        policy = TorchTrainingPolicy(amp=True, grad_accum_steps=1, grad_clip=1.0)
        policy.setup()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(
                store,
                store.artifact_store,
                checkpoint_codec=TorchCheckpointCodec(),
                training_policy=policy,
            )
            config = resolve_config(
                TorchConfig,
                {
                    "seed": 3,
                    "epochs": 2,
                    "lr": 0.02,
                    "batch_size": 16,
                    "n_train": 64,
                    "n_val": 32,
                    "amp": True,
                    "grad_accum_steps": 1,
                    "grad_clip": 1.0,
                    "project": "torch-policy",
                    "experiment": "amp",
                    "log_every": 0,
                },
            )
            train_loader, val_loader = make_loaders(config.schema)
            task = build_task(config.schema, device="cuda")
            run = runner.fit(task, train_loader, val_loader, config)
            self.assertIsNotNone(run)
            self.assertTrue(policy.amp)
            self.assertIsNotNone(policy.state_dict()["scaler"])
            record = runner.evaluate(
                task,
                val_loader,
                EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
                run,
            )
            self.assertTrue(math.isfinite(record.metrics["mae"]))


if __name__ == "__main__":
    unittest.main()
