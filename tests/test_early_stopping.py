from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dlef.config import ConfigError, resolve_config
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {
                    "seed": 1,
                    "epochs": 5,
                    "n_train": 32,
                    "n_val": 16,
                    "batch_size": 8,
                    "log_every": 0,
                    "early_stopping": {
                        "enabled": True,
                        "monitor": "mae",
                        "mode": "min",
                        "patience": 1,
                        "min_delta": 100.0,
                    },
                },
            )
            task = ToyTask(config.schema)
            train_data = ToyDataset(32, 8, 0.1, 1)
            val_data = ToyDataset(16, 8, 0.1, 2)

            run = runner.fit(task, train_data, val_data, config)

            self.assertIsNotNone(run)
            self.assertEqual(run.manifest.status, "completed")
            rows = [
                json.loads(line)
                for line in run.metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            val_rows = [row for row in rows if row.get("phase") == "val"]
            self.assertEqual(len(val_rows), 2)

            log = run.log_path.read_text(encoding="utf-8")
            self.assertIn("Early stopping: triggered", log)

            best = run.best_checkpoint_path
            self.assertTrue(best.is_file())
            best_payload = json.loads(best.read_text(encoding="utf-8"))
            last_payload = json.loads(run.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(best_payload["epoch"], 2)
            self.assertEqual(last_payload["epoch"], 2)
            self.assertIn("checkpoint:best", run.manifest.artifacts)
            self.assertIn("Checkpoint: best", log)

            task2 = ToyTask(config.schema)
            optimizer2 = task2.configure_optimizer(task2.get_model())
            runner.load_checkpoint(best, task2, optimizer2)
            self.assertEqual(task2.get_model().state_dict(), best_payload["model"])

    def test_best_checkpoint_does_not_require_early_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {
                    "seed": 1,
                    "epochs": 2,
                    "n_train": 32,
                    "n_val": 16,
                    "batch_size": 8,
                    "log_every": 0,
                    "selection": {"monitor": "mae", "mode": "min"},
                    "early_stopping": {"enabled": False, "monitor": "mae"},
                },
            )
            task = ToyTask(config.schema)
            run = runner.fit(
                task,
                ToyDataset(32, 8, 0.1, 1),
                ToyDataset(16, 8, 0.1, 2),
                config,
            )

            self.assertIsNotNone(run)
            self.assertTrue((run.directory / "checkpoints" / "best.json").is_file())
            self.assertTrue(run.checkpoint_path_for("best_raw").is_file())
            self.assertEqual(run.checkpoint_path_for("last"), run.checkpoint_path)
            self.assertIn("checkpoint:best", run.manifest.artifacts)
            self.assertNotIn("Early stopping: triggered", run.log_path.read_text(encoding="utf-8"))

    def test_ema_monitor_keeps_raw_and_ema_best_separate(self) -> None:
        """EMA 产物（best_ema/final_ema）与 raw best 分开保存；查询时 EMA 优先。"""
        class FakeEMA:
            """Duck-typed EMA: the runner only needs update/copy_to/state_dict."""

            def __init__(self, model, decay):
                self.decay = float(decay)
                self.shadow = dict(model.state_dict())
                self.updates = 0

            def update(self, model):
                self.updates += 1
                decay = self.decay * min(1.0, self.updates / 4.0)
                for key, value in model.state_dict().items():
                    self.shadow[key] = decay * self.shadow[key] + (1 - decay) * value

            def copy_to(self, model):
                model.load_state_dict(self.shadow)

            def state_dict(self):
                return dict(self.shadow)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store, ema_factory=FakeEMA)
            config = resolve_config(
                ToyConfig,
                {
                    "seed": 1,
                    "epochs": 4,
                    "n_train": 32,
                    "n_val": 16,
                    "batch_size": 8,
                    "log_every": 0,
                    "ema_decay": 0.5,
                    "selection": {
                        "monitor": "ema/mae",
                        "mode": "min",
                        "checkpoint": "best_ema",
                    },
                    "early_stopping": {
                        "enabled": True,
                        "monitor": "ema/mae",
                        "mode": "min",
                        "patience": 5,
                        "min_epochs": 99,
                        "smooth_window": 1,
                    },
                },
            )
            task = ToyTask(config.schema)
            run = runner.fit(
                task,
                ToyDataset(32, 8, 0.1, 1),
                ToyDataset(16, 8, 0.1, 2),
                config,
            )

            self.assertIsNotNone(run)
            checkpoints = run.directory / "checkpoints"
            best = checkpoints / "best.json"
            best_ema = checkpoints / "best_ema.json"
            self.assertTrue(best.is_file())
            self.assertTrue(best_ema.is_file())
            final_ema = checkpoints / "final_ema.json"
            self.assertTrue(final_ema.is_file())
            # D-0026：查询首选 checkpoint 时 EMA 优先（final_ema → best_ema → best）；
            # 原始权重仍保留在 best.json 里可查。
            self.assertEqual(run.best_checkpoint_path, final_ema)
            self.assertIn("checkpoint:best", run.manifest.artifacts)
            self.assertIn("checkpoint:best_ema", run.manifest.artifacts)

            log = run.log_path.read_text(encoding="utf-8")
            self.assertIn("Selection(raw):", log)
            self.assertIn("Selection(EMA):", log)

            best_payload = json.loads(best.read_text(encoding="utf-8"))
            ema_payload = json.loads(best_ema.read_text(encoding="utf-8"))
            self.assertNotEqual(best_payload["model"], ema_payload["model"])

    def test_raw_monitor_does_not_write_ema_best(self) -> None:
        class FakeEMA:
            def __init__(self, model, decay):
                self.shadow = dict(model.state_dict())

            def update(self, model):
                pass

            def copy_to(self, model):
                model.load_state_dict(self.shadow)

            def state_dict(self):
                return dict(self.shadow)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store, ema_factory=FakeEMA)
            config = resolve_config(
                ToyConfig,
                {
                    "seed": 1,
                    "epochs": 2,
                    "n_train": 32,
                    "n_val": 16,
                    "batch_size": 8,
                    "log_every": 0,
                    "ema_decay": 0.5,
                    "early_stopping": {
                        "enabled": True,
                        "monitor": "mae",
                        "mode": "min",
                        "patience": 5,
                        "min_epochs": 99,
                    },
                },
            )
            task = ToyTask(config.schema)
            run = runner.fit(
                task,
                ToyDataset(32, 8, 0.1, 1),
                ToyDataset(16, 8, 0.1, 2),
                config,
            )

            self.assertIsNotNone(run)
            checkpoints = run.directory / "checkpoints"
            self.assertTrue((checkpoints / "best.json").is_file())
            self.assertFalse((checkpoints / "best_ema.json").is_file())

    def test_invalid_config(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_config(
                ToyConfig,
                {"early_stopping": {"enabled": True, "monitor": "mae", "mode": "bad"}},
            )
        with self.assertRaises(ConfigError):
            resolve_config(
                ToyConfig,
                {"early_stopping": {"enabled": True, "monitor": "mae", "patience": 0}},
            )
        with self.assertRaises(ConfigError):
            resolve_config(
                ToyConfig,
                {"selection": {"checkpoint": "bad"}},
            )
        with self.assertRaises(ConfigError):
            resolve_config(
                ToyConfig,
                {"ema_decay": 0.0, "selection": {"monitor": "ema/mae"}},
            )
        with self.assertRaises(ConfigError):
            resolve_config(
                ToyConfig,
                {"ema_decay": 0.0, "selection": {"checkpoint": "best_ema"}},
            )


if __name__ == "__main__":
    unittest.main()
