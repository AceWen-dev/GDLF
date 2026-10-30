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
            self.assertEqual(best_payload["epoch"], 1)
            self.assertEqual(last_payload["epoch"], 2)
            self.assertIn("checkpoint:best", run.manifest.artifacts)
            self.assertIn("Checkpoint: best", log)

            task2 = ToyTask(config.schema)
            optimizer2 = task2.configure_optimizer(task2.get_model())
            runner.load_checkpoint(best, task2, optimizer2)
            self.assertEqual(task2.get_model().state_dict(), best_payload["model"])

    def test_no_best_checkpoint_without_early_stopping(self) -> None:
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
            self.assertFalse(run.best_checkpoint_path.is_file())
            self.assertNotIn("checkpoint:best", run.manifest.artifacts)

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


if __name__ == "__main__":
    unittest.main()
