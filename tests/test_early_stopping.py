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
