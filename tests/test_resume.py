from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.seed import capture_rng_state, restore_rng_state
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class ResumeTests(unittest.TestCase):
    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {"seed": 5, "epochs": 3, "lr": 0.05, "n_train": 64, "n_val": 32, "batch_size": 8},
            )
            train_data = ToyDataset(64, 8, 0.1, 5)
            val_data = ToyDataset(32, 8, 0.1, 6)
            task = ToyTask(config.schema)
            run = runner.fit(task, train_data, val_data, config)
            payload = json.loads(run.checkpoint_path.read_text(encoding="utf-8"))

            task2 = ToyTask(config.schema)
            optimizer2 = task2.configure_optimizer(task2.get_model())
            runner.load_checkpoint(run.checkpoint_path, task2, optimizer2)

            self.assertEqual(task2.get_model().state_dict(), payload["model"])
            self.assertEqual(optimizer2.state_dict()["lr"], payload["optimizer"]["lr"])
            rng_state = payload["rng_state"]
            self.assertIn("python", rng_state)
            self.assertEqual(list(random.getstate()[1]), rng_state["python"][1])

            try:
                import numpy as np
            except ImportError:
                pass
            else:
                self.assertIn("numpy", rng_state)
                self.assertEqual(
                    np.random.get_state()[1].tolist(), rng_state["numpy"]["keys"]
                )

            try:
                import torch
            except ImportError:
                pass
            else:
                self.assertIn("torch", rng_state)
                self.assertEqual(torch.get_rng_state().tolist(), rng_state["torch"])

    def test_legacy_python_rng_state(self) -> None:
        random.seed(99)
        legacy = random.getstate()
        restore_rng_state(legacy)
        self.assertEqual(list(random.getstate()[1]), list(legacy[1]))

    def test_capture_restore_python_rng(self) -> None:
        random.seed(123)
        state = capture_rng_state()
        expected = [random.random() for _ in range(3)]
        random.random()
        restore_rng_state(state)
        self.assertEqual([random.random() for _ in range(3)], expected)


if __name__ == "__main__":
    unittest.main()
