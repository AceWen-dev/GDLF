from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from dlef.seed import capture_rng_state, restore_rng_state, seed_everything
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class RecordingTask(ToyTask):
    def __init__(self, config: ToyConfig) -> None:
        super().__init__(config)
        self.python_draws: list[float] = []

    def train_step(self, batch):
        self.python_draws.append(random.random())
        return super().train_step(batch)


class SeedTests(unittest.TestCase):
    def test_seed_everything_reproducible(self) -> None:
        seed_everything(11)
        expected = [random.random()]

        try:
            import numpy as np
        except ImportError:
            np = None
        else:
            expected.append(float(np.random.rand()))

        try:
            import torch
        except ImportError:
            torch = None
        else:
            expected.append(float(torch.rand(1)))

        seed_everything(11)
        actual = [random.random()]
        if np is not None:
            actual.append(float(np.random.rand()))
        if torch is not None:
            actual.append(float(torch.rand(1)))

        self.assertEqual(actual, expected)

    def test_capture_restore_all_rngs(self) -> None:
        seed_everything(42)
        state = capture_rng_state()
        expected = [random.random()]

        try:
            import numpy as np
        except ImportError:
            np = None
        else:
            expected.append(float(np.random.rand()))

        try:
            import torch
        except ImportError:
            torch = None
        else:
            expected.append(float(torch.rand(1)))

        seed_everything(7)
        restore_rng_state(state)

        actual = [random.random()]
        if np is not None:
            actual.append(float(np.random.rand()))
        if torch is not None:
            actual.append(float(torch.rand(1)))

        self.assertEqual(actual, expected)

    def test_runner_seeds_training_rng(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {
                    "seed": 21,
                    "epochs": 1,
                    "max_steps": 1,
                    "n_train": 16,
                    "n_val": 8,
                    "batch_size": 8,
                    "log_every": 0,
                },
            )
            first = RecordingTask(config.schema)
            second = RecordingTask(config.schema)
            runner.fit(
                first,
                ToyDataset(16, 8, 0.0, 21),
                ToyDataset(8, 8, 0.0, 22),
                config,
            )
            runner.fit(
                second,
                ToyDataset(16, 8, 0.0, 21),
                ToyDataset(8, 8, 0.0, 22),
                config,
            )

            self.assertEqual(first.python_draws, second.python_draws)

            seed_everything(config.schema.seed)
            self.assertEqual(first.python_draws[0], random.random())


if __name__ == "__main__":
    unittest.main()
