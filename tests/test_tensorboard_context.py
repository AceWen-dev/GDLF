from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.contracts import CallbackBase
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class CaptureCallback(CallbackBase):
    def __init__(self) -> None:
        self.steps = []
        self.epochs = []
        self.selections = []

    def on_step_end(self, ctx, step):
        self.steps.append(dict(ctx.step_metrics))

    def on_epoch_end(self, ctx):
        self.epochs.append(dict(ctx.epoch_metrics))
        self.selections.append(dict(ctx.selection))


class TaskWithCallback(ToyTask):
    def __init__(self, config, callback):
        super().__init__(config)
        self.callback = callback

    def configure_callbacks(self):
        return [self.callback]


class TensorBoardContextTests(unittest.TestCase):
    def test_runner_publishes_generic_callback_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {
                    "seed": 1,
                    "epochs": 1,
                    "n_train": 16,
                    "n_val": 8,
                    "batch_size": 8,
                    "log_every": 0,
                    "selection": {"monitor": "mae", "mode": "min"},
                },
            )
            callback = CaptureCallback()
            task = TaskWithCallback(config.schema, callback)
            run = runner.fit(
                task,
                ToyDataset(16, 8, 0.0, 1),
                ToyDataset(8, 8, 0.0, 2),
                config,
            )

            self.assertIsNotNone(run)
            self.assertEqual(len(callback.steps), 2)
            self.assertIn("step_time_ms", callback.steps[0])
            self.assertIn("samples_per_s", callback.steps[0])
            self.assertEqual(len(callback.epochs), 1)
            self.assertIn("train_loss", callback.epochs[0])
            self.assertIn("duration_s", callback.epochs[0])
            self.assertEqual(callback.selections[0]["best_raw_epoch"], 1)
            self.assertEqual(callback.selections[0]["checkpoint"], "best_raw")


if __name__ == "__main__":
    unittest.main()
