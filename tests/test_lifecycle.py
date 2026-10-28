from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.contracts import CallbackBase, Scheduler
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class CountingCallback(CallbackBase):
    def __init__(self) -> None:
        self.run_start = 0
        self.epoch_start = 0
        self.step_end = 0
        self.epoch_end = 0
        self.run_end = 0

    def on_run_start(self, ctx):
        self.run_start += 1

    def on_epoch_start(self, ctx):
        self.epoch_start += 1

    def on_step_end(self, ctx, step):
        self.step_end += 1

    def on_epoch_end(self, ctx):
        self.epoch_end += 1

    def on_run_end(self, ctx):
        self.run_end += 1


class CountingScheduler:
    step_unit = "step"

    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state) -> None:
        self.calls = int(state["calls"])


class ToyTaskWithHooks(ToyTask):
    def __init__(self, config, *, callback=None, scheduler=None) -> None:
        super().__init__(config)
        self.callback = callback
        self.scheduler = scheduler

    def configure_callbacks(self):
        return [self.callback] if self.callback is not None else []

    def configure_scheduler(self, optimizer):
        return self.scheduler


class LifecycleTests(unittest.TestCase):
    def test_callback_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {"seed": 1, "epochs": 2, "n_train": 32, "batch_size": 8, "log_every": 0},
            )
            callback = CountingCallback()
            task = ToyTaskWithHooks(config.schema, callback=callback)
            runner.fit(task, ToyDataset(32, 8, 0.1, 1), None, config)
            self.assertEqual(callback.run_start, 1)
            self.assertEqual(callback.epoch_start, 2)
            self.assertEqual(callback.step_end, 8)
            self.assertEqual(callback.epoch_end, 2)
            self.assertEqual(callback.run_end, 1)

    def test_scheduler_step_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {"seed": 1, "epochs": 2, "n_train": 32, "batch_size": 8, "log_every": 0},
            )
            scheduler = CountingScheduler()
            task = ToyTaskWithHooks(config.schema, scheduler=scheduler)
            self.assertIsInstance(scheduler, Scheduler)
            runner.fit(task, ToyDataset(32, 8, 0.1, 1), None, config)
            self.assertEqual(scheduler.calls, 8)

    def test_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            config = resolve_config(
                ToyConfig,
                {"seed": 1, "epochs": 10, "max_steps": 3, "n_train": 64, "batch_size": 8, "log_every": 0},
            )
            callback = CountingCallback()
            task = ToyTaskWithHooks(config.schema, callback=callback)
            run = runner.fit(task, ToyDataset(64, 8, 0.1, 1), None, config)
            self.assertEqual(callback.step_end, 3)
            payload = json.loads(run.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["step"], 3)


if __name__ == "__main__":
    unittest.main()
