from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from dlef.config import resolve_config
from dlef.policy import NoOpTrainingPolicy
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class CountingPolicy(NoOpTrainingPolicy):
    def __init__(self, accum: int = 1) -> None:
        self.accum = accum
        self.setup_count = 0
        self.start_count = 0
        self.forward_count = 0
        self.scale_count = 0
        self.backward_count = 0
        self.should_count = 0
        self.flush_count = 0
        self.step_count = 0
        self.zero_count = 0
        self.end_count = 0

    def setup(self) -> None:
        self.setup_count += 1

    def on_fit_start(self, task, optimizer, config) -> None:
        self.start_count += 1

    def forward_context(self):
        self.forward_count += 1
        return nullcontext()

    def scale_loss(self, loss):
        self.scale_count += 1
        return super().scale_loss(loss)

    def backward(self, loss) -> None:
        self.backward_count += 1
        super().backward(loss)

    def should_step(self, step: int) -> bool:
        self.should_count += 1
        return step % self.accum == 0

    def should_flush(self, step: int) -> bool:
        self.flush_count += 1
        return step % self.accum != 0

    def optimizer_step(self, task, optimizer) -> None:
        self.step_count += 1
        super().optimizer_step(task, optimizer)

    def zero_grad(self, optimizer) -> None:
        self.zero_count += 1
        super().zero_grad(optimizer)

    def on_fit_end(self, task, optimizer) -> None:
        self.end_count += 1

    def state_dict(self):
        return {"steps": self.step_count}

    def load_state_dict(self, state) -> None:
        self.step_count = int(state.get("steps", 0))


class PolicyIntegrationTests(unittest.TestCase):
    def _run(self, policy, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store, training_policy=policy)
            raw = {
                "seed": 1,
                "epochs": 1,
                "n_train": 32,
                "n_val": 16,
                "batch_size": 8,
                "log_every": 0,
            }
            raw.update(overrides)
            config = resolve_config(ToyConfig, raw)
            run = runner.fit(
                ToyTask(config.schema),
                ToyDataset(config.schema.n_train, 8, 0.1, 1),
                ToyDataset(config.schema.n_val, 8, 0.1, 2),
                config,
            )
            payload = (
                json.loads(run.checkpoint_path.read_text(encoding="utf-8"))
                if run is not None
                else None
            )
            return run, payload

    def test_hooks_and_accumulation(self) -> None:
        policy = CountingPolicy(accum=2)
        run, payload = self._run(policy)
        self.assertIsNotNone(run)
        self.assertEqual(policy.setup_count, 1)
        self.assertEqual(policy.start_count, 1)
        self.assertEqual(policy.scale_count, 4)
        self.assertEqual(policy.backward_count, 4)
        self.assertEqual(policy.should_count, 4)
        self.assertEqual(policy.step_count, 2)
        self.assertEqual(policy.zero_count, 2)
        self.assertEqual(policy.end_count, 1)
        self.assertEqual(policy.flush_count, 1)
        self.assertEqual(policy.forward_count, 6)  # 4 train + 2 val batches
        self.assertEqual(payload["policy"]["steps"], 2)

    def test_flush_partial_accumulation(self) -> None:
        policy = CountingPolicy(accum=2)
        run, _ = self._run(policy, max_steps=3)
        self.assertIsNotNone(run)
        self.assertEqual(policy.backward_count, 3)
        self.assertEqual(policy.should_count, 3)
        self.assertEqual(policy.step_count, 2)  # step at 2, flush at 3
        self.assertEqual(policy.zero_count, 2)


if __name__ == "__main__":
    unittest.main()
