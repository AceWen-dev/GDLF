from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.eval import EvalSpec
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from dlef.sweep import (
    Choice,
    GpuPool,
    IntUniform,
    LocalQueue,
    LocalSweep,
    SweepSpec,
    SweepStore,
    TrialResult,
    Uniform,
)
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask

BASE = {
    "seed": 3,
    "epochs": 3,
    "batch_size": 8,
    "n_train": 32,
    "n_val": 16,
    "noise": 0.1,
    "project": "toy-sweep",
    "log_every": 0,
}


class SweepSpecTests(unittest.TestCase):
    def test_grid_generation(self) -> None:
        spec = SweepSpec(
            name="grid",
            params={"a": Choice((1, 2)), "b": Choice((3, 4))},
            method="grid",
        )
        trials = spec.trials("s")
        self.assertEqual(len(trials), 4)
        self.assertEqual(trials[0].params, {"a": 1, "b": 3})
        self.assertEqual(trials[-1].params, {"a": 2, "b": 4})
        self.assertEqual(trials[0].trial_id, "s-t000")

    def test_grid_max_trials(self) -> None:
        spec = SweepSpec(
            name="grid",
            params={"a": Choice((1, 2, 3))},
            method="grid",
            max_trials=2,
        )
        self.assertEqual(len(spec.trials("s")), 2)

    def test_random_generation(self) -> None:
        spec = SweepSpec(
            name="random",
            params={"x": Uniform(0.0, 1.0), "n": IntUniform(1, 3)},
            method="random",
            max_trials=5,
            seed=11,
        )
        trials = spec.trials("s")
        self.assertEqual(len(trials), 5)
        for trial in trials:
            self.assertGreaterEqual(trial.params["x"], 0.0)
            self.assertLessEqual(trial.params["x"], 1.0)
            self.assertIn(trial.params["n"], (1, 2, 3))
        self.assertEqual(
            [trial.params for trial in trials],
            [trial.params for trial in spec.trials("s")],
        )

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            SweepSpec(name="bad", params={}).trials("s")
        with self.assertRaises(ValueError):
            SweepSpec(
                name="bad",
                params={"x": Uniform(0.0, 1.0)},
                method="grid",
            ).trials("s")
        with self.assertRaises(ValueError):
            SweepSpec(
                name="bad",
                params={"x": Uniform(0.0, 1.0)},
                method="random",
            ).trials("s")


class QueueTests(unittest.TestCase):
    def test_gpu_pool(self) -> None:
        pool = GpuPool((0, 1))
        first = pool.acquire(1)
        second = pool.acquire(1)
        self.assertEqual({first[0], second[0]}, {0, 1})
        pool.release(first)
        pool.release(second)
        both = pool.acquire(2)
        self.assertEqual(set(both), {0, 1})
        pool.release(both)

    def test_local_queue_parallel(self) -> None:
        def trial_fn(trial, context):
            time.sleep(0.01)
            return TrialResult(
                trial_id=trial.trial_id,
                metrics={"gpu": context.gpu_ids[0]},
            )

        spec = SweepSpec(
            name="q",
            params={"x": Choice((1, 2, 3, 4))},
            method="grid",
        )
        trials = spec.trials("s")
        queue = LocalQueue((0, 1), max_parallel=2)
        results = queue.run("s", trials, trial_fn)
        self.assertEqual([r.trial_id for r in results], [t.trial_id for t in trials])
        self.assertTrue(all(r.status == "completed" for r in results))
        self.assertEqual({r.metrics["gpu"] for r in results}, {0, 1})


class SweepEndToEndTests(unittest.TestCase):
    def test_local_sweep_toy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(run_store, run_store.artifact_store)
            sweep_store = SweepStore(root / "sweeps")

            def trial_fn(trial, context):
                raw = dict(BASE)
                raw.update(trial.params)
                raw["experiment"] = f"trial-{trial.index}"
                config = resolve_config(ToyConfig, raw)
                train_data = ToyDataset(
                    config.schema.n_train,
                    config.schema.batch_size,
                    config.schema.noise,
                    config.schema.seed,
                )
                val_data = ToyDataset(
                    config.schema.n_val,
                    config.schema.batch_size,
                    config.schema.noise,
                    config.schema.seed + 1,
                )
                task = ToyTask(config.schema)
                run = runner.fit(task, train_data, val_data, config)
                record = runner.evaluate(
                    task,
                    val_data,
                    EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
                    run,
                )
                return TrialResult(
                    trial_id=trial.trial_id,
                    status="completed",
                    run_id=run.run_id,
                    metrics=dict(record.metrics),
                    gpu_ids=context.gpu_ids,
                )

            spec = SweepSpec(
                name="toy",
                params={"lr": Choice((0.1, 0.02))},
                method="grid",
            )
            sweep = LocalSweep(sweep_store, gpu_ids=(0,), max_parallel=1)
            record = sweep.run(spec, trial_fn)

            self.assertEqual(record.status, "completed")
            self.assertEqual(len(record.results), 2)
            self.assertTrue(all(item["status"] == "completed" for item in record.results))
            self.assertTrue(all(item["metrics"].get("mae") is not None for item in record.results))
            self.assertTrue(all(item["run_id"] for item in record.results))
            self.assertTrue((root / "sweeps" / record.sweep_id / "sweep.json").is_file())

    def test_partial_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SweepStore(Path(tmp) / "sweeps")

            def trial_fn(trial, context):
                if trial.index == 0:
                    raise RuntimeError("boom")
                return TrialResult(trial_id=trial.trial_id, metrics={"ok": 1})

            spec = SweepSpec(
                name="fail",
                params={"x": Choice((1, 2))},
                method="grid",
            )
            record = LocalSweep(store).run(spec, trial_fn)
            self.assertEqual(record.status, "partial")
            self.assertEqual(record.results[0]["status"], "failed")
            self.assertEqual(record.results[1]["status"], "completed")


class SweepStoreTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SweepStore(Path(tmp) / "sweeps")
            spec = SweepSpec(
                name="roundtrip",
                params={"x": Choice((1, 2))},
                method="grid",
            )
            record = store.create(spec)
            record.status = "completed"
            store.save(record)
            loaded = store.load(record.sweep_id)
            self.assertEqual(loaded.sweep_id, record.sweep_id)
            self.assertEqual(loaded.status, "completed")
            self.assertEqual(loaded.spec, spec.to_dict())
            self.assertEqual(len(store.list()), 1)


if __name__ == "__main__":
    unittest.main()
