from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.eval import EvalSpec
from dlef.report import format_comparison, write_comparison
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class ToyEndToEndTests(unittest.TestCase):
    def test_two_runs_eval_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            records = []

            for lr in (0.1, 0.02):
                config = resolve_config(
                    ToyConfig,
                    {
                        "seed": 7,
                        "epochs": 20,
                        "lr": lr,
                        "batch_size": 16,
                        "n_train": 128,
                        "n_val": 64,
                        "noise": 0.1,
                        "project": "toy",
                        "experiment": f"lr-{lr}",
                        "log_every": 5,
                    },
                )
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

                self.assertEqual(run.manifest.status, "completed")
                self.assertEqual(run.manifest.config_digest, config.digest)
                self.assertIn("train", run.manifest.data_versions)
                self.assertTrue(run.config_path.is_file())
                self.assertEqual(json.loads(run.config_path.read_text(encoding="utf-8"))["seed"], 7)
                self.assertTrue(run.metrics_path.is_file())
                self.assertTrue(run.checkpoint_path.is_file())
                self.assertIn("checkpoint:last", run.manifest.artifacts)

                losses = [
                    json.loads(line)["loss"]
                    for line in run.metrics_path.read_text(encoding="utf-8").splitlines()
                    if json.loads(line).get("phase") == "train"
                ]
                self.assertTrue(losses)
                self.assertLess(losses[-1], losses[0])

                record = runner.evaluate(
                    task,
                    val_data,
                    EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
                    run,
                )
                self.assertIn("mae", record.metrics)
                self.assertTrue((run.dir / "evals" / f"{record.eval_id}.json").is_file())
                records.append(record)

            table = format_comparison(records)
            self.assertIn("mae", table)
            for record in records:
                self.assertIn(record.eval_id, table)

            report = write_comparison(root / "comparison.md", records)
            self.assertTrue(report.is_file())
            self.assertIn("mae", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
