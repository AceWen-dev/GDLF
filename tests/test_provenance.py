from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dlef.eval import EvalSpec, eval_spec_digest
from dlef.run import RunStore, sha256_file
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask

from dlef.config import resolve_config


def _config(**overrides):
    values = {
        "seed": 1,
        "epochs": 1,
        "n_train": 16,
        "n_val": 8,
        "batch_size": 8,
        "log_every": 0,
        "selection": {"monitor": "mae", "mode": "min"},
    }
    values.update(overrides)
    return resolve_config(ToyConfig, values)


class ProvenanceTests(unittest.TestCase):
    def test_manifest_records_env_extra_and_git_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", code_root=root)
            run = store.create(
                project="p",
                experiment="e",
                config_digest="a" * 64,
                seed=1,
                env_extra={"toydep": "9.9"},
            )
            self.assertEqual(run.manifest.env["toydep"], "9.9")
            self.assertIn("git_dirty", run.manifest.env)
            # temp dir is not a git repo -> dirty is unknown, not False
            self.assertIsNone(run.manifest.env["git_dirty"]["dirty"])

    def test_fit_records_env_extra_and_checkpoint_digests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", code_root=root)
            runner = Runner(store, env_extra={"toydep": "9.9"})
            config = _config()
            task = ToyTask(config.schema)
            run = runner.fit(
                task,
                ToyDataset(16, 8, 0.0, 1),
                ToyDataset(8, 8, 0.0, 2),
                config,
            )

            self.assertEqual(run.manifest.env["toydep"], "9.9")
            self.assertIn("git_dirty", run.manifest.env)

            index = run.dir / "checkpoints" / "digests.json"
            self.assertTrue(index.is_file())
            digests = json.loads(index.read_text(encoding="utf-8"))
            self.assertIn("last.json", digests)
            for name, digest in digests.items():
                self.assertEqual(digest, sha256_file(run.dir / "checkpoints" / name))

    def test_evaluate_saves_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", code_root=root)
            runner = Runner(store)
            config = _config()
            task = ToyTask(config.schema)
            val = ToyDataset(8, 8, 0.0, 2)
            run = runner.fit(task, ToyDataset(16, 8, 0.0, 1), val, config)

            spec = EvalSpec(
                checkpoint=str(run.checkpoint_path),
                split="val",
                save_predictions=True,
            )
            record = runner.evaluate(task, val, spec, run)

            self.assertIsNotNone(record.predictions)
            path = run.dir / str(record.predictions)
            self.assertTrue(path.is_file())
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 1)  # one batch of 8 samples
            self.assertIn("predictions", lines[0])
            self.assertEqual(lines[0]["batch"], 0)

            off = EvalSpec(checkpoint=str(run.checkpoint_path), split="val")
            self.assertNotEqual(eval_spec_digest(spec), eval_spec_digest(off))

            no_pred = runner.evaluate(
                task, val, EvalSpec(checkpoint=str(run.checkpoint_path), split="val"), run
            )
            self.assertIsNone(no_pred.predictions)


if __name__ == "__main__":
    unittest.main()
