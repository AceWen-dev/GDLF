from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dlef.config import resolve_config

try:
    import torch  # noqa: F401

    from core.commands import eval as eval_command
    from core.config import ProjectConfig
    from core.data import make_loaders
    from core.runtime import build_policy, build_runner
    from core.task import build_task

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class EvalCommandTests(unittest.TestCase):
    def test_eval_run_links_parent_and_saves_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = resolve_config(
                ProjectConfig,
                {
                    "seed": 1,
                    "epochs": 1,
                    "project": "smoke",
                    "experiment": "smoke",
                    "log_every": 0,
                    "data": {"n_train": 32, "n_val": 16, "batch_size": 8},
                    "model": {"hidden": 8},
                    "optim": {"lr": 0.02},
                },
            )
            policy = build_policy(config.schema)
            policy.setup()
            runner = build_runner(root, policy)
            train_loader, val_loader = make_loaders(
                config.schema.data, config.schema.seed
            )
            task = build_task(config.schema, device=policy.device)
            run = runner.fit(task, train_loader, val_loader, config)
            self.assertIsNotNone(run)

            argv = [
                "eval.py",
                "--checkpoint",
                str(run.checkpoint_path),
                "--split",
                "val",
                "--set",
                f"output_root={root}",
            ]
            output = io.StringIO()
            previous = sys.argv
            sys.argv = argv
            try:
                with redirect_stdout(output):
                    code = eval_command.main()
            finally:
                sys.argv = previous
            self.assertEqual(code, 0, output.getvalue())

            manifests = sorted(
                (root / "runs" / "smoke" / "smoke-eval").glob("*/manifest.json")
            )
            self.assertEqual(len(manifests), 1)
            eval_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(eval_manifest["parent_run_id"], run.run_id)
            self.assertIn("torch", eval_manifest["env"])

            records = sorted((manifests[0].parent / "evals").glob("*.json"))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertTrue(record["predictions"])
            predictions = manifests[0].parent / record["predictions"]
            self.assertTrue(predictions.is_file())
            lines = predictions.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)  # 16 val samples / batch 8
            self.assertIn("predictions", json.loads(lines[0]))


if __name__ == "__main__":
    unittest.main()
