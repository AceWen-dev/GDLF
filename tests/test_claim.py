from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dlef.claim import Claim, ClaimError, ClaimStore, TestAccessLog
from dlef.cli import main
from dlef.config import resolve_config
from dlef.eval import EvalSpec
from dlef.report import format_claim_report, write_claim_report
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class ClaimTests(unittest.TestCase):
    def test_test_access_and_claim_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            access = TestAccessLog(root / "test_access")
            runner = Runner(store, store.artifact_store, test_access_log=access)
            config = resolve_config(
                ToyConfig,
                {"seed": 1, "epochs": 1, "n_train": 16, "n_val": 16, "batch_size": 8},
            )
            task = ToyTask(config.schema)
            dataset = ToyDataset(16, 8, 0.1, 1)
            run = store.create(
                project="p",
                experiment="e",
                config_digest="a" * 64,
                seed=1,
            )
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"checkpoint")
            spec = EvalSpec(checkpoint=str(checkpoint), split="test")

            first = runner.evaluate(task, dataset, spec, run)
            second = runner.evaluate(task, dataset, spec, run)
            self.assertEqual(first.test_access, 1)
            self.assertEqual(second.test_access, 2)
            self.assertEqual(access.count(dataset.spec.version, "test"), 2)
            self.assertEqual(len(access.entries()), 2)

            found = store.find_eval(second.eval_id)
            self.assertIsNotNone(found)
            found_run, found_record = found
            self.assertEqual(found_run.run_id, run.run_id)
            self.assertEqual(found_record.eval_id, second.eval_id)
            self.assertEqual(len(store.iter_evals()), 2)

            claims = ClaimStore(root / "claims")
            claim = Claim(
                claim_id="c1",
                text="Baseline comparison",
                eval_ids=(second.eval_id,),
                metric="mae",
                unit="points",
            )
            claims.add(claim)
            self.assertEqual(claims.get("c1").text, "Baseline comparison")
            self.assertEqual(len(claims.list()), 1)
            with self.assertRaises(ClaimError):
                claims.add(claim)
            with self.assertRaises(ClaimError):
                claims.add(Claim(claim_id="bad", text="no eval", eval_ids=()))

            report = format_claim_report(
                [claim],
                {second.eval_id: (run, second)},
            )
            self.assertIn("Baseline comparison", report)
            self.assertIn(second.eval_id, report)
            self.assertIn(run.run_id, report)
            self.assertIn("WARN 2", report)

            path = write_claim_report(root / "report.md", [claim], {second.eval_id: (run, second)})
            self.assertTrue(path.is_file())
            self.assertIn("Baseline comparison", path.read_text(encoding="utf-8"))

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "report",
                        "--runs",
                        str(root / "runs"),
                        "--claims",
                        str(root / "claims"),
                        "--out",
                        str(root / "cli-report.md"),
                    ]
                )
            self.assertEqual(exit_code, 0)
            cli_report = (root / "cli-report.md").read_text(encoding="utf-8")
            self.assertIn("Baseline comparison", cli_report)
            self.assertIn(second.eval_id, cli_report)

    def test_missing_eval_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims = ClaimStore(root / "claims")
            claim = Claim(
                claim_id="c1",
                text="Missing eval",
                eval_ids=("does-not-exist",),
            )
            claims.add(claim)
            report = format_claim_report([claim], {})
            self.assertIn("MISSING", report)


if __name__ == "__main__":
    unittest.main()
