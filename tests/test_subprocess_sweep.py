from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from dlef.sweep import (
    Choice,
    SubprocessSweep,
    SweepSpec,
    SweepStore,
    TrialCommand,
)

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "fixtures" / "trial_worker.py"


class SubprocessSweepTests(unittest.TestCase):
    def test_success_and_gpu_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SweepStore(root / "sweeps")
            spec = SweepSpec(
                name="proc",
                params={"x": Choice((1, 2, 3, 4))},
                method="grid",
            )

            def command_fn(trial, context):
                return TrialCommand(
                    argv=[sys.executable, str(WORKER)],
                    cwd=str(ROOT),
                )

            sweep = SubprocessSweep(
                store,
                gpu_ids=(0, 1),
                max_parallel=2,
                gpus_per_trial=1,
            )
            record = sweep.run(spec, command_fn)

            self.assertEqual(record.status, "completed")
            self.assertEqual(len(record.results), 4)
            self.assertTrue(all(item["status"] == "completed" for item in record.results))
            assigned = {tuple(item["metrics"]["gpu"]) for item in record.results}
            self.assertEqual(assigned, {(0,), (1,)})
            self.assertTrue(all(item["log_path"] for item in record.results))
            self.assertTrue(all(item["run_id"] for item in record.results))

            trial_dir = root / "sweeps" / record.sweep_id / "trials"
            self.assertEqual(len(list(trial_dir.glob("*.log"))), 4)
            self.assertEqual(len(list(trial_dir.glob("*.result.json"))), 4)
            logs = "\n".join(path.read_text(encoding="utf-8") for path in trial_dir.glob("*.log"))
            self.assertIn("worker done", logs)

    def test_failure_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SweepStore(root / "sweeps")
            spec = SweepSpec(
                name="fail",
                params={"x": Choice((1, 2))},
                method="grid",
            )

            def command_fn(trial, context):
                return TrialCommand(
                    argv=[sys.executable, str(WORKER)],
                    env={"DLEF_FAIL": "1"} if trial.index == 0 else {},
                    cwd=str(ROOT),
                )

            record = SubprocessSweep(store, gpu_ids=(0,)).run(spec, command_fn)
            self.assertEqual(record.status, "partial")
            self.assertEqual(record.results[0]["status"], "failed")
            self.assertIn("exit code 3", record.results[0]["error"])
            self.assertEqual(record.results[1]["status"], "completed")

    def test_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SweepStore(root / "sweeps")
            spec = SweepSpec(
                name="timeout",
                params={"x": Choice((1,))},
                method="grid",
            )

            def command_fn(trial, context):
                return TrialCommand(
                    argv=[sys.executable, "-c", "import time; time.sleep(2)"],
                    timeout_s=0.2,
                    cwd=str(ROOT),
                )

            record = SubprocessSweep(store, gpu_ids=(0,)).run(spec, command_fn)
            self.assertEqual(record.status, "partial")
            self.assertEqual(record.results[0]["status"], "failed")
            self.assertIn("timeout", record.results[0]["error"])


if __name__ == "__main__":
    unittest.main()
