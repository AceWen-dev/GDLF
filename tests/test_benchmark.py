from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dlef.benchmark import BenchmarkSpec
from dlef.claim import Claim, TestAccessLog
from dlef.config import resolve_config
from dlef.eval import EvalRecord, EvalSpec
from dlef.report import format_claim_report, format_comparison
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner
from examples.toy.components import ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


def benchmark(**overrides) -> BenchmarkSpec:
    values = {
        "name": "bench",
        "dataset": "data",
        "dataset_version": "v1",
        "split": "test",
        "membership_digest": "membership-a",
        "label_version": "labels-v1",
        "transform": "eval-v1",
        "metrics": {"mae": "metric-v1"},
        "aggregation": "mean",
        "thresholds": {},
        "evaluator_version": "eval-code-v1",
    }
    values.update(overrides)
    return BenchmarkSpec(**values)


class BenchmarkIdentityTests(unittest.TestCase):
    def test_test_set_and_benchmark_ids_are_separate(self) -> None:
        base = benchmark()
        transform = benchmark(transform="eval-v2")
        metric = benchmark(metrics={"mae": "metric-v2"})
        labels = benchmark(label_version="labels-v2")
        membership = benchmark(membership_digest="membership-b")

        self.assertEqual(base.test_set_id, transform.test_set_id)
        self.assertEqual(base.test_set_id, metric.test_set_id)
        self.assertEqual(base.test_set_id, labels.test_set_id)
        self.assertNotEqual(base.benchmark_id, transform.benchmark_id)
        self.assertNotEqual(base.benchmark_id, metric.benchmark_id)
        self.assertNotEqual(base.benchmark_id, labels.benchmark_id)

        self.assertNotEqual(base.test_set_id, membership.test_set_id)
        self.assertNotEqual(base.benchmark_id, membership.benchmark_id)

    def test_save_load_and_tamper_detection(self) -> None:
        spec = benchmark()
        with tempfile.TemporaryDirectory() as tmp:
            path = spec.save(Path(tmp) / "bench.json")
            loaded = BenchmarkSpec.load(path)
            self.assertEqual(loaded.benchmark_id, spec.benchmark_id)
            self.assertEqual(loaded.test_set_id, spec.test_set_id)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["benchmark_id"] = "sha256:not-the-real-id"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                BenchmarkSpec.load(path)

    def test_comparison_warns_across_benchmarks(self) -> None:
        def record(eval_id: str, benchmark_id: str) -> EvalRecord:
            return EvalRecord(
                eval_id=eval_id,
                split="test",
                checkpoint_digest="a" * 64,
                data_version="v1",
                eval_spec_digest="b" * 64,
                metrics={"mae": 1.0},
                benchmark_id=benchmark_id,
                test_set_id="test-set",
            )

        table = format_comparison(
            [record("e1", "sha256:bench-a"), record("e2", "sha256:bench-b")]
        )
        self.assertIn("multiple benchmark ids", table)
        self.assertIn("e1", table)
        self.assertIn("e2", table)

    def test_claim_reports_benchmark_mismatch(self) -> None:
        record = EvalRecord(
            eval_id="e1",
            split="test",
            checkpoint_digest="a" * 64,
            data_version="v1",
            eval_spec_digest="b" * 64,
            metrics={"mae": 1.0},
            benchmark_id="sha256:bench-a",
            test_set_id="test-set",
        )
        claim = Claim(
            claim_id="c1",
            text="result",
            eval_ids=("e1",),
            metric="mae",
            benchmark_id="sha256:bench-b",
        )
        run = SimpleNamespace(run_id="run-1")
        report = format_claim_report([claim], {"e1": (run, record)})  # type: ignore[arg-type]
        self.assertIn("MISMATCH", report)


class TestAccessBySetTests(unittest.TestCase):
    def test_runner_records_benchmark_and_counts_test_set(self) -> None:
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
            spec = EvalSpec(
                checkpoint=str(checkpoint),
                split="test",
                benchmark_id="sha256:bench",
                test_set_id="sha256:test-set",
            )

            first = runner.evaluate(task, dataset, spec, run)
            second = runner.evaluate(task, dataset, spec, run)

            self.assertEqual(first.benchmark_id, "sha256:bench")
            self.assertEqual(first.test_set_id, "sha256:test-set")
            self.assertEqual(first.test_access, 1)
            self.assertEqual(second.test_access, 2)
            self.assertEqual(access.count("sha256:test-set", "test"), 2)

    def test_legacy_access_log_is_still_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            access = TestAccessLog(tmp)
            path = access._path("legacy-data-version")
            path.write_text(
                json.dumps(
                    {
                        "data_version": "legacy-data-version",
                        "eval_id": "e1",
                        "split": "test",
                        "n": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(access.count("legacy-data-version", "test"), 1)


if __name__ == "__main__":
    unittest.main()
