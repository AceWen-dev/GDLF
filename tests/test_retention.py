from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from dlef.eval import EvalRecord
from dlef.retention import RetentionPolicy, RetentionService
from dlef.run import ArtifactStore, RunStore
from dlef.sweep import SweepRecord, SweepStore


def make_run(
    store: RunStore,
    *,
    experiment: str = "e",
    started: str,
    seed: int,
    metric: float | None = None,
    tags: dict[str, str] | None = None,
    status: str = "completed",
    parent: str | None = None,
):
    run = store.create(
        project="p",
        experiment=experiment,
        config_digest=f"{seed:064d}",
        seed=seed,
        tags=tags or {},
        parent_run_id=parent,
    )
    if metric is not None:
        record = EvalRecord(
            eval_id=f"eval-{seed}",
            split="val",
            checkpoint_digest="a" * 64,
            data_version="v",
            eval_spec_digest="b" * 64,
            metrics={"mae": metric},
        )
        run.save_json(f"evals/eval-{seed}.json", record.to_dict())
    if status != "running":
        run.finalize(status)
    run.manifest.started_at = started
    run.write_manifest()
    return run


def permissive(**overrides) -> RetentionPolicy:
    values = {
        "keep_last": 0,
        "keep_best": 0,
        "keep_every": 0,
        "keep_tags": frozenset(),
        "protect_status": frozenset(),
        "min_age_days": 0.0,
        "artifact_ttl_days": 0.0,
        "protect_sweep_runs": False,
    }
    values.update(overrides)
    return RetentionPolicy(**values)


class RetentionTests(unittest.TestCase):
    def test_keep_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs")
            runs = [
                make_run(store, started=f"2026-01-0{day}T00:00:00+00:00", seed=day)
                for day in range(1, 5)
            ]
            service = RetentionService(store)
            plan = service.plan(permissive(keep_last=2))
            kept = {action.run_id for action in plan.runs if action.action == "keep"}
            deleted = {action.run_id for action in plan.runs if action.action == "delete"}
            self.assertEqual(kept, {runs[2].run_id, runs[3].run_id})
            self.assertEqual(deleted, {runs[0].run_id, runs[1].run_id})
            self.assertIn("delete", plan.summary())

    def test_keep_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs")
            runs = [
                make_run(store, started=f"2026-02-0{day}T00:00:00+00:00", seed=day, metric=value)
                for day, value in zip((1, 2, 3), (0.5, 0.1, 0.3))
            ]
            plan = RetentionService(store).plan(
                permissive(keep_best=1, metric_key="mae", metric_mode="min")
            )
            kept = {action.run_id for action in plan.runs if action.action == "keep"}
            self.assertEqual(kept, {runs[1].run_id})

    def test_keep_every(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs")
            runs = [
                make_run(store, started=f"2026-03-0{day}T00:00:00+00:00", seed=day)
                for day in range(1, 5)
            ]
            plan = RetentionService(store).plan(permissive(keep_every=2))
            kept = {action.run_id for action in plan.runs if action.action == "keep"}
            self.assertEqual(kept, {runs[0].run_id, runs[2].run_id})

    def test_tags_status_and_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs")
            parent = make_run(store, started="2026-04-01T00:00:00+00:00", seed=1)
            child = make_run(
                store,
                started="2026-04-02T00:00:00+00:00",
                seed=2,
                tags={"reported": "yes"},
                parent=parent.run_id,
            )
            running = make_run(
                store,
                started="2026-04-03T00:00:00+00:00",
                seed=3,
                status="running",
            )
            plan = RetentionService(store).plan(
                permissive(keep_tags=frozenset({"reported"}), protect_status=frozenset({"running"}))
            )
            kept = {action.run_id for action in plan.runs if action.action == "keep"}
            self.assertEqual(kept, {parent.run_id, child.run_id, running.run_id})
            reasons = {action.run_id: action.reason for action in plan.runs}
            self.assertIn("parent-of", reasons[parent.run_id])
            self.assertIn("tag:reported", reasons[child.run_id])
            self.assertIn("status:running", reasons[running.run_id])

    def test_min_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs")
            old = make_run(store, started="2020-01-01T00:00:00+00:00", seed=1)
            young = make_run(
                store,
                started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                seed=2,
            )
            plan = RetentionService(store).plan(permissive(min_age_days=1.0))
            kept = {action.run_id for action in plan.runs if action.action == "keep"}
            deleted = {action.run_id for action in plan.runs if action.action == "delete"}
            self.assertEqual(kept, {young.run_id})
            self.assertEqual(deleted, {old.run_id})

    def test_sweep_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            run = make_run(store, started="2026-05-01T00:00:00+00:00", seed=1)
            sweep_store = SweepStore(root / "sweeps")
            sweep_store.save(
                SweepRecord(
                    sweep_id="sweep-1",
                    spec={},
                    results=[{"run_id": run.run_id}],
                )
            )
            protected = RetentionService(store, sweep_store=sweep_store).plan(
                permissive(protect_sweep_runs=True)
            )
            self.assertEqual({a.action for a in protected.runs}, {"keep"})
            unprotected = RetentionService(store, sweep_store=sweep_store).plan(
                permissive(protect_sweep_runs=False)
            )
            self.assertEqual({a.action for a in unprotected.runs}, {"delete"})

    def test_artifact_gc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            run = make_run(store, started="2026-06-01T00:00:00+00:00", seed=1)
            kept_file = root / "kept.bin"
            kept_file.write_bytes(b"k" * 1024)
            kept_digest = run.save_artifact(kept_file, "checkpoint")

            orphan_file = root / "orphan.bin"
            orphan_file.write_bytes(b"o" * 2048)
            orphan_digest = store.artifact_store.put(orphan_file, "orphan")
            orphan_path = store.artifact_store.get(orphan_digest, "orphan")
            old = time.time() - 40 * 86400
            os.utime(orphan_path, (old, old))

            service = RetentionService(store)
            plan = service.plan(permissive(keep_last=1, artifact_ttl_days=30.0))
            self.assertEqual(plan.delete_runs, ())
            self.assertEqual({a.digest for a in plan.delete_artifacts}, {orphan_digest})
            self.assertTrue(plan.delete_artifacts[0].size_bytes >= 2048)

            dry = service.apply(plan, dry_run=True)
            self.assertTrue(dry.dry_run)
            self.assertTrue(orphan_path.is_file())

            applied = service.apply(plan, dry_run=False)
            self.assertFalse(orphan_path.is_file())
            self.assertTrue(service.artifact_store.exists(kept_digest, "checkpoint"))
            self.assertGreaterEqual(applied.bytes_freed, 2048)

    def test_apply_run_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs")
            run = make_run(store, started="2026-07-01T00:00:00+00:00", seed=1)
            service = RetentionService(store)
            plan = service.plan(permissive())
            result = service.apply(plan, dry_run=False)
            self.assertFalse(run.dir.exists())
            self.assertEqual(result.deleted_runs, [run.run_id])
            self.assertGreater(result.bytes_freed, 0)


if __name__ == "__main__":
    unittest.main()
