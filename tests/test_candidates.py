from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dlef.candidates import (
    Candidate,
    CandidateError,
    CandidateStore,
)
from dlef.cli import main
from dlef.report import (
    candidate_metric,
    format_candidate_report,
    format_report,
)
from dlef.run import RunStore


def make_run(store: RunStore, *, seed: int, experiment: str = "e"):
    run = store.create(
        project="p",
        experiment=experiment,
        config_digest=f"{seed:064d}",
        seed=seed,
    )
    return run


class CandidateStoreTests(unittest.TestCase):
    def test_add_get_list_and_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateStore(Path(tmp) / "candidates")
            candidate = Candidate(
                candidate_id="exp-a",
                title="Experiment A",
                run_ids=("r1", "r2", "r2"),
                reason="promising on val",
                metric="mae",
                approved_by="owner 2026-11-18: ok",
            )
            store.add(candidate)

            self.assertEqual(candidate.run_ids, ("r1", "r2"))
            self.assertEqual(store.get("exp-a").title, "Experiment A")
            self.assertEqual(len(store.list()), 1)
            self.assertEqual(
                store.protected_run_ids(),
                {"r1": ["exp-a"], "r2": ["exp-a"]},
            )

            promoted = store.update(
                "exp-a", status="promoted", claim_ids=("claim-1",)
            )
            self.assertEqual(promoted.status, "promoted")
            self.assertEqual(promoted.claim_ids, ("claim-1",))
            self.assertGreaterEqual(promoted.updated_at, candidate.updated_at)
            self.assertEqual(
                store.protected_run_ids(), {"r1": ["exp-a"], "r2": ["exp-a"]}
            )

            rejected = Candidate(
                candidate_id="exp-b",
                title="Experiment B",
                run_ids=("r3",),
                status="rejected",
                reason="regressed",
                approved_by="owner 2026-11-18: not worth it",
            )
            store.add(rejected)
            self.assertEqual(store.protected_run_ids(), {"r1": ["exp-a"], "r2": ["exp-a"]})
            self.assertEqual(store.list(status="rejected"), [rejected])

            self.assertTrue(store.remove("exp-b"))
            self.assertFalse(store.remove("exp-b"))
            with self.assertRaises(CandidateError):
                store.get("exp-b")

    def test_validation_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateStore(Path(tmp))

            bad_cases = [
                Candidate(candidate_id="x", title="", run_ids=("r1",), reason="r", approved_by="owner"),
                Candidate(candidate_id="x", title="t", run_ids=(), reason="r", approved_by="owner"),
                Candidate(
                    candidate_id="x",
                    title="t",
                    run_ids=("r1",),
                    status="unknown",
                    reason="r",
                    approved_by="owner",
                ),
                Candidate(candidate_id="x", title="t", run_ids=("r1",)),
                Candidate(
                    candidate_id="x",
                    title="t",
                    run_ids=("r1",),
                    status="rejected",
                    approved_by="owner",
                ),
                Candidate(
                    candidate_id="x",
                    title="t",
                    run_ids=("r1",),
                    status="promoted",
                    reason="r",
                    approved_by="owner",
                ),
            ]
            for candidate in bad_cases:
                with self.assertRaises(CandidateError):
                    store.add(candidate)

            good = Candidate(
                candidate_id="x",
                title="t",
                run_ids=("r1",),
                reason="r",
                approved_by="owner 2026-11-18: ok",
            )
            store.add(good)
            with self.assertRaises(CandidateError):
                store.add(good)
            with self.assertRaises(CandidateError):
                CandidateStore(Path(tmp)).get("../escape")
            with self.assertRaises(CandidateError):
                store.update("x", unknown="value")
            with self.assertRaises(CandidateError):
                store.update("x", candidate_id="y")

    def test_legacy_record_needs_reapproval(self) -> None:
        """Old candidate JSON without approved_by cannot mutate until re-approved."""

        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateStore(Path(tmp))
            payload = {
                "candidate_id": "legacy",
                "title": "Legacy",
                "run_ids": ["r1"],
                "reason": "imported",
                "status": "candidate",
            }
            path = store.root / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            legacy = store.get("legacy")
            self.assertEqual(legacy.approved_by, "")
            with self.assertRaises(CandidateError):
                store.update("legacy", status="promoted", claim_ids=("c1",))
            reapproved = store.update(
                "legacy", status="promoted", claim_ids=("c1",), approved_by="owner: ok"
            )
            self.assertEqual(reapproved.status, "promoted")
            self.assertEqual(reapproved.approved_by, "owner: ok")

    def test_update_ignores_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateStore(Path(tmp))
            store.add(
                Candidate(
                    candidate_id="x",
                    title="t",
                    run_ids=("r1",),
                    reason="r",
                    next="look at bins",
                    approved_by="owner: ok",
                )
            )
            updated = store.update("x", next=None, metric="mae")
            self.assertEqual(updated.next, "look at bins")
            self.assertEqual(updated.metric, "mae")


class CandidateReportTests(unittest.TestCase):
    def test_candidate_table_and_metric_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs")
            runs = [make_run(store, seed=seed) for seed in (1, 2)]
            for run, value in zip(runs, (0.4, 0.6)):
                run.log_metrics({"phase": "val", "epoch": 1, "mae": value})

            candidates = [
                Candidate(
                    candidate_id="exp-a",
                    title="Experiment A",
                    run_ids=(runs[0].run_id, runs[1].run_id, "missing-run"),
                    reason="promising on val",
                    metric="mae",
                    experiment="e",
                    approved_by="owner: ok",
                )
            ]
            mean, resolved, total = candidate_metric(
                candidates[0], {run.run_id: run for run in runs}
            )
            self.assertAlmostEqual(mean, 0.5)  # type: ignore[arg-type]
            self.assertEqual((resolved, total), (2, 3))

            table = format_candidate_report(
                candidates, {run.run_id: run for run in runs}
            )
            self.assertIn("exp-a", table)
            self.assertIn("0.5 (2/3)", table)
            self.assertIn("promising on val", table)
            self.assertIn("1 candidate(s), 3 run(s)", table)

    def test_report_puts_shortlist_before_claims(self) -> None:
        candidate = Candidate(
            candidate_id="exp-a",
            title="Experiment A",
            run_ids=("r1",),
            reason="promising",
            approved_by="owner: ok",
        )
        text = format_report([], {}, candidates=[candidate], runs={})
        self.assertLess(text.index("## Shortlist"), text.index("## Claims"))
        self.assertIn("exp-a", text)
        self.assertIn("owner: ok", text)
        self.assertIn("(no claims)", text)

    def test_report_note_is_rendered_under_the_title(self) -> None:
        text = format_report([], {}, note="generated view; rebuild me")
        self.assertIn("# Experiment report", text)
        self.assertLess(text.index("> generated view"), text.index("## Claims"))

    def test_cli_report_with_only_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs")
            run = make_run(store, seed=1)
            run.log_metrics({"phase": "val", "epoch": 1, "mae": 0.25})

            candidates = CandidateStore(root / "candidates")
            candidates.add(
                Candidate(
                    candidate_id="exp-a",
                    title="Experiment A",
                    run_ids=(run.run_id,),
                    reason="promising",
                    metric="mae",
                    approved_by="owner: ok",
                )
            )

            out = root / "report.md"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "report",
                        "--runs",
                        str(root / "runs"),
                        "--claims",
                        str(root / "claims"),
                        "--candidates",
                        str(root / "candidates"),
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("## Shortlist", text)
            self.assertIn("exp-a", text)
            self.assertIn("0.25", text)
            self.assertIn("(no claims)", text)


if __name__ == "__main__":
    unittest.main()
