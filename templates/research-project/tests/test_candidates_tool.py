from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dlef.claim import Claim, ClaimStore
from dlef.run import RunStore

from core.tools import candidates as candidates_tool


APPROVAL = "twj 2026-11-18: approved for tests"


class CandidatesToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runs_root = root / "runs"
        self.candidates_root = root / "candidates"
        self.claims_root = root / "claims"
        store = RunStore(self.runs_root)
        self.runs = []
        for experiment, seed, metric in (
            ("v11-ranksim", 1, 2.5),
            ("v11-ranksim", 2, 2.3),
            ("v11-baseline", 1, 3.1),
        ):
            run = store.create(
                project="p",
                experiment=experiment,
                config_digest=f"{seed:064d}",
                seed=seed,
            )
            run.log_metrics({"phase": "val", "epoch": 1, "mae": metric})
            run.finalize("completed")
            self.runs.append(run)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _call(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        previous = sys.argv
        sys.argv = ["candidates.py", *args]
        try:
            with redirect_stdout(output):
                code = candidates_tool.main()
        except SystemExit as exc:  # argparse usage errors
            return int(exc.code or 2), output.getvalue()
        finally:
            sys.argv = previous
        return code, output.getvalue()

    def _paths(self, *args: str) -> tuple[int, str]:
        return self._call(
            "--runs",
            str(self.runs_root),
            "--candidates",
            str(self.candidates_root),
            "--claims",
            str(self.claims_root),
            *args,
        )

    def test_add_list_show(self) -> None:
        code, output = self._paths(
            "add",
            "--id",
            "ranksim",
            "--title",
            "RankSim, 2 seeds",
            "--run",
            self.runs[0].run_id,
            "--run",
            self.runs[1].run_id,
            "--reason",
            "best val mae",
            "--metric",
            "mae",
            "--approved-by",
            APPROVAL,
        )
        self.assertEqual(code, 0)
        self.assertIn("candidate added", output)
        self.assertIn(APPROVAL, output)

        payload = json.loads(
            (self.candidates_root / "ranksim.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["status"], "candidate")
        self.assertEqual(payload["experiment"], "v11-ranksim")
        self.assertEqual(payload["approved_by"], APPROVAL)
        self.assertEqual(len(payload["run_ids"]), 2)

        code, output = self._paths("list", "--json")
        self.assertEqual(code, 0)
        listed = json.loads(output)
        self.assertEqual(listed[0]["candidate_id"], "ranksim")

        code, output = self._paths("show", "ranksim")
        self.assertEqual(code, 0)
        self.assertIn(self.runs[0].run_id, output)
        self.assertIn("mae=2.5", output)
        self.assertIn(APPROVAL, output)

    def test_add_without_approval_is_refused(self) -> None:
        code, _ = self._paths(
            "add",
            "--id",
            "nope",
            "--title",
            "Nope",
            "--run",
            self.runs[0].run_id,
            "--reason",
            "test",
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.candidates_root / "nope.json").exists())

    def test_promote_and_reject(self) -> None:
        for candidate_id, run in (
            ("good", self.runs[0]),
            ("bad", self.runs[2]),
        ):
            code, _ = self._paths(
                "add",
                "--id",
                candidate_id,
                "--title",
                candidate_id,
                "--run",
                run.run_id,
                "--reason",
                "test",
                "--metric",
                "mae",
                "--approved-by",
                APPROVAL,
            )
            self.assertEqual(code, 0)

        ClaimStore(self.claims_root).add(
            Claim(
                claim_id="claim-good",
                text="good result",
                eval_ids=("eval-1",),
                metric="mae",
                approved_by=APPROVAL,
            )
        )

        code, output = self._paths(
            "promote",
            "good",
            "--claim",
            "claim-good",
            "--approved-by",
            APPROVAL,
        )
        self.assertEqual(code, 0)
        self.assertIn("promoted", output)
        promoted = json.loads(
            (self.candidates_root / "good.json").read_text(encoding="utf-8")
        )
        self.assertEqual(promoted["status"], "promoted")
        self.assertEqual(promoted["claim_ids"], ["claim-good"])
        self.assertEqual(promoted["approved_by"], APPROVAL)

        code, output = self._paths(
            "reject",
            "bad",
            "--reason",
            "regressed on the D cohort",
            "--approved-by",
            APPROVAL,
        )
        self.assertEqual(code, 0)
        rejected = json.loads(
            (self.candidates_root / "bad.json").read_text(encoding="utf-8")
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("regressed", rejected["reason"])

        code, output = self._paths("list")
        self.assertEqual(code, 0)
        self.assertIn("good", output)
        self.assertNotIn("bad", output)

        code, output = self._paths("list", "--all")
        self.assertEqual(code, 0)
        self.assertIn("bad", output)

    def test_promote_requires_existing_claim(self) -> None:
        self._paths(
            "add",
            "--id",
            "candidate",
            "--title",
            "Candidate",
            "--run",
            self.runs[0].run_id,
            "--reason",
            "test",
            "--approved-by",
            APPROVAL,
        )
        code, _ = self._paths(
            "promote",
            "candidate",
            "--claim",
            "missing",
            "--approved-by",
            APPROVAL,
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
