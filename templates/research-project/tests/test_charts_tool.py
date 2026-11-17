from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dlef.candidates import Candidate, CandidateStore
from dlef.claim import Claim, ClaimStore
from dlef.eval import EvalRecord
from dlef.run import RunStore

from core.tools import charts as charts_tool


class ChartsToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.output = self.root / "output"
        self.views = self.root / "charts.yaml"
        self.views.write_text(
            "\n".join(
                [
                    "defaults:",
                    "  mode: min",
                    "charts:",
                    "  - id: experiments-val",
                    "    title: 'Experiments {metric}'",
                    "    source: experiments",
                    "    metric: mae",
                    "  - id: candidates-val",
                    "    title: 'Candidates {metric}'",
                    "    source: candidates",
                    "    metric: mae",
                    "  - id: claims-test",
                    "    title: 'Claims {metric}'",
                    "    source: claims",
                    "    metric: mae",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        store = RunStore(self.output / "runs")
        self.runs = []
        for experiment, seed, value in (
            ("baseline", 1, 0.5),
            ("baseline", 2, 0.4),
            ("tuned", 1, 0.2),
        ):
            run = store.create(
                project="p",
                experiment=experiment,
                config_digest=f"{seed:064d}",
                seed=seed,
            )
            run.log_metrics({"phase": "val", "epoch": 1, "mae": value})
            run.finalize("completed")
            self.runs.append(run)

        candidate = CandidateStore(self.output / "candidates")
        candidate.add(
            Candidate(
                candidate_id="tuned-candidate",
                title="Tuned",
                run_ids=(self.runs[2].run_id,),
                reason="best val",
                metric="mae",
            )
        )

        record = EvalRecord(
            eval_id="eval-1",
            split="test",
            checkpoint_digest="a" * 64,
            data_version="v1",
            eval_spec_digest="b" * 64,
            metrics={"mae": 0.18},
            benchmark_id="sha256:" + "c" * 64,
        )
        self.runs[2].save_json("evals/eval-1.json", record.to_dict())
        ClaimStore(self.output / "claims").add(
            Claim(
                claim_id="main-result",
                text="tuned test mae",
                eval_ids=("eval-1",),
                metric="mae",
                value=0.18,
            )
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *extra: str) -> tuple[int, str]:
        output = io.StringIO()
        previous = sys.argv
        sys.argv = [
            "charts.py",
            "--output",
            str(self.output),
            "--views",
            str(self.views),
            *extra,
        ]
        try:
            with redirect_stdout(output):
                code = charts_tool.main()
        finally:
            sys.argv = previous
        return code, output.getvalue()

    def test_generates_index_and_svgs(self) -> None:
        code, output = self._run()
        self.assertEqual(code, 0)
        self.assertIn("charts:", output)

        charts = self.output / "charts"
        self.assertTrue((charts / "index.md").is_file())
        for name in ("experiments-val.svg", "candidates-val.svg", "claims-test.svg"):
            path = charts / name
            self.assertTrue(path.is_file(), name)
            self.assertIn("<svg", path.read_text(encoding="utf-8"))

        index = (charts / "index.md").read_text(encoding="utf-8")
        self.assertIn("Experiments mae", index)
        self.assertIn("baseline", (charts / "experiments-val.svg").read_text(encoding="utf-8"))
        self.assertIn("tuned-candidate", (charts / "candidates-val.svg").read_text(encoding="utf-8"))
        self.assertIn("main-result", (charts / "claims-test.svg").read_text(encoding="utf-8"))

    def test_json_summary(self) -> None:
        code, output = self._run("--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(
            [item["id"] for item in payload],
            ["experiments-val", "candidates-val", "claims-test"],
        )

    def test_stale_svg_is_removed(self) -> None:
        charts = self.output / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        stale = charts / "old-view.svg"
        stale.write_text("<svg/>", encoding="utf-8")
        code, _ = self._run()
        self.assertEqual(code, 0)
        self.assertFalse(stale.exists())

    def test_missing_metric_renders_empty_state(self) -> None:
        self.views.write_text(
            "charts:\n"
            "  - id: missing\n"
            "    source: experiments\n"
            "    metric: does_not_exist\n",
            encoding="utf-8",
        )
        code, _ = self._run()
        self.assertEqual(code, 0)
        svg = (self.output / "charts" / "missing.svg").read_text(encoding="utf-8")
        self.assertIn("no data for this view", svg)


if __name__ == "__main__":
    unittest.main()
