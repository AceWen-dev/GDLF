from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dlef.run import RunStore

from core.tools import digests as digests_tool


class DigestsToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runs_root = root / "runs"
        store = RunStore(self.runs_root)
        run = store.create(
            project="p",
            experiment="e",
            config_digest="a" * 64,
            seed=1,
        )
        ckpt = run.dir / "checkpoints" / "last.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"checkpoint-bytes")
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        (run.dir / "checkpoints" / "digests.json").write_text(
            json.dumps({"last.pt": digest}), encoding="utf-8"
        )
        self.run = run
        self.digest = digest
        run.save_json(
            "evals/eval-1.json",
            {
                "eval_id": "eval-1",
                "split": "val",
                "checkpoint_digest": digest,
                "data_version": "v",
                "eval_spec_digest": "b" * 64,
                "metrics": {"mae": 0.5},
            },
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _call(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        previous = sys.argv
        sys.argv = ["digests.py", "--runs", str(self.runs_root), *args]
        try:
            with redirect_stdout(output):
                code = digests_tool.main()
        except SystemExit as exc:  # argparse usage errors
            return int(exc.code or 2), output.getvalue()
        finally:
            sys.argv = previous
        return code, output.getvalue()

    def test_find_resolves_run_file_and_evals(self) -> None:
        code, output = self._call("--find", self.digest[:10])
        self.assertEqual(code, 0, output)
        self.assertIn("last.pt", output)
        self.assertIn(self.run.run_id, output)
        self.assertIn("eval-1", output)

    def test_run_lists_digests(self) -> None:
        code, output = self._call("--run", self.run.run_id)
        self.assertEqual(code, 0, output)
        self.assertIn("last.pt", output)
        self.assertIn(self.digest[:16], output)

    def test_find_without_match(self) -> None:
        code, output = self._call("--find", "ffffffffffff")
        self.assertEqual(code, 1)
        self.assertIn("no checkpoint matches", output)

    def test_find_rejects_short_prefix(self) -> None:
        code, _ = self._call("--find", "abc")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
