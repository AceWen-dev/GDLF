from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dlef.cli import main


class InitTests(unittest.TestCase):
    def test_init_copies_template_and_fills_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "demo"
            code = main(["init", str(dest), "--project", "demo", "--no-git"])
            self.assertEqual(code, 0)
            self.assertTrue((dest / "PROJECT.yaml").is_file())
            self.assertTrue((dest / "STATE.md").is_file())
            self.assertTrue((dest / "DECISIONS.md").is_file())
            self.assertFalse((dest / "TUTORIAL.md").exists())
            self.assertTrue((dest / "input" / "configs" / "base.yaml").is_file())

            project = (dest / "PROJECT.yaml").read_text(encoding="utf-8")
            self.assertIn('project: "demo"', project)
            self.assertNotIn("<project-id>", project)
            self.assertNotIn('dlef_commit: "<sha>"', project)

            state = (dest / "STATE.md").read_text(encoding="utf-8")
            self.assertIn('project: "demo"', state)
            self.assertNotIn("updated: YYYY-MM-DD", state)

            self.assertIn(
                'name = "demo"', (dest / "pyproject.toml").read_text(encoding="utf-8")
            )
            self.assertIn(
                "project: demo",
                (dest / "input" / "configs" / "base.yaml").read_text(encoding="utf-8"),
            )
            self.assertFalse(any(dest.rglob("__pycache__")))

    def test_init_refuses_non_empty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "demo"
            dest.mkdir()
            (dest / "keep.txt").write_text("do not overwrite", encoding="utf-8")
            code = main(["init", str(dest), "--project", "demo", "--no-git"])
            self.assertEqual(code, 2)
            self.assertTrue((dest / "keep.txt").is_file())
            self.assertFalse((dest / "PROJECT.yaml").exists())


if __name__ == "__main__":
    unittest.main()
