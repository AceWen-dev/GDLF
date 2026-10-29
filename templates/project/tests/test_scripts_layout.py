from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ALLOWED_AREAS = (
    "data/acquire",
    "data/prepare",
    "data/split",
    "analysis",
    "ops",
)
ROOT_FILES = {"__init__.py", "_bootstrap.py"}


class ScriptLayoutTests(unittest.TestCase):
    def test_scripts_live_in_purpose_areas(self) -> None:
        misplaced: list[str] = []
        for path in SCRIPTS.rglob("*.py"):
            if path.name in ROOT_FILES:
                continue
            relative = path.relative_to(SCRIPTS).as_posix()
            if not any(relative.startswith(area + "/") for area in ALLOWED_AREAS):
                misplaced.append(relative)
        self.assertFalse(
            misplaced,
            "scripts must live under scripts/{data/{acquire,prepare,split},analysis,ops}: "
            f"{misplaced}",
        )


if __name__ == "__main__":
    unittest.main()
