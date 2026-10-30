from __future__ import annotations

import base64
import io
import unittest
import zipfile
from pathlib import Path

from dlef._template_data import TEMPLATE_B64

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "research-project"
KEYS = (
    "AGENTS.md",
    "PROJECT.yaml",
    "STATE.md",
    "DECISIONS.md",
    "TUTORIAL.md",
    "Makefile",
    "input/configs/base.yaml",
    "core/tools/check.py",
)


class EmbeddedTemplateTests(unittest.TestCase):
    def test_embedded_template_is_current(self) -> None:
        payload = base64.b64decode(TEMPLATE_B64)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            for key in KEYS:
                name = f"research-project/{key}"
                self.assertIn(name, names)
                self.assertEqual(
                    archive.read(name),
                    (TEMPLATE / key).read_bytes(),
                    f"{key} changed; run `make template-data`",
                )
            self.assertFalse(
                any("__pycache__" in name for name in names),
                "embedded template contains __pycache__",
            )
            self.assertFalse(
                any("output/test_access" in name for name in names),
                "embedded template contains generated test_access files",
            )


if __name__ == "__main__":
    unittest.main()
