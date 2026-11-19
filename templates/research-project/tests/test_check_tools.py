from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import yaml

from core.tools import audit_data, check

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_main(module, argv: list[str]) -> tuple[int, str]:
    output = io.StringIO()
    previous = sys.argv
    sys.argv = [module.__name__, *argv]
    try:
        with redirect_stdout(output):
            code = module.main()
    finally:
        sys.argv = previous
    return code, output.getvalue()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class CheckToolTests(unittest.TestCase):
    def test_template_fails_the_initialization_gate(self) -> None:
        code, output = run_main(
            check,
            [
                "--project",
                str(ROOT / "PROJECT.yaml"),
                "--config",
                str(ROOT / "input" / "configs" / "base.yaml"),
                "--json",
            ],
        )
        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertTrue(
            any("project" in error for error in payload["errors"]),
            payload["errors"],
        )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class AuditToolTests(unittest.TestCase):
    def test_synthetic_template_audit_runs(self) -> None:
        code, output = run_main(
            audit_data,
            [
                "--config",
                str(ROOT / "input" / "configs" / "base.yaml"),
                "--project",
                str(ROOT / "PROJECT.yaml"),
                "--limit",
                "1",
                "--json",
            ],
        )
        payload = json.loads(output)
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(set(payload["splits"]), {"train", "val", "test"})


class CardAuditTests(unittest.TestCase):
    """Provenance cards: dataset card, weights card, historical versions."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._old_root = os.environ.get("DLEF_DATA_ROOT")
        os.environ["DLEF_DATA_ROOT"] = str(self.root / "data")
        (self.root / "data").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self._old_root is None:
            os.environ.pop("DLEF_DATA_ROOT", None)
        else:
            os.environ["DLEF_DATA_ROOT"] = self._old_root
        self._tmp.cleanup()

    def _config(self, *, name="ds", version="v1", init="") -> SimpleNamespace:
        return SimpleNamespace(
            data=SimpleNamespace(name=name, version=version),
            model=SimpleNamespace(init_checkpoint=init),
        )

    def test_missing_dataset_card_is_an_error(self) -> None:
        errors: list[str] = []
        audit_data._audit_cards(self._config(), errors, [], input_root=self.root)
        self.assertTrue(any("dataset card missing" in item for item in errors), errors)

    def test_weights_card_missing_is_an_error(self) -> None:
        weights = self.root / "data" / "warm.pt"
        weights.write_bytes(b"warm")
        errors: list[str] = []
        audit_data._audit_cards(
            self._config(init="warm.pt"), errors, [], input_root=self.root
        )
        self.assertTrue(any("weights card missing" in item for item in errors), errors)

    def test_weights_card_with_matching_hash_passes(self) -> None:
        dataset_cards = self.root / "input" / "datasets" / "ds"
        dataset_cards.mkdir(parents=True, exist_ok=True)
        (dataset_cards / "v1.yaml").write_text(
            "name: ds\nversion: v1\n", encoding="utf-8"
        )
        weights = self.root / "data" / "warm.pt"
        weights.write_bytes(b"warm")
        cards = self.root / "input" / "weights"
        cards.mkdir(parents=True, exist_ok=True)
        (cards / "warm.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "warm",
                    "path": "warm.pt",
                    "sha256": hashlib.sha256(b"warm").hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        errors: list[str] = []
        audit_data._audit_cards(
            self._config(init="warm.pt"), errors, [], input_root=self.root
        )
        self.assertEqual(errors, [])

    def test_run_versions_without_cards_warn(self) -> None:
        runs = self.root / "runs" / "p" / "e" / "r1"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "manifest.json").write_text(
            json.dumps({"data_versions": {"train": "ds@v1"}}), encoding="utf-8"
        )
        warnings: list[str] = []
        audit_data._audit_run_data_versions(
            self.root / "runs", warnings, input_root=self.root
        )
        self.assertTrue(any("ds@v1" in item for item in warnings), warnings)


if __name__ == "__main__":
    unittest.main()
