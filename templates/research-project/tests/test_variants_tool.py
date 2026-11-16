"""Variant scanner: declared structures in configs, observed structures in runs.

The configs view must not count a size sweep as a variant; the runs view must
catch one config digest producing two structures (in-place model edits).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dlef.run import ArtifactStore, RunStore

from core.tools import variants as variants_tool


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class VariantsConfigViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.configs = Path(self._tmp.name) / "configs"
        _write(
            self.configs / "base.yaml",
            "experiment: e1\nmodel:\n  name: mlp\n  hidden: 16\n",
        )
        # A width sweep: same architecture, so it must stay one signature.
        _write(
            self.configs / "experiments" / "e1" / "wide.yaml",
            "experiment: e1\nmodel:\n  name: mlp\n  hidden: 64\n",
        )
        _write(
            self.configs / "experiments" / "e1" / "deep.yaml",
            "experiment: e1\nmodel:\n  name: mlp\n  hidden: 128\n",
        )
        # A different topology selector: this is a second variant.
        _write(
            self.configs / "experiments" / "e1" / "lin.yaml",
            "experiment: e1\nmodel:\n  name: linear\n",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_size_sweep_collapses_into_one_variant(self) -> None:
        payload = variants_tool.collect_configs(self.configs)
        groups = {group["model"]: group for group in payload["groups"]}

        self.assertEqual(payload["total_configs"], 4)
        self.assertEqual(payload["total_variants"], 2)
        self.assertEqual(groups["mlp"]["configs"], 3)
        self.assertEqual(groups["mlp"]["variants"], 1)
        self.assertEqual(groups["linear"]["configs"], 1)
        self.assertEqual(groups["linear"]["variants"], 1)

    def test_baseline_is_the_most_common_signature(self) -> None:
        payload = variants_tool.collect_configs(self.configs)
        groups = {group["model"]: group for group in payload["groups"]}
        baseline = [row for row in groups["mlp"]["rows"] if row["is_baseline"]]
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["configs"], 3)
        self.assertEqual(baseline[0]["diff"], {})

    def test_invalid_config_is_reported_not_fatal(self) -> None:
        _write(
            self.configs / "experiments" / "e1" / "broken.yaml",
            "experiment: e1\nmodel:\n  name: nope\n",
        )
        payload = variants_tool.collect_configs(self.configs)
        self.assertEqual(len(payload["invalid"]), 1)
        self.assertIn("broken.yaml", payload["invalid"][0]["config"])
        self.assertEqual(payload["total_configs"], 4)
        self.assertEqual(variants_tool._check(payload, 3), 1)

    def test_signature_source_is_declared_method(self) -> None:
        payload = variants_tool.collect_configs(self.configs)
        self.assertEqual(
            payload["signature_source"], "ModelConfig.structural_signature()"
        )

    def test_missing_dir_yields_empty_payload(self) -> None:
        payload = variants_tool.collect_configs(Path(self._tmp.name) / "absent")
        self.assertEqual(payload["total_configs"], 0)
        self.assertEqual(payload["groups"], [])


class VariantsRunsViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runs_root = root / "runs"
        store = RunStore(self.runs_root, ArtifactStore(root / "artifacts"))

        # Same config digest, two structures: the model was edited in place.
        self._make(store, "a", seed=1, config_digest="c" * 64, structure="s1", topology="t1")
        self._make(store, "b", seed=2, config_digest="c" * 64, structure="s2", topology="t1")
        # A genuinely different config, no drift.
        self._make(store, "c", seed=3, config_digest="d" * 64, structure="s1", topology="t1")

        # A run that predates the fingerprint.
        legacy = store.create(
            project="p", experiment="e", config_digest="e" * 64, seed=4
        )
        legacy.manifest.status = "completed"
        legacy.write_manifest()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make(
        self,
        store: RunStore,
        label: str,
        *,
        seed: int,
        config_digest: str,
        structure: str,
        topology: str,
    ) -> None:
        run = store.create(project="p", experiment="e", config_digest=config_digest, seed=seed)
        run.manifest.status = "completed"
        run.write_manifest()
        run.save_json("config.resolved.json", {"model": {"name": "mlp"}})
        run.save_json(
            "model_structure.json",
            {
                "digest": structure,
                "topology_digest": topology,
                "n_params": 17,
                "n_trainable": 17,
                "n_layers": 4,
            },
        )

    def test_counts_structures_and_topologies(self) -> None:
        payload = variants_tool.collect_runs(self.runs_root)
        self.assertEqual(payload["total_runs"], 3)
        group = payload["groups"][0]
        self.assertEqual(group["model"], "mlp")
        self.assertEqual(group["runs"], 3)
        self.assertEqual(group["structures"], 2)
        self.assertEqual(group["topologies"], 1)

    def test_detects_drift_on_shared_config_digest(self) -> None:
        payload = variants_tool.collect_runs(self.runs_root)
        self.assertEqual(len(payload["drift"]), 1)
        item = payload["drift"][0]
        self.assertEqual(item["config_digest"], "c" * 64)
        self.assertEqual({s["digest"] for s in item["structures"]}, {"s1", "s2"})
        self.assertEqual(variants_tool._check(payload, 3), 1)

    def test_reports_runs_without_fingerprint(self) -> None:
        payload = variants_tool.collect_runs(self.runs_root)
        self.assertEqual(len(payload["runs_without_fingerprint"]), 1)

    def test_no_drift_when_every_config_has_one_structure(self) -> None:
        self._make(
            RunStore(self.runs_root),
            "d",
            seed=9,
            config_digest="f" * 64,
            structure="s9",
            topology="t9",
        )
        payload = variants_tool.collect_runs(self.runs_root)
        self.assertEqual(payload["total_runs"], 4)
        self.assertEqual(len(payload["drift"]), 1)  # still only the c*64 pair
        self.assertEqual(payload["drift"][0]["config_digest"], "c" * 64)


class VariantsCheckTests(unittest.TestCase):
    def test_configs_view_fails_above_max_variants(self) -> None:
        payload = {
            "view": "configs",
            "invalid": [],
            "groups": [
                {"model": "v11", "configs": 56, "variants": 22},
                {"model": "mlp", "configs": 3, "variants": 1},
            ],
        }
        self.assertEqual(variants_tool._check(payload, 3), 1)
        self.assertEqual(variants_tool._check(payload, 22), 0)

    def test_runs_view_passes_without_drift(self) -> None:
        self.assertEqual(
            variants_tool._check({"view": "runs", "drift": []}, 3), 0
        )


class VariantsCliTests(unittest.TestCase):
    def _call(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        previous = sys.argv
        sys.argv = ["variants.py", *args]
        try:
            with redirect_stdout(output):
                code = variants_tool.main()
        finally:
            sys.argv = previous
        return code, output.getvalue()

    def test_configs_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            configs = Path(tmp) / "configs"
            _write(configs / "base.yaml", "experiment: e\nmodel:\n  name: mlp\n")
            code, out = self._call("configs", "--configs-dir", str(configs), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["view"], "configs")
        self.assertEqual(payload["total_configs"], 1)
        self.assertEqual(payload["groups"][0]["model"], "mlp")

    def test_runs_text_view_prints_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._call("runs", "--runs", str(Path(tmp) / "runs"))
        self.assertEqual(code, 0)
        self.assertIn("runs_root:", out)
        self.assertIn("total: 0 runs", out)


if __name__ == "__main__":
    unittest.main()
