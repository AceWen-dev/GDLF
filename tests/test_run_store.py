from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dlef.run import ArtifactStore, RunStore


class RunStoreTests(unittest.TestCase):
    def test_create_log_artifact_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            run = store.create(
                project="p",
                experiment="e",
                config_digest="a" * 64,
                seed=3,
                config_source="configs/base.yaml",
            )
            self.assertEqual(run.manifest.status, "running")
            self.assertEqual(run.manifest.config_source, "configs/base.yaml")
            self.assertTrue(run.manifest_path.is_file())

            run.log("hello from the run")
            self.assertTrue(run.log_path.is_file())
            self.assertIn("hello from the run", run.log_path.read_text(encoding="utf-8"))

            run.log_metrics({"step": 1, "loss": 0.5})
            run.log_metrics({"step": 2, "loss": 0.4})
            lines = run.metrics_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["step"], 1)

            payload = root / "artifact.txt"
            payload.write_text("hello", encoding="utf-8")
            digest = run.save_artifact(payload, "artifact")
            self.assertEqual(len(digest), 64)
            self.assertTrue(store.artifact_store.exists(digest, "artifact"))
            self.assertIn("artifact", run.manifest.artifacts)

            run.finalize("completed")
            self.assertEqual(run.manifest.status, "completed")
            self.assertIsNotNone(run.manifest.ended_at)

            listed = store.list()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].run_id, run.run_id)

            reopened = store.open(run.dir)
            self.assertEqual(reopened.manifest.run_id, run.run_id)


if __name__ == "__main__":
    unittest.main()
