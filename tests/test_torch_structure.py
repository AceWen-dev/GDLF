"""Structure fingerprint: golden shape, sensitivity boundaries, run-dir record.

The two digests have different jobs and the tests pin that difference down:
``digest`` moves on any change (including width sweeps and freezing), while
``topology_digest`` moves only when the module set or its wiring changes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    from torch import nn

    from adapters.torch import StructureCallback, structure_fingerprint, structure_layers
    from dlef.contracts import RunContext

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _net(width: int = 4, activation: bool = True) -> "nn.Module":
    layers: list = [nn.Linear(2, width)]
    if activation:
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _context(run_dir: str = "") -> "RunContext":
    return RunContext(run_id="run-1", config=None, manifest=None, run_dir=run_dir)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class StructureFingerprintTests(unittest.TestCase):
    def test_records_root_layers_shapes_and_counts(self) -> None:
        fingerprint = structure_fingerprint(_net())
        # root + Linear + ReLU + Linear
        self.assertEqual(fingerprint["n_layers"], 4)
        # Linear(2,4) = 12, Linear(4,1) = 5
        self.assertEqual(fingerprint["n_params"], 17)
        self.assertEqual(fingerprint["n_trainable"], 17)
        self.assertEqual(
            [layer["type"] for layer in fingerprint["layers"]],
            ["Sequential", "Linear", "ReLU", "Linear"],
        )
        self.assertEqual(fingerprint["layers"][0]["name"], "<root>")
        self.assertEqual(fingerprint["layers"][1]["name"], "0")
        self.assertEqual(fingerprint["layers"][1]["shapes"], [[4, 2], [4]])
        self.assertEqual(len(fingerprint["digest"]), 64)

    def test_parameterless_modules_are_recorded(self) -> None:
        # A ReLU owns no parameters; dropping it from the record would hide a
        # real topology change.
        layers = structure_layers(_net())
        activation = [layer for layer in layers if layer["type"] == "ReLU"]
        self.assertEqual(len(activation), 1)
        self.assertNotIn("shapes", activation[0])
        self.assertNotIn("frozen", activation[0])

    def test_digest_is_deterministic_across_instances(self) -> None:
        self.assertEqual(
            structure_fingerprint(_net())["digest"],
            structure_fingerprint(_net())["digest"],
        )
        self.assertEqual(
            structure_fingerprint(_net())["topology_digest"],
            structure_fingerprint(_net())["topology_digest"],
        )

    def test_width_sweep_moves_digest_but_keeps_topology(self) -> None:
        narrow = structure_fingerprint(_net(width=4))
        wide = structure_fingerprint(_net(width=8))
        self.assertNotEqual(narrow["digest"], wide["digest"])
        self.assertEqual(narrow["topology_digest"], wide["topology_digest"])

    def test_inserted_activation_moves_topology_without_moving_params(self) -> None:
        plain = structure_fingerprint(_net(activation=False))
        activated = structure_fingerprint(_net(activation=True))
        self.assertEqual(plain["n_params"], activated["n_params"])
        self.assertNotEqual(plain["topology_digest"], activated["topology_digest"])
        self.assertNotEqual(plain["digest"], activated["digest"])

    def test_freezing_moves_digest_and_trainable_count_only(self) -> None:
        module = _net(width=4)
        before = structure_fingerprint(module)
        for param in module[0].parameters():
            param.requires_grad = False
        after = structure_fingerprint(module)

        self.assertNotEqual(before["digest"], after["digest"])
        self.assertEqual(before["topology_digest"], after["topology_digest"])
        self.assertEqual(after["n_params"], before["n_params"])
        self.assertEqual(after["n_trainable"], before["n_trainable"] - 12)
        self.assertTrue(after["layers"][1]["frozen"])
        self.assertFalse(after["layers"][3]["frozen"])

    def test_callback_writes_fingerprint_into_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = StructureCallback(_net())
            callback.on_run_start(_context(tmp))

            path = Path(tmp) / "model_structure.json"
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["digest"], callback.fingerprint["digest"])
            self.assertEqual(payload["n_layers"], 4)

    def test_callback_without_run_dir_still_exposes_fingerprint(self) -> None:
        callback = StructureCallback(_net())
        callback.on_run_start(_context(""))
        self.assertIsNotNone(callback.fingerprint)
        self.assertIn("digest", callback.fingerprint)

    def test_callback_honours_custom_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            StructureCallback(_net(), filename="net.json").on_run_start(_context(tmp))
            self.assertTrue((Path(tmp) / "net.json").is_file())


if __name__ == "__main__":
    unittest.main()
