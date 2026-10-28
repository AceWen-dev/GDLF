from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dlef.config import ConfigError, load_config, resolve_config
from examples.toy.config import ToyConfig


class ConfigTests(unittest.TestCase):
    def test_resolve_and_digest(self) -> None:
        config = resolve_config(ToyConfig, {"seed": 3, "epochs": 4, "lr": 0.1})
        self.assertEqual(config.schema.seed, 3)
        self.assertEqual(config.schema.epochs, 4)
        self.assertEqual(len(config.digest), 64)
        again = resolve_config(ToyConfig, {"seed": 3, "epochs": 4, "lr": 0.1})
        self.assertEqual(config.digest, again.digest)
        different = resolve_config(ToyConfig, {"seed": 3, "epochs": 5, "lr": 0.1})
        self.assertNotEqual(config.digest, different.digest)

    def test_unknown_key_is_an_error(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_config(ToyConfig, {"seed": 1, "nope": 2})

    def test_type_error(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_config(ToyConfig, {"epochs": "many"})

    def test_override(self) -> None:
        config = resolve_config(ToyConfig, {"seed": 1, "lr": 0.1}, [{"lr": 0.2}])
        self.assertEqual(config.schema.lr, 0.2)

    def test_validation(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_config(ToyConfig, {"lr": 0.0})

    def test_json_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"seed": 9, "epochs": 2, "lr": 0.05}), encoding="utf-8")
            config = load_config(path, ToyConfig)
            self.assertEqual(config.schema.seed, 9)
            self.assertEqual(config.source, str(path))


if __name__ == "__main__":
    unittest.main()
