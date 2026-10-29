from __future__ import annotations

import unittest
from pathlib import Path

try:
    from dlef.config import load_config
    from myproject.config import ProjectConfig
    from myproject.data.build import DATASETS, TRANSFORMS
    from myproject.data.splits import SPLITTERS
    from myproject.losses.build import BUILDERS as LOSSES
    from myproject.metrics.build import BUILDERS as METRICS
    from myproject.models.build import BUILDERS as MODELS
    from myproject.optimizers.build import BUILDERS as OPTIMIZERS
    from myproject.schedulers.build import BUILDERS as SCHEDULERS

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ConfigContractTests(unittest.TestCase):
    def test_every_selector_resolves_to_a_builder(self) -> None:
        config = load_config(str(ROOT / "configs" / "base.yaml"), ProjectConfig).schema

        self.assertIn(config.data.name, DATASETS)
        self.assertIn(config.data.split, SPLITTERS)
        self.assertIn(config.data.train_transform, TRANSFORMS)
        self.assertIn(config.data.eval_transform, TRANSFORMS)

        self.assertIn(config.model.name, MODELS)
        self.assertIn(config.loss.name, LOSSES)
        self.assertIn(config.optim.name, OPTIMIZERS)
        self.assertIn(config.scheduler.name, SCHEDULERS)
        for name in config.metrics:
            self.assertIn(name, METRICS)
            self.assertIn(name, config.benchmark.metric_versions)


if __name__ == "__main__":
    unittest.main()
