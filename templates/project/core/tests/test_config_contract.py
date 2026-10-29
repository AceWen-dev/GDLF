from __future__ import annotations

import unittest
from pathlib import Path

try:
    from dlef.config import load_config
    from core.config import ProjectConfig
    from core.data.build import DATASETS, TRANSFORMS
    from core.data.splits import SPLITTERS
    from core.losses.build import BUILDERS as LOSSES
    from core.metrics.build import BUILDERS as METRICS
    from core.models.build import BUILDERS as MODELS
    from core.optimizers.build import BUILDERS as OPTIMIZERS
    from core.schedulers.build import BUILDERS as SCHEDULERS

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ConfigContractTests(unittest.TestCase):
    def test_every_selector_resolves_to_a_builder(self) -> None:
        config = load_config(str(ROOT / "input" / "configs" / "base.yaml"), ProjectConfig).schema

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
