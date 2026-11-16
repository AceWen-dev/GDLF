"""回归测试：TorchModel 必须暴露 train/eval/training（D-0041）。

否则 runner._validate 的 `hasattr(model, "eval")` 为 False，验证/评估会一直
停留在 train 模式（BN 用批统计）。
"""

from __future__ import annotations

import unittest

try:
    import torch

    from adapters.torch.components import TorchModel
    from dlef.contracts import ModelOutput

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchModelModeTests(unittest.TestCase):
    def _model(self) -> TorchModel:
        module = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.BatchNorm1d(1))
        return TorchModel(module, lambda m, b: ModelOutput(predictions=b["x"][:, :1]))

    def test_eval_and_train_toggle_training_flag(self) -> None:
        model = self._model()
        self.assertTrue(model.training)
        model.eval()
        self.assertFalse(model.training)
        model.train()
        self.assertTrue(model.training)

    def test_bn_uses_running_stats_in_eval(self) -> None:
        model = self._model()
        bn = model.module[1]
        model.eval()
        self.assertFalse(bn.training)
        model.train()
        self.assertTrue(bn.training)


if __name__ == "__main__":
    unittest.main()
