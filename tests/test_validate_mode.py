"""回归测试：Runner._validate 必须以 eval 模式运行模型并恢复 train（D-0023）。

此前 _validate 从不切换模式，BN 网络的验证一直在 train 模式下进行
（批统计 + running stats 被验证数据污染）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dlef.contracts import Metric, MetricResult, StepOutput
from dlef.policy import NoOpTrainingPolicy
from dlef.run import ArtifactStore, RunStore
from dlef.runner import Runner


class RecordingModel:
    def __init__(self) -> None:
        self.training = True
        self.modes: list[bool] = []

    def train(self, mode: bool = True) -> "RecordingModel":
        self.training = mode
        return self

    def eval(self) -> "RecordingModel":
        return self.train(False)


class RecordingTask:
    """只实现 _validate 用到的最小面。"""

    def __init__(self) -> None:
        self.model = RecordingModel()
        self.modes = self.model.modes

    def get_model(self) -> RecordingModel:
        return self.model

    def val_step(self, batch):
        self.modes.append(self.model.training)
        return StepOutput(loss=None, predictions=[0.0])


class MeanMetric(Metric):
    def reset(self) -> None:
        self.values_list: list[float] = []

    def update(self, output, batch) -> None:
        self.values_list.extend(float(p) for p in output.predictions)

    def compute(self) -> MetricResult:
        mean = sum(self.values_list) / len(self.values_list) if self.values_list else 0.0
        return MetricResult(values={"mae": mean})


class ValidateModeTests(unittest.TestCase):
    def test_validate_uses_eval_mode_and_restores_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs", ArtifactStore(root / "artifacts"))
            runner = Runner(store, store.artifact_store)
            task = RecordingTask()
            values = runner._validate(
                task, [{}, {}, {}], {"mae": MeanMetric()}, NoOpTrainingPolicy()
            )
            self.assertEqual(values["mae"], 0.0)
            # 验证期间必须是 eval 模式
            self.assertEqual(task.modes, [False, False, False])
            # 验证结束后恢复 train（下一轮训练不受影响）
            self.assertTrue(task.model.training)


if __name__ == "__main__":
    unittest.main()
