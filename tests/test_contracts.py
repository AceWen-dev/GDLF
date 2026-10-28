from __future__ import annotations

import random
import unittest

from dlef.contracts import (
    Callback,
    CallbackBase,
    Dataset,
    Loss,
    Metric,
    Model,
    ModelOutput,
    Optimizer,
    Task,
    to_float,
    validate_task,
)
from examples.toy.components import LinearModel, MAEMetric, MSELoss, SGD, ToyDataset
from examples.toy.config import ToyConfig
from examples.toy.task import ToyTask


class ContractTests(unittest.TestCase):
    def test_structural_checks(self) -> None:
        rng = random.Random(0)
        self.assertIsInstance(ToyDataset(8, 4, 0.1, 1), Dataset)
        self.assertIsInstance(LinearModel(rng), Model)
        self.assertIsInstance(MSELoss(), Loss)
        self.assertIsInstance(SGD([], 0.1), Optimizer)
        self.assertIsInstance(MAEMetric(), Metric)
        self.assertIsInstance(ToyTask(ToyConfig()), Task)
        self.assertIsInstance(CallbackBase(), Callback)

    def test_validate_task(self) -> None:
        validate_task(ToyTask(ToyConfig()))
        with self.assertRaises(TypeError):
            validate_task(object())

    def test_to_float(self) -> None:
        self.assertEqual(to_float(3), 3.0)
        self.assertEqual(to_float(2.5), 2.5)

    def test_metric_golden(self) -> None:
        metric = MAEMetric()
        metric.update(
            ModelOutput(predictions=[1.0, 2.0]),
            {"y": [2.0, 4.0]},
        )
        self.assertAlmostEqual(metric.compute().values["mae"], 1.5)


if __name__ == "__main__":
    unittest.main()
