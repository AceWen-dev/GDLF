from __future__ import annotations

import unittest

from dlef.contracts import RunContext, StepOutput

from core.tensorboard import TensorBoardCallback


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.hparams: list[tuple[dict, dict]] = []
        self.closed = False

    def add_scalar(self, tag, value, step) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def add_text(self, tag, value, step) -> None:
        self.texts.append((tag, str(value), int(step)))

    def add_hparams(self, hparams, metrics) -> None:
        self.hparams.append((dict(hparams), dict(metrics)))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeConfig:
    def to_dict(self):
        return {
            "seed": 7,
            "model": {"name": "linear"},
            "metrics": ["mae"],
        }


class TensorBoardCallbackTests(unittest.TestCase):
    def test_step_and_epoch_scalars(self) -> None:
        callback = TensorBoardCallback()
        writer = FakeWriter()
        callback.writer = writer
        ctx = RunContext(
            run_id="r1",
            config=FakeConfig(),
            manifest=None,
            step=3,
            epoch=1,
            step_metrics={"lr": 0.01, "grad_norm": 2.5},
            epoch_metrics={"train_loss": 0.4},
            metrics={"mae": 0.2},
            selection={"best_raw_score": 0.2, "checkpoint": "best_raw"},
        )
        step = StepOutput(
            loss=0.4,
            terms={"mse": 0.4},
            metrics={"custom": 1.5},
            meta={"aux": 2.0, "vector": [1, 2]},
        )

        callback.on_step_end(ctx, step)
        callback.on_epoch_end(ctx)
        callback.on_run_end(ctx)

        tags = {tag for tag, _, _ in writer.scalars}
        self.assertIn("train/loss", tags)
        self.assertIn("train/loss/mse", tags)
        self.assertIn("train/lr", tags)
        self.assertIn("train/grad_norm", tags)
        self.assertIn("train/step/custom", tags)
        self.assertIn("train/meta/aux", tags)
        self.assertNotIn("train/meta/vector", tags)
        self.assertIn("val/mae", tags)
        self.assertIn("train/epoch/train_loss", tags)
        self.assertIn("select/best_raw_score", tags)
        self.assertIn(("select/checkpoint", "best_raw", 1), writer.texts)
        self.assertTrue(writer.hparams)
        self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
