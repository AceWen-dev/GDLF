"""TensorBoard callback: an optional mirror, not the experiment record.

RunStore / ``metrics.jsonl`` / ``logs/train.log`` remain the source of truth.
This callback writes scalar series, text and hyperparameters under
``<run_dir>/<tensorboard.subdir>/`` so TensorBoard can show the run at a glance.

The callback is task-agnostic. The project supplies metric names and, through
``StepOutput.metrics`` / ``StepOutput.meta``, optional scalar values; the core
only knows generic lifecycle and scalar tags.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dlef.contracts import CallbackBase, RunContext, StepOutput, to_float


def _scalar(value: Any) -> float | None:
    """Return a finite-ish scalar float, or None for tensors/lists/objects."""
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a config tree into TensorBoard hparam-friendly values."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value, ensure_ascii=False, default=str)}
    if value is None:
        return {prefix: "none"}
    if isinstance(value, (str, int, float, bool)):
        return {prefix: value}
    return {prefix: str(value)}


class TensorBoardCallback(CallbackBase):
    def __init__(
        self,
        *,
        subdir: str = "tensorboard",
        flush_secs: int = 10,
        log_hparams: bool = True,
    ) -> None:
        self.subdir = subdir
        self.flush_secs = flush_secs
        self.log_hparams = log_hparams
        self.writer: Any | None = None

    def on_run_start(self, ctx: RunContext) -> None:
        if not ctx.run_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print(
                "warning: tensorboard is enabled but not installed; "
                "run `pip install tensorboard` or set tensorboard.enabled=false",
                file=sys.stderr,
            )
            return

        log_dir = Path(ctx.run_dir) / self.subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=self.flush_secs)
        self.writer.add_text(
            "run/config",
            json.dumps(ctx.config.to_dict(), indent=2, ensure_ascii=False, default=str),
            0,
        )
        if ctx.manifest is not None:
            self.writer.add_text(
                "run/manifest",
                json.dumps(
                    ctx.manifest.to_dict(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                0,
            )

    def on_step_end(self, ctx: RunContext, step: StepOutput) -> None:
        if self.writer is None:
            return
        if step.loss is not None:
            self.writer.add_scalar("train/loss", to_float(step.loss), ctx.step)
        for name, value in step.terms.items():
            scalar = _scalar(value)
            if scalar is not None:
                self.writer.add_scalar(f"train/loss/{name}", scalar, ctx.step)

        # Generic scalars supplied by the Runner or by the project task.
        for name, value in ctx.step_metrics.items():
            scalar = _scalar(value)
            if scalar is not None:
                self.writer.add_scalar(f"train/{name}", scalar, ctx.step)
        for name, value in step.metrics.items():
            scalar = _scalar(value)
            if scalar is not None:
                self.writer.add_scalar(f"train/step/{name}", scalar, ctx.step)
        for name, value in step.meta.items():
            scalar = _scalar(value)
            if scalar is not None:
                self.writer.add_scalar(f"train/meta/{name}", scalar, ctx.step)

    def on_epoch_end(self, ctx: RunContext) -> None:
        if self.writer is None:
            return
        for name, value in ctx.metrics.items():
            scalar = _scalar(value)
            if scalar is not None:
                self.writer.add_scalar(f"val/{name}", scalar, ctx.epoch)
        for name, value in ctx.epoch_metrics.items():
            scalar = _scalar(value)
            if scalar is not None:
                self.writer.add_scalar(f"train/epoch/{name}", scalar, ctx.epoch)
        for name, value in ctx.selection.items():
            scalar = _scalar(value)
            if scalar is not None:
                self.writer.add_scalar(f"select/{name}", scalar, ctx.epoch)
            elif isinstance(value, str):
                self.writer.add_text(f"select/{name}", value, ctx.epoch)

    def on_run_end(self, ctx: RunContext) -> None:
        if self.writer is None:
            return
        if self.log_hparams:
            config = getattr(ctx.config, "to_dict", None)
            if callable(config):
                hparams = _flatten(config())
                metrics = {
                    name: value
                    for name, value in ctx.metrics.items()
                    if _scalar(value) is not None
                }
                if hparams:
                    try:
                        self.writer.add_hparams(hparams, metrics)
                    except (TypeError, ValueError):
                        # HParams is a convenience view; never fail a run for it.
                        pass
        self._close()

    def on_exception(self, ctx: RunContext, error: BaseException) -> None:
        if self.writer is not None:
            self.writer.add_text("run/error", repr(error), ctx.epoch)
        self._close()

    def _close(self) -> None:
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()
        self.writer = None


def log_eval_record(config: Any, run: Any, record: Any) -> None:
    """Mirror one EvalRecord into its eval-run TensorBoard directory.

    Test eval remains a separate run; TensorBoard simply displays its scalars
    beside training runs when both live under the same ``--logdir``.
    """
    tensorboard = getattr(config, "tensorboard", None)
    if tensorboard is None or not getattr(tensorboard, "enabled", False):
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return

    log_dir = Path(run.dir) / tensorboard.subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=tensorboard.flush_secs)
    try:
        writer.add_text(
            "eval/record",
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False, default=str),
            0,
        )
        writer.add_text("eval/eval_id", str(record.eval_id), 0)
        writer.add_text("eval/split", str(record.split), 0)
        writer.add_text("eval/benchmark_id", str(record.benchmark_id), 0)
        writer.add_text("eval/test_set_id", str(record.test_set_id), 0)
        writer.add_text("eval/checkpoint_digest", str(record.checkpoint_digest), 0)
        writer.add_scalar("eval/test_access", float(record.test_access), 0)
        for name, value in record.metrics.items():
            scalar = _scalar(value)
            if scalar is not None:
                writer.add_scalar(f"eval/{record.split}/{name}", scalar, 0)
    finally:
        writer.flush()
        writer.close()
