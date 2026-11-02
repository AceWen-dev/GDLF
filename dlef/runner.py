"""The Runner owns lifecycle, not domain math.

Standard path:
    task.train_step(batch) -> StepOutput(loss=...)
    policy.scale_loss(loss) -> policy.backward(...)
    policy.should_step(...) -> policy.optimizer_step(...)
    policy.zero_grad(...)

The policy seam carries AMP, gradient accumulation, clipping and DDP.
The default policy reproduces plain single-process training.
"""

from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import CheckpointCodec, JsonCheckpointCodec
from .claim import TestAccessLog
from .config import Config
from .contracts import (
    Batch,
    Callback,
    Checkpoint,
    DataSpec,
    Metric,
    MetricResult,
    ModelOutput,
    Optimizer,
    RunContext,
    Scheduler,
    StepOutput,
    Task,
    to_float,
    validate_task,
)
from .eval import EvalRecord, EvalSpec, eval_spec_digest
from .policy import NoOpTrainingPolicy, TrainingPolicy
from .run import ArtifactStore, Run, RunStore, sha256_file, utc_now
from .seed import capture_rng_state, restore_rng_state, seed_everything


class Runner:
    """Minimal, engine-agnostic train/eval loop."""

    def __init__(
        self,
        run_store: RunStore,
        artifact_store: ArtifactStore | None = None,
        checkpoint_codec: CheckpointCodec | None = None,
        training_policy: TrainingPolicy | None = None,
        test_access_log: TestAccessLog | None = None,
        ema_factory: Any | None = None,
    ) -> None:
        self.run_store = run_store
        self.artifact_store = artifact_store or run_store.artifact_store
        self.checkpoint_codec: CheckpointCodec = checkpoint_codec or JsonCheckpointCodec()
        self.training_policy: TrainingPolicy = training_policy or NoOpTrainingPolicy()
        self.test_access_log = test_access_log or TestAccessLog(
            run_store.root.parent / "test_access"
        )
        # EMA 的具体实现属于 torch 适配层；runner 只要求它提供
        # update/copy_to/state_dict 三个方法。这样核心仍然不 import torch。
        self.ema_factory = ema_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        task: Task,
        train_data: Any,
        val_data: Any | None,
        config: Config,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> Run | None:
        validate_task(task)
        schema = config.schema
        early = schema.early_stopping
        policy = self.training_policy
        policy.setup()
        # Safety net for direct Runner users. Entry points should call
        # seed_everything before building data/task so parameter init is also
        # reproducible; this call makes the training loop itself deterministic.
        seed_everything(schema.seed)

        run: Run | None = None
        if policy.is_main_process:
            run = self.run_store.create(
                project=schema.project,
                experiment=schema.experiment,
                config_digest=config.digest,
                seed=schema.seed,
                data_versions=_data_versions(train_data, val_data),
                tags={"framework": "dlef", **dict(tags or {})},
                config_source=config.source,
            )
            run.save_json("config.resolved.json", config.resolved)
            run.log("=" * 78)
            run.log(f"Run: {run.run_id}")
            run.log(f"Project/experiment: {schema.project} / {schema.experiment}")
            run.log(f"Seed: {schema.seed}  Config: {config.digest}")
            run.log(
                f"Config source: {config.source or '-'}  "
                f"Code: {run.manifest.code_version}"
            )
            run.log(
                f"Training: epochs={getattr(schema, 'epochs', '?')} "
                f"max_steps={getattr(schema, 'max_steps', '?')} "
                f"log_every={getattr(schema, 'log_every', '?')} "
                f"checkpoint_every={getattr(schema, 'checkpoint_every', '?')}"
            )
            run.log(
                f"Early stopping: enabled={early.enabled} monitor={early.monitor or '-'} "
                f"mode={early.mode} patience={early.patience} "
                f"min_delta={early.min_delta} min_epochs={early.min_epochs} "
                f"smooth_window={early.smooth_window}"
            )
            run.log(
                f"EMA: decay={getattr(schema, 'ema_decay', 0.0)} (0 = disabled)"
            )
            sel = getattr(schema, "selection", None)
            if sel is not None:
                run.log(
                    f"Selection: monitor={sel.monitor or early.monitor or '-'} "
                    f"mode={sel.mode if sel.monitor else early.mode} "
                    f"smooth_window={sel.smooth_window if sel.monitor else early.smooth_window} "
                    f"checkpoint={sel.checkpoint}"
                )
            for key in ("data", "model", "loss", "optim", "scheduler", "benchmark"):
                value = config.resolved.get(key)
                if value is not None:
                    run.log(
                        f"{key.capitalize()}: "
                        + json.dumps(
                            value,
                            sort_keys=True,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
            if "metrics" in config.resolved:
                run.log(f"Metrics: {config.resolved['metrics']}")
            run.log(f"Config file: {run.config_path}")
            run.log("=" * 78)

        model = task.get_model()
        optimizer = task.configure_optimizer(model)
        scheduler = task.configure_scheduler(optimizer)
        metrics = dict(task.configure_metrics())
        callbacks = list(task.configure_callbacks())
        ctx = RunContext(
            run_id=run.run_id if run is not None else "nonmain",
            config=schema,
            manifest=run.manifest if run is not None else None,
            run_dir=str(run.directory) if run is not None else "",
        )

        global_step = 0
        last_epoch = 0
        # Selection and early stopping are independent:
        #   selection        -> which checkpoint is best
        #   early_stopping   -> when to stop training
        selection = getattr(schema, "selection", None)
        selection_monitor = (
            selection.monitor if selection is not None and selection.monitor else early.monitor
        )
        selection_mode = (
            selection.mode if selection is not None and selection.monitor else early.mode
        )
        selection_window = max(
            1,
            int(
                selection.smooth_window
                if selection is not None and selection.monitor
                else early.smooth_window
            ),
        )
        selection_checkpoint = (
            selection.checkpoint if selection is not None else "best_raw"
        )
        raw_monitor = (
            selection_monitor[len("ema/") :]
            if selection_monitor.startswith("ema/")
            else selection_monitor
        )
        ema_monitor = (
            selection_monitor
            if selection_monitor.startswith("ema/")
            else f"ema/{raw_monitor}"
            if raw_monitor
            else ""
        )

        # Raw and EMA have separate best trajectories.
        best_raw_score: float | None = None
        best_raw_epoch = 0
        best_path: Path | None = None
        best_ema_score: float | None = None
        best_ema_epoch = 0
        best_ema_path: Path | None = None

        # Early stopping has its own history and checkpoint-independent score.
        best_score: float | None = None
        best_epoch = 0
        bad_epochs = 0
        early_stopped = False
        score_history: list[float] = []
        window = max(1, int(early.smooth_window))
        raw_history: list[float] = []
        ema_history: list[float] = []

        def improved(value: float, best: float, *, mode: str, min_delta: float = 0.0) -> bool:
            if mode == "min":
                return value < best - min_delta
            return value > best + min_delta

        ema = None
        if getattr(schema, "ema_decay", 0.0) > 0:
            if self.ema_factory is None:
                raise RuntimeError(
                    "ema_decay > 0 but this Runner has no ema_factory; "
                    "pass ema_factory=ModelEMA (adapters.torch.ema)"
                )
            ema = self.ema_factory(model, schema.ema_decay)
        try:
            policy.on_fit_start(task, optimizer, schema)
            self._call(callbacks, "on_run_start", ctx)
            for epoch in range(1, schema.epochs + 1):
                last_epoch = epoch
                ctx.epoch = epoch
                if run is not None:
                    run.log(f"Epoch(train) [{epoch}/{schema.epochs}] start")
                set_epoch = getattr(train_data, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)
                self._call(callbacks, "on_epoch_start", ctx)
                epoch_started = time.perf_counter()

                train_losses: list[float] = []
                grad_norms: list[float] = []
                reached_limit = False
                total_batches = _length(train_data)
                batch_index = 0
                for batch in train_data:
                    batch_index += 1
                    global_step += 1
                    ctx.step = global_step
                    step_started = time.perf_counter()
                    grad_norm: float | None = None
                    with policy.forward_context():
                        step = self._train_step(task, batch)

                    if step.loss is not None:
                        policy.backward(policy.scale_loss(step.loss))
                        train_losses.append(to_float(step.loss))
                        if policy.should_step(global_step):
                            policy.optimizer_step(task, optimizer)
                            policy.zero_grad(optimizer)
                            norm = getattr(policy, "last_grad_norm", None)
                            if norm is not None:
                                grad_norm = to_float(norm)
                                grad_norms.append(grad_norm)
                            if ema is not None:
                                ema.update(task.get_model())
                            if (
                                scheduler is not None
                                and getattr(scheduler, "step_unit", "epoch") == "step"
                            ):
                                scheduler.step()

                    should_log_step = (
                        global_step % schema.log_every == 0
                        if schema.log_every
                        else False
                    )
                    if (
                        schema.log_every
                        and (should_log_step or batch_index == total_batches)
                        and run is not None
                    ):
                        record = {
                            "phase": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": to_float(step.loss) if step.loss is not None else None,
                        }
                        record.update({key: to_float(value) for key, value in step.terms.items()})
                        run.log_metrics(record)
                        terms = " ".join(
                            f"{key}={to_float(value):.6g}"
                            for key, value in step.terms.items()
                        )
                        progress = (
                            f"{batch_index}/{total_batches}"
                            if total_batches is not None
                            else str(batch_index)
                        )
                        lr = _current_lr(optimizer)
                        lr_text = f" lr={lr:.4e}" if lr is not None else ""
                        run.log(
                            f"Epoch(train) [{epoch}/{schema.epochs}][{progress}]"
                            f"{lr_text} loss={record['loss']} {terms}".rstrip()
                        )

                    step_seconds = max(0.0, time.perf_counter() - step_started)
                    step_metrics: dict[str, float] = {
                        "step_time_ms": step_seconds * 1000.0,
                    }
                    step_lr = _current_lr(optimizer)
                    if step_lr is not None:
                        step_metrics["lr"] = step_lr
                    if grad_norm is not None:
                        step_metrics["grad_norm"] = grad_norm
                    batch_size = _batch_size(batch)
                    if batch_size is not None and step_seconds > 0:
                        step_metrics["samples_per_s"] = batch_size / step_seconds
                    ctx.step_metrics = step_metrics
                    self._call(callbacks, "on_step_end", ctx, step)

                    if schema.max_steps is not None and global_step >= schema.max_steps:
                        reached_limit = True
                        break

                # Flush a partial accumulation window at the epoch boundary.
                if policy.should_flush(global_step):
                    policy.optimizer_step(task, optimizer)
                    policy.zero_grad(optimizer)
                    if scheduler is not None and getattr(scheduler, "step_unit", "epoch") == "step":
                        scheduler.step()

                if scheduler is not None and getattr(scheduler, "step_unit", "epoch") == "epoch":
                    scheduler.step()

                val_metrics = (
                    self._validate(task, val_data, metrics, policy)
                    if val_data is not None
                    else {}
                )
                if ema is not None and val_data is not None:
                    # 同一套验证集再跑一遍 EMA 权重：原始权重与 EMA 各留一份指标，
                    # 选模可以显式指定用哪一个，避免“选了谁”说不清。
                    live_model = task.get_model()
                    # runner 不假设 state_dict 是 torch 张量：deepcopy 对
                    # float/list/tensor 一视同仁（旧代码的 .detach().clone()
                    # 让非 torch 的 Task 无法使用 EMA 验证）。
                    raw_state = deepcopy(dict(live_model.state_dict()))
                    ema.copy_to(live_model)
                    ema_metrics = self._validate(task, val_data, metrics, policy)
                    live_model.load_state_dict(raw_state)
                    val_metrics = dict(val_metrics)
                    val_metrics.update(
                        {f"ema/{key}": value for key, value in ema_metrics.items()}
                    )
                ctx.metrics = val_metrics
                epoch_seconds = max(0.0, time.perf_counter() - epoch_started)
                epoch_record = {
                    "phase": "val",
                    "epoch": epoch,
                    "train_loss": _mean(train_losses),
                    "train_batches": batch_index,
                    "duration_s": epoch_seconds,
                }
                if grad_norms:
                    ordered = sorted(grad_norms)
                    epoch_record["grad_norm_median"] = ordered[len(ordered) // 2]
                    epoch_record["grad_norm_max"] = ordered[-1]
                epoch_record.update(val_metrics)
                if run is not None:
                    run.log_metrics(epoch_record)
                    run.log(
                        f"Epoch(train) [{epoch}/{schema.epochs}] "
                        f"train_loss={epoch_record['train_loss']}"
                    )
                    if val_metrics:
                        val_text = " ".join(
                            f"{key}={value:.6g}" for key, value in val_metrics.items()
                        )
                        run.log(f"Epoch(val) [{epoch}/{schema.epochs}] {val_text}")

                # Selection is independent of early stopping.
                if not val_metrics:
                    if run is not None:
                        run.log("Selection: skipped (no val metrics)")
                elif not selection_monitor:
                    if run is not None:
                        run.log("Selection: skipped (no monitor configured)")
                elif selection_monitor not in val_metrics:
                    if run is not None:
                        run.log(
                            f"Selection: skipped (monitor {selection_monitor!r} "
                            "not in val metrics)"
                        )
                else:
                    # Raw weights: always track a best checkpoint when a raw
                    # monitor is available. This no longer depends on
                    # early_stopping.enabled.
                    if run is not None and raw_monitor in val_metrics:
                        raw_score = float(val_metrics[raw_monitor])
                        raw_history.append(raw_score)
                        raw_smooth = sum(raw_history[-selection_window:]) / len(
                            raw_history[-selection_window:]
                        )
                        if best_raw_score is None or improved(
                            raw_smooth,
                            best_raw_score,
                            mode=selection_mode,
                        ):
                            best_raw_score = raw_smooth
                            best_raw_epoch = epoch
                            best_path = self._save_checkpoint(
                                run,
                                task,
                                optimizer,
                                scheduler,
                                epoch,
                                global_step,
                                "best",
                                artifact=False,
                            )
                            run.log(
                                f"Selection(raw): {raw_monitor}={raw_smooth:.6g} "
                                f"(smoothed over {len(raw_history[-selection_window:])}, "
                                f"raw={raw_score:.6g}) at epoch {epoch}"
                            )

                    # EMA weights: save best_ema when explicitly selected, or
                    # when the legacy save_best_ema flag is set.
                    if (
                        run is not None
                        and ema is not None
                        and ema_monitor in val_metrics
                        and (
                            selection_checkpoint == "best_ema"
                            or bool(getattr(schema, "save_best_ema", False))
                        )
                    ):
                        ema_score = float(val_metrics[ema_monitor])
                        ema_history.append(ema_score)
                        ema_smooth = sum(ema_history[-selection_window:]) / len(
                            ema_history[-selection_window:]
                        )
                        if best_ema_score is None or improved(
                            ema_smooth,
                            best_ema_score,
                            mode=selection_mode,
                        ):
                            best_ema_score = ema_smooth
                            best_ema_epoch = epoch
                            best_ema_path = self._save_checkpoint(
                                run,
                                task,
                                optimizer,
                                scheduler,
                                epoch,
                                global_step,
                                "best_ema",
                                artifact=False,
                                model_state=ema.state_dict(),
                            )
                            run.log(
                                f"Selection(EMA): {ema_monitor}={ema_smooth:.6g} "
                                f"(smoothed over {len(ema_history[-selection_window:])}, "
                                f"raw={ema_score:.6g}) at epoch {epoch}"
                            )

                # Early stopping has its own monitor and patience; it does not
                # decide which checkpoint is selected.
                if early.enabled:
                    if not val_metrics:
                        if run is not None:
                            run.log("Early stopping: skipped (no val metrics)")
                    elif early.monitor not in val_metrics:
                        if run is not None:
                            run.log(
                                f"Early stopping: skipped (monitor {early.monitor!r} "
                                "not in val metrics)"
                            )
                    else:
                        score = float(val_metrics[early.monitor])
                        score_history.append(score)
                        smooth = sum(score_history[-window:]) / len(
                            score_history[-window:]
                        )
                        if best_score is None or improved(
                            smooth,
                            best_score,
                            mode=early.mode,
                            min_delta=early.min_delta,
                        ):
                            best_score = smooth
                            best_epoch = epoch
                            bad_epochs = 0
                            if run is not None:
                                run.log(
                                    f"Early stopping: best {early.monitor}={smooth:.6g} "
                                    f"(smoothed over {len(score_history[-window:])}, "
                                    f"raw={score:.6g}) at epoch {epoch}"
                                )
                        else:
                            bad_epochs += 1
                            if run is not None:
                                run.log(
                                    f"Early stopping: no improvement for "
                                    f"{bad_epochs}/{early.patience} epochs "
                                    f"({early.monitor}={score:.6g}, "
                                    f"smooth={smooth:.6g}, "
                                    f"best={best_score:.6g} @ epoch {best_epoch})"
                                )
                            if (
                                bad_epochs >= early.patience
                                and epoch >= early.min_epochs
                            ):
                                early_stopped = True
                                if run is not None:
                                    run.log(
                                        f"Early stopping: triggered at epoch {epoch} "
                                        f"after {bad_epochs} bad epochs"
                                    )

                if (
                    schema.checkpoint_every
                    and epoch % schema.checkpoint_every == 0
                    and run is not None
                ):
                    self._save_checkpoint(
                        run,
                        task,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        f"epoch-{epoch:04d}",
                        artifact=False,
                        ema_state=ema.state_dict() if ema is not None else None,
                    )
                    keep = int(getattr(schema, "checkpoint_keep", 0) or 0)
                    if keep > 0:
                        periodic = sorted(
                            p for p in (run.directory / "checkpoints").glob("epoch-*")
                            if p.is_file()
                        )
                        for stale in periodic[:-keep]:
                            stale.unlink(missing_ok=True)

                ctx.epoch_metrics = {
                    "train_loss": epoch_record["train_loss"],
                    "train_batches": epoch_record["train_batches"],
                    "duration_s": epoch_record["duration_s"],
                }
                if "grad_norm_median" in epoch_record:
                    ctx.epoch_metrics["grad_norm_median"] = epoch_record["grad_norm_median"]
                    ctx.epoch_metrics["grad_norm_max"] = epoch_record["grad_norm_max"]
                ctx.selection = {
                    "best_raw_epoch": best_raw_epoch,
                    "best_raw_score": best_raw_score,
                    "best_ema_epoch": best_ema_epoch,
                    "best_ema_score": best_ema_score,
                    "early_best_epoch": best_epoch,
                    "early_best_score": best_score,
                    "bad_epochs": bad_epochs,
                    "early_stopped": int(early_stopped),
                    "checkpoint": selection_checkpoint,
                }
                self._call(callbacks, "on_epoch_end", ctx)
                if reached_limit or early_stopped:
                    break

            if run is not None:
                self._save_checkpoint(
                    run,
                    task,
                    optimizer,
                    scheduler,
                    last_epoch,
                    global_step,
                    "last",
                    ema_state=ema.state_dict() if ema is not None else None,
                )
                if best_path is not None and best_path.is_file():
                    run.save_artifact(best_path, "checkpoint:best")
                if best_ema_path is not None and best_ema_path.is_file():
                    run.save_artifact(best_ema_path, "checkpoint:best_ema")
                if ema is not None:
                    # 训练结束时的 EMA 权重（D-0024）：选模优先 EMA 时作为首选
                    # checkpoint。窄 val 场景下 best_ema 会停得很早，final_ema 更稳。
                    final_ema_path = self._save_checkpoint(
                        run,
                        task,
                        optimizer,
                        scheduler,
                        last_epoch,
                        global_step,
                        "final_ema",
                        artifact=False,
                        model_state=ema.state_dict(),
                    )
                    run.save_artifact(final_ema_path, "checkpoint:final_ema")
            policy.on_fit_end(task, optimizer)
            self._call(callbacks, "on_run_end", ctx)
            if run is not None:
                run.log("Status: completed")
                run.finalize("completed")
            return run
        except BaseException as exc:  # noqa: BLE001 - finalize on any failure
            self._call(callbacks, "on_exception", ctx, exc)
            if run is not None:
                run.log(f"Status: failed error={exc!r}")
                run.log("traceback:\n" + traceback.format_exc().rstrip())
                run.finalize("failed", error=repr(exc))
            raise

    def evaluate(
        self,
        task: Task,
        dataset: Any,
        spec: EvalSpec,
        run: Run,
    ) -> EvalRecord:
        validate_task(task)
        metrics = dict(task.configure_metrics())
        run.log(
            f"Eval [{spec.split}] start benchmark={spec.benchmark_id or '-'} "
            f"test_set={spec.test_set_id or '-'}"
        )
        values = self._validate(task, dataset, metrics, self.training_policy)

        eval_id = f"{run.run_id}-{spec.split}-{_short_id()}"
        data_version = _data_version(dataset)
        test_set_id = spec.test_set_id or data_version
        test_access = (
            self.test_access_log.record(test_set_id, eval_id, spec.split)
            if spec.split == "test"
            else 0
        )
        record = EvalRecord(
            eval_id=eval_id,
            split=spec.split,
            checkpoint_digest=sha256_file(spec.checkpoint),
            data_version=data_version,
            eval_spec_digest=eval_spec_digest(spec),
            metrics=values,
            created_at=utc_now(),
            test_access=test_access,
            benchmark_id=spec.benchmark_id,
            test_set_id=test_set_id,
        )
        path = run.save_json(f"evals/{record.eval_id}.json", record.to_dict())
        run.save_artifact(path, f"eval:{record.eval_id}")
        metric_text = " ".join(f"{key}={value:.6g}" for key, value in values.items())
        run.log(
            f"Eval [{spec.split}] done eval_id={record.eval_id} "
            f"metrics={metric_text} test_access={test_access}"
        )
        return record

    def load_checkpoint(
        self,
        path: str | Path,
        task: Task,
        optimizer: Optimizer,
        scheduler: Scheduler | None = None,
    ) -> Checkpoint:
        payload = self.checkpoint_codec.load(path)
        task.get_model().load_state_dict(payload.model)
        optimizer.load_state_dict(payload.optimizer)
        if scheduler is not None and payload.scheduler is not None:
            scheduler.load_state_dict(payload.scheduler)
        if payload.policy:
            self.training_policy.load_state_dict(payload.policy)
        restore_rng_state(payload.rng_state)
        return payload

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_step(self, task: Task, batch: Batch) -> StepOutput:
        custom = getattr(task, "custom_train_step", None)
        if callable(custom):
            return custom(batch)
        return task.train_step(batch)

    def _validate(
        self,
        task: Task,
        dataset: Any,
        metrics: Mapping[str, Metric],
        policy: TrainingPolicy,
    ) -> dict[str, float]:
        # 验证必须用 eval 模式：BN 用 running 统计（而非批统计），Dropout 关闭。
        # 此前漏掉 eval()，所有 in-run 验证与 eval 评估都在 train 模式下跑
        # （D-0023）。torch 之外的任务按 duck-typing 容忍缺失。
        model = task.get_model()
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        try:
            for metric in metrics.values():
                metric.reset()

            with policy.eval_context():
                for batch in dataset:
                    with policy.forward_context():
                        step = task.val_step(batch)
                    output = ModelOutput(predictions=step.predictions, aux=step.meta)
                    for metric in metrics.values():
                        metric.update(output, batch)
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        return _metric_values(metrics)

    def _save_checkpoint(
        self,
        run: Run,
        task: Task,
        optimizer: Optimizer,
        scheduler: Scheduler | None,
        epoch: int,
        step: int,
        name: str,
        *,
        artifact: bool = True,
        model_state: Mapping[str, Any] | None = None,
        ema_state: Mapping[str, Any] | None = None,
    ) -> Path:
        checkpoint = Checkpoint(
            model=(
                model_state
                if model_state is not None
                else task.get_model().state_dict()
            ),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            rng_state=capture_rng_state(),
            epoch=epoch,
            step=step,
            manifest=run.manifest.to_dict(),
            data_position=getattr(task, "data_position", None),
            policy=self.training_policy.state_dict(),
            ema=ema_state,
        )
        path = run.dir / "checkpoints" / f"{name}{self.checkpoint_codec.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_codec.dump(path, checkpoint)
        if artifact:
            run.save_artifact(path, f"checkpoint:{name}")
        run.log(f"Checkpoint: {name} -> {path.name}")
        return path

    @staticmethod
    def _call(callbacks: list[Callback], hook: str, *args: Any) -> None:
        for callback in callbacks:
            method = getattr(callback, hook, None)
            if callable(method):
                method(*args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metric_values(metrics: Mapping[str, Metric]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, metric in metrics.items():
        result = metric.compute()
        items = result.values if isinstance(result, MetricResult) else dict(result)
        if len(items) == 1 and name in items:
            values.update({key: float(value) for key, value in items.items()})
        else:
            values.update({f"{name}/{key}": float(value) for key, value in items.items()})
    return values


def _data_version(dataset: Any) -> str:
    spec = getattr(dataset, "spec", None)
    if isinstance(spec, DataSpec):
        return spec.version
    if isinstance(spec, Mapping):
        return str(spec.get("version", "unknown"))
    return "unknown"


def _data_versions(train_data: Any, val_data: Any | None) -> dict[str, str]:
    versions = {"train": _data_version(train_data)}
    if val_data is not None:
        versions["val"] = _data_version(val_data)
    return versions


def _length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _batch_size(batch: Any) -> int | None:
    """Best-effort batch size without importing a tensor library."""
    if not isinstance(batch, Mapping):
        return None
    for value in batch.values():
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return int(shape[0])
            except (IndexError, TypeError, ValueError):
                pass
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def _current_lr(optimizer: Any) -> float | None:
    inner = getattr(optimizer, "optimizer", optimizer)
    groups = getattr(inner, "param_groups", None)
    if groups:
        return float(groups[0].get("lr", 0.0))
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _short_id() -> str:
    import uuid

    return uuid.uuid4().hex[:6]
