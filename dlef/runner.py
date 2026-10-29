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

import random
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


class Runner:
    """Minimal, engine-agnostic train/eval loop."""

    def __init__(
        self,
        run_store: RunStore,
        artifact_store: ArtifactStore | None = None,
        checkpoint_codec: CheckpointCodec | None = None,
        training_policy: TrainingPolicy | None = None,
        test_access_log: TestAccessLog | None = None,
    ) -> None:
        self.run_store = run_store
        self.artifact_store = artifact_store or run_store.artifact_store
        self.checkpoint_codec: CheckpointCodec = checkpoint_codec or JsonCheckpointCodec()
        self.training_policy: TrainingPolicy = training_policy or NoOpTrainingPolicy()
        self.test_access_log = test_access_log or TestAccessLog(
            run_store.root.parent / "test_access"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        task: Task,
        train_data: Any,
        val_data: Any | None,
        config: Config,
    ) -> Run | None:
        validate_task(task)
        schema = config.schema
        policy = self.training_policy
        policy.setup()

        run: Run | None = None
        if policy.is_main_process:
            run = self.run_store.create(
                project=schema.project,
                experiment=schema.experiment,
                config_digest=config.digest,
                seed=schema.seed,
                data_versions=_data_versions(train_data, val_data),
                tags={"framework": "dlef"},
                config_source=config.source,
            )
            run.save_json("config.resolved.json", config.resolved)

        model = task.get_model()
        optimizer = task.configure_optimizer(model)
        scheduler = task.configure_scheduler(optimizer)
        metrics = dict(task.configure_metrics())
        callbacks = list(task.configure_callbacks())
        ctx = RunContext(
            run_id=run.run_id if run is not None else "nonmain",
            config=schema,
            manifest=run.manifest if run is not None else None,
        )

        global_step = 0
        last_epoch = 0
        try:
            policy.on_fit_start(task, optimizer, schema)
            self._call(callbacks, "on_run_start", ctx)
            for epoch in range(1, schema.epochs + 1):
                last_epoch = epoch
                ctx.epoch = epoch
                set_epoch = getattr(train_data, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)
                self._call(callbacks, "on_epoch_start", ctx)

                train_losses: list[float] = []
                reached_limit = False
                for batch in train_data:
                    global_step += 1
                    ctx.step = global_step
                    with policy.forward_context():
                        step = self._train_step(task, batch)

                    if step.loss is not None:
                        policy.backward(policy.scale_loss(step.loss))
                        train_losses.append(to_float(step.loss))
                        if policy.should_step(global_step):
                            policy.optimizer_step(task, optimizer)
                            policy.zero_grad(optimizer)
                            if (
                                scheduler is not None
                                and getattr(scheduler, "step_unit", "epoch") == "step"
                            ):
                                scheduler.step()

                    if schema.log_every and global_step % schema.log_every == 0 and run is not None:
                        record = {
                            "phase": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": to_float(step.loss) if step.loss is not None else None,
                        }
                        record.update({key: to_float(value) for key, value in step.terms.items()})
                        run.log_metrics(record)

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
                ctx.metrics = val_metrics
                epoch_record = {
                    "phase": "val",
                    "epoch": epoch,
                    "train_loss": _mean(train_losses),
                }
                epoch_record.update(val_metrics)
                if run is not None:
                    run.log_metrics(epoch_record)

                if (
                    schema.checkpoint_every
                    and epoch % schema.checkpoint_every == 0
                    and run is not None
                ):
                    self._save_checkpoint(
                        run, task, optimizer, scheduler, epoch, global_step, f"epoch-{epoch:04d}"
                    )

                self._call(callbacks, "on_epoch_end", ctx)
                if reached_limit:
                    break

            if run is not None:
                self._save_checkpoint(
                    run, task, optimizer, scheduler, last_epoch, global_step, "last"
                )
            policy.on_fit_end(task, optimizer)
            self._call(callbacks, "on_run_end", ctx)
            if run is not None:
                run.finalize("completed")
            return run
        except BaseException as exc:  # noqa: BLE001 - finalize on any failure
            self._call(callbacks, "on_exception", ctx, exc)
            if run is not None:
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
        rng_state = payload.rng_state
        if rng_state:
            if isinstance(rng_state, list):
                rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])
            random.setstate(rng_state)
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
        for metric in metrics.values():
            metric.reset()

        for batch in dataset:
            with policy.forward_context():
                step = task.val_step(batch)
            output = ModelOutput(predictions=step.predictions, aux=step.meta)
            for metric in metrics.values():
                metric.update(output, batch)

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
    ) -> Path:
        checkpoint = Checkpoint(
            model=task.get_model().state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            rng_state=random.getstate(),
            epoch=epoch,
            step=step,
            manifest=run.manifest.to_dict(),
            data_position=getattr(task, "data_position", None),
            policy=self.training_policy.state_dict(),
        )
        path = run.dir / "checkpoints" / f"{name}{self.checkpoint_codec.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_codec.dump(path, checkpoint)
        run.save_artifact(path, f"checkpoint:{name}")
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


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _short_id() -> str:
    import uuid

    return uuid.uuid4().hex[:6]
