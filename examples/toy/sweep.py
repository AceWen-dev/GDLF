"""Toy sweep: grid search over lr/batch, one trial = one run.

Run from the repo root:

    python examples/toy/sweep.py

The queue is local and GPU-aware. This machine example uses one GPU slot;
tests cover multi-slot allocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dlef.config import resolve_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from dlef.run import ArtifactStore, RunStore  # noqa: E402
from dlef.runner import Runner  # noqa: E402
from dlef.seed import seed_everything  # noqa: E402
from dlef.sweep import Choice, LocalSweep, SweepSpec, SweepStore, TrialResult  # noqa: E402
from examples.toy.components import ToyDataset  # noqa: E402
from examples.toy.config import ToyConfig  # noqa: E402
from examples.toy.task import ToyTask  # noqa: E402

BASE = {
    "seed": 7,
    "epochs": 30,
    "batch_size": 16,
    "n_train": 256,
    "n_val": 64,
    "noise": 0.1,
    "project": "toy-sweep",
    "log_every": 0,
}


def make_trial_fn(run_store: RunStore, runner: Runner):
    def trial_fn(trial, context):
        raw = dict(BASE)
        raw.update(trial.params)
        raw["experiment"] = f"trial-{trial.index:02d}"
        config = resolve_config(ToyConfig, raw)
        seed_everything(config.schema.seed)
        train_data = ToyDataset(
            config.schema.n_train,
            config.schema.batch_size,
            config.schema.noise,
            config.schema.seed,
        )
        val_data = ToyDataset(
            config.schema.n_val,
            config.schema.batch_size,
            config.schema.noise,
            config.schema.seed + 1,
        )
        task = ToyTask(config.schema)
        run = runner.fit(task, train_data, val_data, config)
        record = runner.evaluate(
            task,
            val_data,
            EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
            run,
        )
        return TrialResult(
            trial_id=trial.trial_id,
            status="completed",
            run_id=run.run_id,
            metrics=dict(record.metrics),
            gpu_ids=context.gpu_ids,
        )

    return trial_fn


def main() -> int:
    outputs = Path(__file__).resolve().parent / "outputs"
    run_store = RunStore(outputs / "runs")
    runner = Runner(run_store, ArtifactStore(outputs / "artifacts"))
    sweep_store = SweepStore(outputs / "sweeps")

    spec = SweepSpec(
        name="toy-lr-batch",
        params={
            "lr": Choice((0.1, 0.02, 0.005)),
            "batch_size": Choice((8, 16)),
        },
        method="grid",
        max_trials=4,
        seed=7,
    )
    sweep = LocalSweep(sweep_store, gpu_ids=(0,), max_parallel=1)
    record = sweep.run(spec, make_trial_fn(run_store, runner))

    trials = {trial["trial_id"]: trial for trial in record.trials}
    rows = []
    for result in record.results:
        trial = trials[result["trial_id"]]
        rows.append((trial, result))
    rows.sort(key=lambda item: item[1]["metrics"].get("mae", float("inf")))

    print(f"sweep: {record.sweep_id} status={record.status}")
    print()
    print("| trial | lr | batch | mae | gpu | run |")
    print("|---|---|---|---|---|---|")
    for trial, result in rows:
        params = trial["params"]
        print(
            f"| {trial['trial_id']} | {params.get('lr')} | {params.get('batch_size')} "
            f"| {result['metrics'].get('mae'):.6g} | {result['gpu_ids']} | {result['run_id']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
