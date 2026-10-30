"""End-to-end toy: Config -> Run -> Eval -> Compare.

Run from the repo root:

    python examples/toy/run.py

It uses only the standard library. No torch, numpy, or YAML is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dlef.config import resolve_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from dlef.report import format_comparison, write_comparison  # noqa: E402
from dlef.run import ArtifactStore, RunStore  # noqa: E402
from dlef.runner import Runner  # noqa: E402
from dlef.seed import seed_everything  # noqa: E402
from examples.toy.components import ToyDataset  # noqa: E402
from examples.toy.config import ToyConfig  # noqa: E402
from examples.toy.task import ToyTask  # noqa: E402


def main() -> int:
    outputs = Path(__file__).resolve().parent / "outputs"
    run_store = RunStore(outputs / "runs")
    artifact_store = ArtifactStore(outputs / "artifacts")
    runner = Runner(run_store, artifact_store)

    records = []
    for lr in (0.1, 0.02):
        config = resolve_config(
            ToyConfig,
            {
                "seed": 7,
                "epochs": 40,
                "lr": lr,
                "batch_size": 16,
                "n_train": 512,
                "n_val": 128,
                "noise": 0.1,
                "project": "toy",
                "experiment": f"lr-{lr}",
                "log_every": 20,
            },
        )
        seed_everything(config.schema.seed)
        train_data = ToyDataset(
            n=config.schema.n_train,
            batch_size=config.schema.batch_size,
            noise=config.schema.noise,
            seed=config.schema.seed,
            w_true=config.schema.w_true,
            b_true=config.schema.b_true,
        )
        val_data = ToyDataset(
            n=config.schema.n_val,
            batch_size=config.schema.batch_size,
            noise=config.schema.noise,
            seed=config.schema.seed + 1,
            w_true=config.schema.w_true,
            b_true=config.schema.b_true,
        )

        task = ToyTask(config.schema)
        run = runner.fit(task, train_data, val_data, config)
        record = runner.evaluate(
            task,
            val_data,
            EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
            run,
        )
        records.append(record)
        print(f"lr={lr:g}  run={run.run_id}  metrics={record.metrics}  dir={run.dir}")

    table = format_comparison(records)
    print()
    print(table)
    write_comparison(outputs / "comparison.md", records, title="Toy comparison")
    print()
    print(f"report: {outputs / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
