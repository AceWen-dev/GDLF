"""One subprocess trial for the torch sweep example."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    print("torch is not installed; run: pip install -e .[torch]")
    raise SystemExit(1)

from adapters.torch import TorchCheckpointCodec, TorchTrainingPolicy  # noqa: E402
from dlef.config import resolve_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from dlef.run import ArtifactStore, RunStore  # noqa: E402
from dlef.runner import Runner  # noqa: E402
from dlef.sweep import TrialResult  # noqa: E402
from examples.torch_regression.config import TorchConfig  # noqa: E402
from examples.torch_regression.data import make_loaders  # noqa: E402
from examples.torch_regression.task import build_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--experiment", default="trial")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    result_path = os.environ.get("DLEF_TRIAL_RESULT")
    if not result_path:
        print("missing DLEF_TRIAL_RESULT; run through SubprocessSweep", file=sys.stderr)
        return 2

    config = resolve_config(
        TorchConfig,
        {
            "seed": 7,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": 32,
            "n_train": 512,
            "n_val": 128,
            "noise": 0.1,
            "hidden": 16,
            "amp": args.amp,
            "grad_accum_steps": args.grad_accum_steps,
            "grad_clip": args.grad_clip,
            "project": "torch-sweep",
            "experiment": args.experiment,
            "log_every": 0,
        },
    )

    policy = TorchTrainingPolicy(
        amp=config.schema.amp,
        grad_accum_steps=config.schema.grad_accum_steps,
        grad_clip=config.schema.grad_clip,
    )
    policy.setup()
    device = policy.device

    outputs = Path(__file__).resolve().parent / "outputs"
    run_store = RunStore(outputs / "runs")
    artifact_store = ArtifactStore(outputs / "artifacts")
    runner = Runner(
        run_store,
        artifact_store,
        checkpoint_codec=TorchCheckpointCodec(),
        training_policy=policy,
    )

    train_loader, val_loader = make_loaders(config.schema)
    task = build_task(config.schema, device=device)
    run = runner.fit(task, train_loader, val_loader, config)
    if run is None:
        print("non-main process; no run written")
        return 0

    record = runner.evaluate(
        task,
        val_loader,
        EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
        run,
    )
    gpu_ids = [int(item) for item in os.environ.get("DLEF_GPU_IDS", "").split(",") if item]
    result = TrialResult(
        trial_id=os.environ.get("DLEF_TRIAL_ID", "unknown"),
        status="completed",
        run_id=run.run_id,
        metrics=dict(record.metrics),
        gpu_ids=tuple(gpu_ids),
    )
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle)
    print(f"trial {result.trial_id}: {record.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
