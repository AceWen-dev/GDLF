"""Synthetic torch regression through the adapter.

    pip install -e .[torch]
    python examples/torch_regression/run.py

DDP (multi-GPU):
    torchrun --nproc_per_node=2 examples/torch_regression/run.py
"""

from __future__ import annotations

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
from examples.torch_regression.config import TorchConfig  # noqa: E402
from examples.torch_regression.data import make_loaders  # noqa: E402
from examples.torch_regression.task import build_task  # noqa: E402


def main() -> int:
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    config = resolve_config(
        TorchConfig,
        {
            "seed": 7,
            "epochs": 30,
            "lr": 1e-2,
            "batch_size": 32,
            "n_train": 512,
            "n_val": 128,
            "noise": 0.1,
            "hidden": 16,
            "amp": True,
            "grad_accum_steps": 2,
            "grad_clip": 1.0,
            "project": "torch-toy",
            "experiment": "amp-accum-clip",
            "log_every": 10,
        },
    )

    policy = TorchTrainingPolicy(
        amp=config.schema.amp,
        grad_accum_steps=config.schema.grad_accum_steps,
        grad_clip=config.schema.grad_clip,
        ddp=ddp,
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

    train_loader, val_loader = make_loaders(config.schema, distributed=ddp)
    task = build_task(config.schema, device=device)
    run = runner.fit(task, train_loader, val_loader, config)
    if run is None:
        print(f"rank {policy._rank}: training complete (non-main)")
        return 0

    record = runner.evaluate(
        task,
        val_loader,
        EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
        run,
    )
    print(f"run: {run.run_id}")
    print(f"device: {device}")
    print(f"amp={policy.amp} accum={policy.grad_accum_steps} clip={policy.grad_clip} ddp={policy.ddp}")
    print(f"metrics: {record.metrics}")
    print(f"checkpoint: {run.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
