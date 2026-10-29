"""Thin training entry point: config -> data/task -> Runner.fit -> eval."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import load_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from myproject.config import ProjectConfig  # noqa: E402
from myproject.data import make_loaders  # noqa: E402
from myproject.runtime import (  # noqa: E402
    build_policy,
    build_runner,
    parse_overrides,
    write_trial_result,
)
from myproject.task import build_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="typed override, e.g. --set model.name=linear --set loss.name=huber",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    config = load_config(
        args.config,
        ProjectConfig,
        overrides=[overrides] if overrides else [],
    )
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    policy = build_policy(config.schema, ddp=ddp)
    policy.setup()
    device = policy.device

    runner = build_runner(ROOT / "outputs", policy)
    train_loader, val_loader = make_loaders(
        config.schema.data,
        config.schema.seed,
        distributed=ddp,
    )
    task = build_task(config.schema, device=device)
    run = runner.fit(task, train_loader, val_loader, config)
    if run is None:
        print(f"rank {getattr(policy, '_rank', 0)}: training complete (non-main)")
        return 0

    record = runner.evaluate(
        task,
        val_loader,
        EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
        run,
    )
    write_trial_result(run, record.metrics)
    print(f"run: {run.run_id}")
    print(f"metrics: {record.metrics}")
    print(f"manifest: {run.manifest_path}")
    print(f"checkpoint: {run.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
