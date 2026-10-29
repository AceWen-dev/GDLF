"""Thin evaluation entry point: checkpoint -> separate eval run -> EvalRecord.

By default the eval reuses the training run's ``config.resolved.json`` (found via
the checkpoint manifest). This is what keeps model/loss changes in sync: no
manual ``--set`` is needed for evaluation. ``--config`` is only a fallback when
the original run directory is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.torch import TorchCheckpointCodec  # noqa: E402
from dlef.config import load_config, resolve_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from myproject.config import ProjectConfig  # noqa: E402
from myproject.data import make_loaders  # noqa: E402
from myproject.runtime import build_policy, build_runner, parse_overrides  # noqa: E402
from myproject.task import build_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="fallback config if run dir is missing")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="optional overrides applied on top of the training run config",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    overrides = parse_overrides(args.overrides)
    checkpoint_payload = TorchCheckpointCodec().load(checkpoint)
    manifest = getattr(checkpoint_payload, "manifest", {}) or {}
    train_project = str(manifest.get("project", ""))
    train_experiment = str(manifest.get("experiment", ""))
    train_run_id = str(manifest.get("run_id", ""))

    resolved_path = (
        ROOT
        / "outputs"
        / "runs"
        / train_project
        / train_experiment
        / train_run_id
        / "config.resolved.json"
    )
    if resolved_path.is_file():
        base_resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        config = resolve_config(
            ProjectConfig,
            base_resolved,
            overrides=[overrides] if overrides else [],
        )
        source = f"training run {train_run_id}"
    else:
        if not args.config:
            print(
                "could not find the training run config; pass --config as fallback",
                file=sys.stderr,
            )
            return 2
        config = load_config(
            args.config,
            ProjectConfig,
            overrides=[overrides] if overrides else [],
        )
        source = args.config

    if not train_experiment:
        train_experiment = config.schema.experiment

    policy = build_policy(config.schema)
    policy.setup()
    device = policy.device

    runner = build_runner(ROOT / "outputs", policy)
    _, val_loader = make_loaders(config.schema.data, config.schema.seed)
    task = build_task(config.schema, device=device)
    optimizer = task.configure_optimizer(task.get_model())
    policy.on_fit_start(task, optimizer, config.schema)
    runner.load_checkpoint(checkpoint, task, optimizer)

    run = runner.run_store.create(
        project=config.schema.project,
        experiment=f"{train_experiment}-eval",
        config_digest=config.digest,
        seed=config.schema.seed,
        data_versions={"val": val_loader.spec.version},
        tags={"kind": "eval"},
    )
    run.save_json("config.resolved.json", config.resolved)
    try:
        record = runner.evaluate(
            task,
            val_loader,
            EvalSpec(checkpoint=str(checkpoint), split="val"),
            run,
        )
        run.finalize("completed")
    except BaseException as exc:  # noqa: BLE001
        run.finalize("failed", error=repr(exc))
        raise

    print(f"config: {source}")
    print(f"eval run: {run.run_id}")
    print(f"metrics: {record.metrics}")
    print(f"eval record: {run.dir / 'evals' / (record.eval_id + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
