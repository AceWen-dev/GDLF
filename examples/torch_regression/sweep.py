"""Subprocess sweep for the torch regression example.

    pip install -e .[torch]
    python examples/torch_regression/sweep.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("torch") is None:
    print("torch is not installed; run: pip install -e .[torch]")
    raise SystemExit(1)

from dlef.sweep import (  # noqa: E402
    Choice,
    SubprocessSweep,
    SweepSpec,
    SweepStore,
    TrialCommand,
)


def main() -> int:
    outputs = Path(__file__).resolve().parent / "outputs"
    store = SweepStore(outputs / "sweeps")
    spec = SweepSpec(
        name="torch-lr-accum",
        params={
            "lr": Choice((0.02, 0.005)),
            "grad_accum_steps": Choice((1, 2)),
        },
        method="grid",
        max_trials=4,
        seed=7,
    )

    def command_fn(trial, context):
        return TrialCommand(
            argv=[
                sys.executable,
                str(ROOT / "examples" / "torch_regression" / "trial.py"),
                "--lr",
                str(trial.params["lr"]),
                "--grad-accum-steps",
                str(trial.params["grad_accum_steps"]),
                "--grad-clip",
                "1.0",
                "--epochs",
                "10",
                "--amp",
                "--experiment",
                f"trial-{trial.index:02d}",
            ],
            cwd=str(ROOT),
        )

    record = SubprocessSweep(
        store,
        gpu_ids=(0,),
        max_parallel=1,
        gpus_per_trial=1,
    ).run(spec, command_fn)

    trials = {item["trial_id"]: item for item in record.trials}
    print(f"sweep: {record.sweep_id} status={record.status}")
    print()
    print("| trial | lr | accum | status | mae | run | log |")
    print("|---|---|---|---|---|---|---|")
    for result in record.results:
        trial = trials[result["trial_id"]]
        params = trial["params"]
        mae = result.get("metrics", {}).get("mae")
        print(
            f"| {result['trial_id']} | {params.get('lr')} | {params.get('grad_accum_steps')} "
            f"| {result['status']} | {mae} | {result.get('run_id')} | {result.get('log_path')} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
