"""List runs with their config source and latest eval metric.

RunStore is the source of truth; this command is only a readable index. It
never edits run data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dlef.eval import EvalRecord  # noqa: E402
from dlef.run import Run, RunStore  # noqa: E402


def _evals(run: Run) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for path in run.eval_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(EvalRecord.from_dict(payload))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return records


def _latest_eval(records: list[EvalRecord]) -> EvalRecord | None:
    if not records:
        return None
    return max(records, key=lambda record: record.created_at)


def _benchmarks(evals: list[EvalRecord]) -> str:
    ids = sorted({record.benchmark_id for record in evals if record.benchmark_id})
    return ", ".join(value[:12] for value in ids) if ids else "-"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(ROOT / "outputs" / "runs"))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--config", default=None, help="substring of config_source")
    parser.add_argument("--status", default=None)
    parser.add_argument("--metric-key", default="mae")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    store = RunStore(Path(args.runs))
    runs = sorted(store.iter_runs(), key=lambda run: run.manifest.started_at, reverse=True)

    rows: list[tuple[str, ...]] = []
    for run in runs:
        manifest = run.manifest
        if args.experiment and manifest.experiment != args.experiment:
            continue
        if args.status and manifest.status != args.status:
            continue
        if args.config and args.config not in manifest.config_source:
            continue
        evals = _evals(run)
        latest = _latest_eval(evals)
        metric = ""
        split = ""
        if latest is not None:
            split = latest.split
            value = latest.metrics.get(args.metric_key)
            if value is not None:
                try:
                    metric = f"{float(value):.6g}"
                except (TypeError, ValueError):
                    metric = str(value)
        rows.append(
            (
                run.run_id,
                manifest.status,
                manifest.experiment,
                manifest.config_source or "-",
                str(manifest.seed),
                _benchmarks(evals),
                split,
                metric,
            )
        )
        if len(rows) >= args.limit:
            break

    headers = (
        "run_id",
        "status",
        "experiment",
        "config",
        "seed",
        "benchmark",
        "split",
        "metric",
    )
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if not rows:
        print("(no runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
