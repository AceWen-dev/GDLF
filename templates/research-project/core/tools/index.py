"""Generate output/INDEX.md: the one-glance front door of this project.

INDEX.md is a **rendered view**: it is 100% derived from the run store plus the
candidate and claim registries, it is never hand-edited, and ``make index``
regenerates it. Facts live in ``runs/``; judgments live in ``candidates/`` and
``claims/``; everything disposable lives in ``scratch/``. If a number is not in
this file, look it up with ``core/tools/runs.py`` / ``run_info.py``, not by
opening JSONs by hand.

Usage:
  python core/tools/index.py
  make index            # same thing
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools import runs as runs_tool  # noqa: E402
from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.candidates import CandidateStore  # noqa: E402
from dlef.claim import ClaimStore  # noqa: E402
from dlef.report import format_candidate_report  # noqa: E402
from dlef.run import Run, RunStore  # noqa: E402

_DEFAULT_OUTPUT = Path(os.environ.get("DLEF_OUTPUT_ROOT", str(ROOT / "output")))


def _clip(value: object, width: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _md_table(headers: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "-")) for key in headers) + " |")
    return "\n".join(lines)


def _funnel_section(runs: list[Run], candidates: list[Any], claims: list[Any]) -> str:
    statuses: dict[str, int] = {}
    for run in runs:
        statuses[run.manifest.status] = statuses.get(run.manifest.status, 0) + 1
    candidate_status: dict[str, int] = {}
    for candidate in candidates:
        candidate_status[candidate.status] = candidate_status.get(candidate.status, 0) + 1
    unapproved = sum(1 for item in [*candidates, *claims] if not item.approved_by.strip())

    headers = [
        "runs",
        "running",
        "completed",
        "failed",
        "candidates",
        "promoted",
        "rejected",
        "claims",
        "unapproved",
    ]
    row = {
        "runs": len(runs),
        "running": statuses.get("running", 0),
        "completed": statuses.get("completed", 0),
        "failed": statuses.get("failed", 0),
        "candidates": len(candidates),
        "promoted": candidate_status.get("promoted", 0),
        "rejected": candidate_status.get("rejected", 0),
        "claims": len(claims),
        "unapproved": unapproved or "-",
    }
    text = _md_table(headers, [row])
    if unapproved:
        text += (
            "\n\n> WARN: "
            f"{unapproved} candidate/claim record(s) have no `approved_by`. "
            "They predate approval tracking or bypassed it; re-approve or "
            "remove them."
        )
    return text


def _latest_val(run: Run, metric_key: str) -> float | None:
    value = run.latest_metrics("val").get(metric_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _experiments_section(runs: list[Run], args: argparse.Namespace) -> str:
    """Per-experiment rows straight from the latest val records.

    The front door must work right after training, before any eval run
    exists; `runs.py` remains the eval-record index.
    """

    if not runs:
        return "(no runs)"
    groups: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.manifest.project, run.manifest.experiment), []).append(run)

    rows = []
    for (project, experiment), items in sorted(groups.items(), key=lambda item: item[0][1]):
        items = sorted(items, key=runs_tool._started)
        latest = items[-1]
        statuses: dict[str, int] = {}
        values: list[float] = []
        tags: set[str] = set()
        for run in items:
            statuses[run.manifest.status] = statuses.get(run.manifest.status, 0) + 1
            tags.update(tag for tag in run.manifest.tags if tag not in runs_tool._HIDDEN_TAGS)
            value = _latest_val(run, args.metric_key)
            if value is not None:
                values.append(value)
        last_value = _latest_val(latest, args.metric_key)
        best_value = (
            min(values) if args.metric_mode == "min" and values else (max(values) if values else None)
        )
        rows.append(
            {
                "experiment": experiment,
                "runs": len(items),
                "latest": runs_tool._timestamp(runs_tool._started(latest)),
                "status": " ".join(f"{name}={count}" for name, count in sorted(statuses.items())),
                f"last val {args.metric_key}": runs_tool._format_metric(last_value),
                f"best val {args.metric_key}": runs_tool._format_metric(best_value),
                "tags": ",".join(sorted(tags)) or "-",
            }
        )
    headers = [
        "experiment",
        "runs",
        "latest",
        "status",
        f"last val {args.metric_key}",
        f"best val {args.metric_key}",
        "tags",
    ]
    return _md_table(headers, rows)


def _recent_section(runs: list[Run], args: argparse.Namespace) -> str:
    if not runs:
        return "(no runs)"
    ordered = sorted(runs, key=lambda run: runs_tool._started(run), reverse=True)
    rows = []
    for run in ordered[: args.recent]:
        rows.append(
            {
                "run_id": run.run_id,
                "experiment": run.manifest.experiment,
                "seed": run.manifest.seed,
                "status": run.manifest.status,
                "started": runs_tool._timestamp(runs_tool._started(run)),
                f"val {args.metric_key}": runs_tool._format_metric(
                    _latest_val(run, args.metric_key)
                ),
            }
        )
    headers = ["run_id", "experiment", "seed", "status", "started", f"val {args.metric_key}"]
    return _md_table(headers, rows)


def _claims_section(claims: list[Any], evals_by_id: dict[str, Any]) -> str:
    if not claims:
        return "(no claims)"
    rows = []
    for claim in claims:
        value = claim.value
        split = ""
        if value is None:
            record = evals_by_id.get(claim.eval_ids[0]) if claim.eval_ids else None
            if record is not None:
                split = record.split
                value = record.metrics.get(claim.metric)
        rendered = f"{value:.6g}" if isinstance(value, (int, float)) else "-"
        rows.append(
            {
                "claim": claim.claim_id,
                "text": _clip(claim.text, 48) or "-",
                "metric": claim.metric or "-",
                "value": rendered,
                "unit": claim.unit or "-",
                "split": split or "-",
                "approved_by": _clip(claim.approved_by, 24) or "UNAPPROVED",
            }
        )
    return _md_table(
        ["claim", "text", "metric", "value", "unit", "split", "approved_by"], rows
    )


def _build_index(args: argparse.Namespace) -> str:
    run_store = RunStore(Path(args.runs))
    runs = run_store.iter_runs()

    candidates = (
        CandidateStore(Path(args.candidates)).list()
        if Path(args.candidates).is_dir()
        else []
    )
    claims = (
        ClaimStore(Path(args.claims)).list() if Path(args.claims).is_dir() else []
    )
    runs_by_id = {run.run_id: run for run in runs}
    evals_by_id = {
        record.eval_id: record for _, record in run_store.iter_evals()
    }
    projects = sorted({run.manifest.project for run in runs})
    title = " / ".join(projects) if projects else "(no project)"

    sections = [
        f"# INDEX — {title}",
        "> generated by `make index` at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}; this file is a "
        "rebuildable view — do not edit, regenerate.",
        "## Funnel\n\n" + _funnel_section(runs, candidates, claims),
        "## Experiments\n\n" + _experiments_section(runs, args),
        "## Recent runs\n\n" + _recent_section(runs, args),
        "## Shortlist\n\n" + format_candidate_report(candidates, runs_by_id),
        "## Claims\n\n" + _claims_section(claims, evals_by_id),
        "## Where to look\n\n"
        "- one run: `python core/tools/run_info.py --run <run_id>`\n"
        "- full claim report: `make report` → `reports/report.md`\n"
        "- curves: `make tensorboard`\n"
        "- cleanup plan: `make gc-dry`\n"
        "- disposable drafts: `scratch/` (never cited by claims or this index)",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", default=str(_DEFAULT_OUTPUT / "runs"))
    parser.add_argument("--candidates", default=str(_DEFAULT_OUTPUT / "candidates"))
    parser.add_argument("--claims", default=str(_DEFAULT_OUTPUT / "claims"))
    parser.add_argument("--metric-key", default="mae")
    parser.add_argument("--metric-mode", choices=("min", "max"), default="min")
    parser.add_argument(
        "--recent", type=int, default=10, help="recent runs to list (default 10)"
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUTPUT / "INDEX.md"),
        help="output file (default output/INDEX.md)",
    )
    args = parser.parse_args()

    text = _build_index(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"index: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
