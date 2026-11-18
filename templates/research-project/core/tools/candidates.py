"""Manage the candidate shortlist: runs worth studying before they become claims.

A candidate is a named pointer to one or more runs (normally one configuration
across seeds) plus *why* it is worth keeping and *what to look at next*. It is
not a claim: claims record reported numbers with an eval/checkpoint/run evidence
chain. Candidates are the triage layer in between.

**Owner approval is required.** The shortlist and the claim registry are the
owner's decisions, not the agent's. `add` / `promote` / `reject` / `archive`
refuse to run without ``--approved-by``, which must quote the owner's approval
(e.g. ``'twj 2026-11-18: ranksim-g20 may enter'``). An agent may *propose* —
prepare the exact command and the expected record — but it must ask the owner
first and never write the approval itself.

The tool only edits ``output/candidates/``. RunStore remains the source of truth;
runs are never modified.

Views / commands:
  list      one row per candidate (default; active candidates only)
  show      one candidate with its runs and claims
  add       shortlist a run set (owner approval)
  promote   attach claim ids and promote (owner approval)
  reject    close a candidate with a reason (owner approval)
  archive   keep the record but stop protecting its runs (owner approval)

Examples:
  python core/tools/candidates.py
  python core/tools/candidates.py list --all
  python core/tools/candidates.py show v11-ranksim-g20

  python core/tools/candidates.py add \
    --id v11-ranksim-g20 \
    --title "RankSim + log target, 3 seeds" \
    --run <run1> --run <run2> --run <run3> \
    --metric v11/worst_bin \
    --reason "best worst_bin so far; verify the D cohort" \
    --next "per-bin MAE, then test eval if it holds" \
    --approved-by "twj 2026-11-18: v11-ranksim-g20 may enter the shortlist"

  python core/tools/candidates.py reject v11-ranksim-g20 \
    --reason "D cohort regressed; not worth test access" \
    --approved-by "twj 2026-11-18: agreed, close it"

  python core/tools/candidates.py promote v11-ranksim-g20 \
    --claim best-d-worst \
    --approved-by "twj 2026-11-18: promote with the d-worst claim"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.candidates import (  # noqa: E402
    CANDIDATE_ACTIVE_STATUSES,
    CANDIDATE_STATUSES,
    Candidate,
    CandidateError,
    CandidateStore,
)
from dlef.claim import ClaimStore  # noqa: E402
from dlef.report import format_candidate_report  # noqa: E402
from dlef.run import Run, RunStore  # noqa: E402

_DEFAULT_OUTPUT = Path(os.environ.get("DLEF_OUTPUT_ROOT", str(ROOT / "output")))


def _default(name: str) -> str:
    return str(_DEFAULT_OUTPUT / name)


def _find_run(store: RunStore, selector: str) -> Run:
    path = Path(selector)
    if path.exists():
        return store.open(path)
    for run in store.iter_runs():
        if run.run_id == selector or run.dir.name == selector:
            return run
    raise CandidateError(f"run not found: {selector} (under {store.root})")


def _parse_tags(items: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        name = name.strip()
        if not name:
            raise CandidateError(f"invalid --tag: {item!r}")
        tags[name] = value if separator else "true"
    return tags


def _metric_value(candidate: Candidate, run: Run) -> float | None:
    if not candidate.metric:
        return None
    value = run.latest_metrics("val").get(candidate.metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _store(args: argparse.Namespace) -> CandidateStore:
    return CandidateStore(args.candidates)


def _runs(args: argparse.Namespace) -> dict[str, Run]:
    return {run.run_id: run for run in RunStore(args.runs).iter_runs()}


def _cmd_list(args: argparse.Namespace) -> int:
    candidates = _store(args).list()
    if args.status:
        candidates = [item for item in candidates if item.status == args.status]
    elif not args.all:
        candidates = [
            item for item in candidates if item.status in CANDIDATE_ACTIVE_STATUSES
        ]
    if not candidates:
        print("(no candidates)")
        return 0
    if args.json:
        print(
            json.dumps(
                [item.to_dict() for item in candidates],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(format_candidate_report(candidates, _runs(args)))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    candidate = _store(args).get(args.candidate_id)
    runs = _runs(args)
    print(json.dumps(candidate.to_dict(), indent=2, ensure_ascii=False))
    print()
    print("runs:")
    for run_id in candidate.run_ids:
        run = runs.get(run_id)
        if run is None:
            print(f"  {run_id}  MISSING")
            continue
        value = _metric_value(candidate, run)
        rendered = "-" if value is None else f"{value:.6g}"
        print(
            f"  {run_id}  experiment={run.manifest.experiment} "
            f"seed={run.manifest.seed} status={run.manifest.status} "
            f"{candidate.metric or 'metric'}={rendered}"
        )
    print()
    print("claims:", ", ".join(candidate.claim_ids) or "-")
    print("approved_by:", candidate.approved_by or "(not recorded)")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    run_store = RunStore(args.runs)
    found = [_find_run(run_store, selector) for selector in args.run]
    experiments = sorted({run.manifest.experiment for run in found})
    if len(experiments) > 1:
        print(
            "warning: candidate spans multiple experiments: "
            + ", ".join(experiments)
        )
    candidate = Candidate(
        candidate_id=args.id,
        title=args.title,
        run_ids=tuple(run.run_id for run in found),
        reason=args.reason,
        next=args.next,
        metric=args.metric,
        experiment=args.experiment or experiments[0],
        priority=args.priority,
        tags=_parse_tags(args.tag),
        approved_by=args.approved_by,
    )
    _store(args).add(candidate)
    print(f"candidate added: {candidate.candidate_id} ({len(found)} run(s))")
    print(f"approved_by: {candidate.approved_by}")
    print(f"candidates: {args.candidates}")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    claim_store = ClaimStore(args.claims)
    missing: list[str] = []
    for claim_id in args.claim:
        try:
            claim_store.get(claim_id)
        except FileNotFoundError:
            missing.append(claim_id)
    if missing:
        raise CandidateError(
            "claim not found: " + ", ".join(missing) + f" (under {args.claims})"
        )
    updated = _store(args).update(
        args.candidate_id,
        status="promoted",
        claim_ids=tuple(args.claim),
        approved_by=args.approved_by,
    )
    print(f"candidate promoted: {updated.candidate_id}")
    print("claims:", ", ".join(updated.claim_ids))
    print(f"approved_by: {updated.approved_by}")
    return 0


def _cmd_reject(args: argparse.Namespace) -> int:
    updated = _store(args).update(
        args.candidate_id,
        status="rejected",
        reason=args.reason,
        approved_by=args.approved_by,
    )
    print(f"candidate rejected: {updated.candidate_id}")
    print(f"approved_by: {updated.approved_by}")
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    updated = _store(args).update(
        args.candidate_id,
        status="archived",
        approved_by=args.approved_by,
    )
    print(f"candidate archived: {updated.candidate_id}")
    print(f"approved_by: {updated.approved_by}")
    return 0


def _common_paths() -> argparse.ArgumentParser:
    """Path flags usable before or after the subcommand."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs", default=argparse.SUPPRESS, help="run store root")
    parser.add_argument(
        "--candidates",
        default=argparse.SUPPRESS,
        help="candidate store root",
    )
    parser.add_argument(
        "--claims",
        default=argparse.SUPPRESS,
        help="claim store root (used by promote)",
    )
    return parser


def _approval_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--approved-by",
        required=True,
        metavar="OWNER_APPROVAL",
        help="the owner's approval, quoted, e.g. "
        "'twj 2026-11-18: ranksim-g20 may enter the shortlist'. "
        "Agents must ask the owner first and quote the answer; "
        "never write this on their own judgment",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", default=_default("runs"), help="run store root")
    parser.add_argument(
        "--candidates", default=_default("candidates"), help="candidate store root"
    )
    parser.add_argument(
        "--claims", default=_default("claims"), help="claim store root"
    )
    sub = parser.add_subparsers(dest="command")
    common = _common_paths()

    list_parser = sub.add_parser(
        "list", parents=[common], help="list candidates (default)"
    )
    list_parser.add_argument("--status", choices=CANDIDATE_STATUSES)
    list_parser.add_argument(
        "--all", action="store_true", help="include rejected and archived"
    )
    list_parser.add_argument("--json", action="store_true", help="machine-readable")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = sub.add_parser(
        "show", parents=[common], help="show one candidate and its runs"
    )
    show_parser.add_argument("candidate_id")
    show_parser.set_defaults(func=_cmd_show)

    add_parser = sub.add_parser(
        "add", parents=[common], help="shortlist a run set (owner approval required)"
    )
    add_parser.add_argument("--id", required=True, help="candidate id (= file name)")
    add_parser.add_argument("--title", required=True)
    add_parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="run id or run directory; repeat for multiple seeds",
    )
    add_parser.add_argument(
        "--reason", required=True, help="why this run set is worth keeping"
    )
    add_parser.add_argument("--next", default="", help="what to inspect next")
    add_parser.add_argument(
        "--metric", default="", help="val metric to show in the shortlist"
    )
    add_parser.add_argument(
        "--experiment", default="", help="display grouping (default: first run's)"
    )
    add_parser.add_argument("--priority", type=int, default=0)
    add_parser.add_argument(
        "--tag", action="append", default=[], metavar="NAME[=VALUE]"
    )
    _approval_argument(add_parser)
    add_parser.set_defaults(func=_cmd_add)

    promote_parser = sub.add_parser(
        "promote", parents=[common], help="attach claim ids and promote (owner approval required)"
    )
    promote_parser.add_argument("candidate_id")
    promote_parser.add_argument(
        "--claim",
        action="append",
        required=True,
        help="claim id created from this candidate; repeat if needed",
    )
    _approval_argument(promote_parser)
    promote_parser.set_defaults(func=_cmd_promote)

    reject_parser = sub.add_parser(
        "reject", parents=[common], help="close a candidate with a reason (owner approval required)"
    )
    reject_parser.add_argument("candidate_id")
    reject_parser.add_argument("--reason", required=True)
    _approval_argument(reject_parser)
    reject_parser.set_defaults(func=_cmd_reject)

    archive_parser = sub.add_parser(
        "archive", parents=[common], help="stop protecting the candidate's runs (owner approval required)"
    )
    archive_parser.add_argument("candidate_id")
    _approval_argument(archive_parser)
    archive_parser.set_defaults(func=_cmd_archive)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        # Bare invocation is the list view.
        args.status = None
        args.all = False
        args.json = False
        return _cmd_list(args)
    try:
        return args.func(args)
    except CandidateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
