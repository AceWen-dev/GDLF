"""Tiny CLI: ``doctor``, ``compare``, and ``gc``.

Training is project-specific by design; a project provides a thin entry point
that builds its Task/Dataset and calls Runner.fit.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

from .claim import ClaimStore
from .eval import EvalRecord
from .report import format_claim_report, format_comparison
from .retention import RetentionPolicy, RetentionService
from .run import RunStore
from .sweep import SweepStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dlef")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the local environment")

    compare = sub.add_parser("compare", help="print a comparison table from eval records")
    compare.add_argument("paths", nargs="+", help="eval JSON files or run directories")

    report = sub.add_parser("report", help="render claims from eval records")
    report.add_argument("--runs", default="runs")
    report.add_argument("--claims", default="claims")
    report.add_argument("--out", default="report.md")

    gc = sub.add_parser("gc", help="plan or apply retention")
    gc.add_argument("--runs", default="runs", help="run store root")
    gc.add_argument("--sweeps", default="", help="sweep store root (optional)")
    gc.add_argument("--keep-last", type=int, default=5)
    gc.add_argument("--keep-best", type=int, default=0)
    gc.add_argument("--keep-every", type=int, default=0)
    gc.add_argument("--metric-key", default="")
    gc.add_argument("--metric-mode", choices=("min", "max"), default="min")
    gc.add_argument("--keep-tag", action="append", default=[])
    gc.add_argument("--protect-status", action="append", default=["running"])
    gc.add_argument("--min-age-days", type=float, default=0.0)
    gc.add_argument("--artifact-ttl-days", type=float, default=30.0)
    gc.add_argument("--no-protect-sweep-runs", action="store_true")
    gc.add_argument("--sweep-ttl-days", type=float, default=None)
    gc.add_argument("--verbose", action="store_true", help="show kept runs/artifacts too")
    gc.add_argument("--apply", action="store_true", help="actually delete; default is dry-run")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "compare":
        return compare_records(args.paths)
    if args.command == "report":
        return report_command(args)
    if args.command == "gc":
        return gc_command(args)
    return 2


def doctor() -> int:
    cwd = Path.cwd()
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {cwd}")
    print(f"runs/: {(cwd / 'runs').is_dir()}")
    print(f"artifacts/: {(cwd / 'artifacts').is_dir()}")
    free_gb = shutil.disk_usage(cwd).free / 1_000_000_000
    print(f"disk free: {free_gb:.1f} GB")

    try:
        import torch  # type: ignore

        cuda = torch.cuda.is_available()
        print(f"torch: {torch.__version__} (cuda={cuda})")
    except ImportError:
        print("torch: not installed (core does not require it)")

    try:
        import yaml  # type: ignore

        print(f"pyyaml: {yaml.__version__}")
    except ImportError:
        print("pyyaml: not installed (JSON configs still work)")

    print("core: ok")
    return 0


def compare_records(paths: list[str]) -> int:
    records: list[EvalRecord] = []
    for raw in paths:
        path = Path(raw)
        candidates = sorted(path.rglob("evals/*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            records.append(EvalRecord.from_dict(payload))
    print(format_comparison(records))
    return 0


def report_command(args: argparse.Namespace) -> int:
    run_store = RunStore(args.runs)
    claim_store = ClaimStore(args.claims)
    claims = claim_store.list()
    if not claims:
        print(f"no claims in {args.claims}")
        return 1

    resolved = {}
    missing: list[str] = []
    for claim in claims:
        for eval_id in claim.eval_ids:
            if eval_id in resolved:
                continue
            found = run_store.find_eval(eval_id)
            if found is None:
                missing.append(eval_id)
            else:
                resolved[eval_id] = found

    text = f"# Experiment report\n\n{format_claim_report(claims, resolved)}\n"
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"report: {out}")
    if missing:
        print(f"missing eval ids: {', '.join(sorted(set(missing)))}")
    return 0


def gc_command(args: argparse.Namespace) -> int:
    run_store = RunStore(args.runs)
    sweep_store = SweepStore(args.sweeps) if args.sweeps else None
    service = RetentionService(run_store, sweep_store=sweep_store)

    keep_tags = frozenset(args.keep_tag) if args.keep_tag else frozenset({"reported"})
    policy = RetentionPolicy(
        keep_last=args.keep_last,
        keep_best=args.keep_best,
        keep_every=args.keep_every,
        metric_key=args.metric_key,
        metric_mode=args.metric_mode,
        keep_tags=keep_tags,
        protect_status=frozenset(args.protect_status),
        min_age_days=args.min_age_days,
        artifact_ttl_days=args.artifact_ttl_days,
        protect_sweep_runs=not args.no_protect_sweep_runs,
        sweep_ttl_days=args.sweep_ttl_days,
    )
    try:
        plan = service.plan(policy)
    except ValueError as exc:
        print(f"invalid retention policy: {exc}")
        return 2

    print(plan.summary(include_kept=args.verbose))
    result = service.apply(plan, dry_run=not args.apply)
    print()
    print(result.summary())
    if not args.apply:
        print("dry-run; pass --apply to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
