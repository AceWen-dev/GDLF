"""Comparison/report helpers. Numbers come from EvalRecord, never hand-copied."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .candidates import CANDIDATE_STATUSES, Candidate
from .claim import Claim
from .eval import EvalRecord
from .run import Run


def format_comparison(records: Sequence[EvalRecord]) -> str:
    if not records:
        return "(no eval records)"

    metric_keys = sorted({key for record in records for key in record.metrics})
    headers = ["eval_id", "split", "benchmark", "checkpoint", *metric_keys]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for record in records:
        row = [
            record.eval_id,
            record.split,
            _short(record.benchmark_id),
            record.checkpoint_digest[:8],
        ]
        row.extend(_format_value(record.metrics.get(key)) for key in metric_keys)
        lines.append("| " + " | ".join(row) + " |")
    warning = _benchmark_warning(records)
    return (warning + "\n" if warning else "") + "\n".join(lines)


def write_comparison(path: str | Path, records: Sequence[EvalRecord], title: str = "Comparison") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{format_comparison(records)}\n", encoding="utf-8")
    return path


def format_claim_report(
    claims: Sequence[Claim],
    resolved: Mapping[str, tuple[Run, EvalRecord]],
) -> str:
    if not claims:
        return "(no claims)"

    headers = [
        "claim",
        "text",
        "metric",
        "value",
        "split",
        "benchmark",
        "eval",
        "run",
        "test_access",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for claim in claims:
        eval_ids = claim.eval_ids or ("",)
        for eval_id in eval_ids:
            item = resolved.get(eval_id)
            if item is None:
                lines.append(
                    f"| {claim.claim_id} | {claim.text} | {claim.metric} | "
                    f"{_format_value(claim.value)} |  | "
                    f"{_short(claim.benchmark_id)} | {eval_id} | MISSING |  |"
                )
                continue
            run, record = item
            value = claim.value
            if value is None and claim.metric:
                value = record.metrics.get(claim.metric)
            rendered = _format_value(value)
            if claim.unit and rendered:
                rendered = f"{rendered} {claim.unit}"
            benchmark = claim.benchmark_id or record.benchmark_id
            if (
                claim.benchmark_id
                and record.benchmark_id
                and claim.benchmark_id != record.benchmark_id
            ):
                benchmark = "MISMATCH"
            test_access = ""
            if record.split == "test":
                test_access = str(record.test_access)
                if record.test_access > 1:
                    test_access = f"WARN {test_access}"
            lines.append(
                f"| {claim.claim_id} | {claim.text} | {claim.metric} | {rendered} | "
                f"{record.split} | {_short(benchmark)} | {eval_id} | "
                f"{run.run_id} | {test_access} |"
            )
    lines.extend(
        [
            "",
            "`WARN` means the test split was accessed more than once for this test set.",
            "Rows with different benchmark ids are not directly comparable.",
            "Every row resolves to a run manifest and a checkpoint digest via its eval record.",
        ]
    )
    return "\n".join(lines)


def write_claim_report(
    path: str | Path,
    claims: Sequence[Claim],
    resolved: Mapping[str, tuple[Run, EvalRecord]],
    title: str = "Experiment report",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{format_claim_report(claims, resolved)}\n", encoding="utf-8")
    return path


_STATUS_ORDER = {name: index for index, name in enumerate(CANDIDATE_STATUSES)}


def candidate_metric(
    candidate: Candidate, runs: Mapping[str, Run]
) -> tuple[float | None, int, int]:
    """Return ``(mean, resolved, total)`` for a candidate's val metric.

    ``resolved`` counts runs that exist and carry the metric; ``total`` is the
    candidate's declared run count, so the report can show partial coverage
    instead of silently averaging over missing runs.
    """

    total = len(candidate.run_ids)
    if not candidate.metric:
        return None, 0, total
    values: list[float] = []
    for run_id in candidate.run_ids:
        run = runs.get(run_id)
        if run is None:
            continue
        value = run.latest_metrics("val").get(candidate.metric)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None, 0, total
    return sum(values) / len(values), len(values), total


def format_candidate_report(
    candidates: Sequence[Candidate], runs: Mapping[str, Run] | None = None
) -> str:
    """Render the shortlist: only candidates, not the full run store."""

    if not candidates:
        return "(no candidates)"

    runs = runs or {}
    headers = [
        "candidate",
        "status",
        "experiment",
        "runs",
        "metric",
        "val",
        "priority",
        "updated",
        "reason",
        "next",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    ordered = sorted(
        candidates,
        key=lambda item: (
            _STATUS_ORDER.get(item.status, len(_STATUS_ORDER)),
            item.priority,
            item.updated_at,
        ),
    )
    for candidate in ordered:
        mean, resolved, total = candidate_metric(candidate, runs)
        if mean is None:
            value = "-"
        else:
            value = _format_value(mean)
            if resolved < total:
                value = f"{value} ({resolved}/{total})"
        experiment = candidate.experiment
        if not experiment:
            first = runs.get(candidate.run_ids[0])
            experiment = first.manifest.experiment if first is not None else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate.candidate_id,
                    candidate.status,
                    experiment or "-",
                    str(total),
                    candidate.metric or "-",
                    value,
                    str(candidate.priority),
                    _clip(candidate.updated_at.replace("T", " ")[:19], 19) or "-",
                    _clip(candidate.reason, 48) or "-",
                    _clip(candidate.next, 40) or "-",
                ]
            )
            + " |"
        )
    distinct_runs = len({run_id for candidate in candidates for run_id in candidate.run_ids})
    return f"{len(candidates)} candidate(s), {distinct_runs} run(s)\n\n" + "\n".join(lines)


def format_report(
    claims: Sequence[Claim],
    resolved: Mapping[str, tuple[Run, EvalRecord]],
    *,
    candidates: Sequence[Candidate] = (),
    runs: Mapping[str, Run] | None = None,
    title: str = "Experiment report",
) -> str:
    """Shortlist first, then claims: the funnel at a glance."""

    sections = [f"# {title}"]
    if candidates:
        sections.append("## Shortlist\n\n" + format_candidate_report(candidates, runs))
    sections.append("## Claims\n\n" + format_claim_report(claims, resolved))
    return "\n\n".join(sections)


def write_report(
    path: str | Path,
    claims: Sequence[Claim],
    resolved: Mapping[str, tuple[Run, EvalRecord]],
    *,
    candidates: Sequence[Candidate] = (),
    runs: Mapping[str, Run] | None = None,
    title: str = "Experiment report",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_report(
            claims,
            resolved,
            candidates=candidates,
            runs=runs,
            title=title,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _clip(value: object, width: int) -> str:
    """Collapse whitespace and truncate for a Markdown table cell."""

    text = " ".join(str(value or "").split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _benchmark_warning(records: Sequence[EvalRecord]) -> str:
    ids = {record.benchmark_id for record in records if record.benchmark_id}
    if len(ids) > 1:
        return (
            "> WARN: multiple benchmark ids in this comparison; "
            "the rows are not directly comparable."
        )
    if ids and len(ids) < len(records):
        return (
            "> WARN: some eval records have no benchmark id; "
            "comparability is not fully verified."
        )
    return ""


def _short(value: str, length: int = 12) -> str:
    return value[:length] if value else ""


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
