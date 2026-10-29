"""Comparison/report helpers. Numbers come from EvalRecord, never hand-copied."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

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
