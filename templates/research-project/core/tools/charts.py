"""Render static experiment charts from the run store.

Charts are a derived projection: they never replace the run store, eval
records or claims. The project declares *what* to draw in
``input/views/charts.yaml``; this tool resolves the data and writes SVG files
plus a Markdown index under ``output/charts/``.

Views:
  experiments  one bar per experiment, from the latest val record in metrics.jsonl
  candidates   one bar per candidate, mean val metric across its runs
  claims       one bar per reported claim

Examples:
  python core/tools/charts.py
  python core/tools/charts.py --output output --views input/views/charts.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.candidates import (  # noqa: E402
    CANDIDATE_ACTIVE_STATUSES,
    CANDIDATE_STATUSES,
    CandidateStore,
)
from dlef.charts import BarChart, ChartBar, render_bar_chart_svg  # noqa: E402
from dlef.claim import ClaimStore  # noqa: E402
from dlef.run import Run, RunStore  # noqa: E402

_DEFAULT_OUTPUT = Path(os.environ.get("DLEF_OUTPUT_ROOT", str(ROOT / "output")))
_DEFAULT_VIEWS = ROOT / "input" / "views" / "charts.yaml"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SOURCES = {"experiments", "candidates", "claims"}
_AGGREGATES = {"best", "mean", "latest"}
_STATUS_COLORS = {
    "candidate": "#4C78A8",
    "promoted": "#2E7D32",
    "rejected": "#9E9E9E",
    "archived": "#BDBDBD",
}


def _load_views(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"view spec not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid view spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"view spec must be a mapping: {path}")

    defaults = payload.get("defaults") or {}
    charts = payload.get("charts") or []
    if not isinstance(defaults, dict) or not isinstance(charts, list) or not charts:
        raise SystemExit(f"view spec needs a non-empty 'charts' list: {path}")

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(charts):
        if not isinstance(item, dict):
            raise SystemExit(f"chart #{index} must be a mapping")
        spec = {**defaults, **item}
        chart_id = str(spec.get("id", "")).strip()
        source = str(spec.get("source", "")).strip()
        metric = str(spec.get("metric", "")).strip()
        mode = str(spec.get("mode", "min")).strip()
        aggregate = str(spec.get("aggregate", "best")).strip()
        if not _SAFE_ID.match(chart_id):
            raise SystemExit(f"chart #{index}: invalid id {chart_id!r}")
        if chart_id in seen:
            raise SystemExit(f"duplicate chart id: {chart_id}")
        if source not in _SOURCES:
            raise SystemExit(
                f"chart {chart_id}: source must be one of {sorted(_SOURCES)}"
            )
        if not metric:
            raise SystemExit(f"chart {chart_id}: metric is required")
        if mode not in {"min", "max"}:
            raise SystemExit(f"chart {chart_id}: mode must be 'min' or 'max'")
        if aggregate not in _AGGREGATES:
            raise SystemExit(
                f"chart {chart_id}: aggregate must be one of {sorted(_AGGREGATES)}"
            )
        try:
            max_items = int(spec.get("max_items", 20))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"chart {chart_id}: max_items must be an integer") from exc
        if max_items < 0:
            raise SystemExit(f"chart {chart_id}: max_items must be >= 0")
        statuses = spec.get("statuses") or list(CANDIDATE_ACTIVE_STATUSES)
        if not isinstance(statuses, list):
            raise SystemExit(f"chart {chart_id}: statuses must be a list")
        unknown_statuses = {str(value) for value in statuses} - set(CANDIDATE_STATUSES)
        if unknown_statuses:
            raise SystemExit(
                f"chart {chart_id}: unknown status(es): "
                + ", ".join(sorted(unknown_statuses))
            )
        exclude_tags = spec.get("exclude_tags") or []
        if not isinstance(exclude_tags, list):
            raise SystemExit(f"chart {chart_id}: exclude_tags must be a list")
        seen.add(chart_id)
        specs.append(
            {
                "id": chart_id,
                "title": str(
                    spec.get("title") or f"{source} — {metric}"
                ).replace("{metric}", metric),
                "source": source,
                "metric": metric,
                "mode": mode,
                "aggregate": aggregate,
                "max_items": max_items,
                "unit": str(spec.get("unit", "")),
                "statuses": {str(value) for value in statuses},
                "exclude_tags": {str(value) for value in exclude_tags},
            }
        )
    return specs


def _val_value(run: Run, metric: str) -> float | None:
    value = run.latest_metrics("val").get(metric)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _ordered(bars: list[ChartBar], mode: str) -> list[ChartBar]:
    return sorted(
        bars,
        key=lambda bar: (bar.value if mode == "min" else -bar.value, bar.label),
    )


def _limited(bars: list[ChartBar], max_items: int) -> list[ChartBar]:
    return bars if max_items == 0 else bars[:max_items]


def _experiment_chart(
    spec: dict[str, Any], runs: list[Run]
) -> BarChart:
    metric = spec["metric"]
    groups: dict[tuple[str, str], list[tuple[Run, float]]] = defaultdict(list)
    for run in runs:
        if spec["exclude_tags"] & set(run.manifest.tags):
            continue
        value = _val_value(run, metric)
        if value is None:
            continue
        groups[(run.manifest.project, run.manifest.experiment)].append((run, value))

    multi_project = (
        len({items[0][0].manifest.project for items in groups.values()}) > 1
    )
    bars: list[ChartBar] = []
    for (project, experiment), items in groups.items():
        values = [value for _, value in items]
        if spec["aggregate"] == "mean":
            value = sum(values) / len(values)
        elif spec["aggregate"] == "latest":
            _, value = max(
                items, key=lambda item: (item[0].manifest.started_at or "")
            )
        else:
            value = min(values) if spec["mode"] == "min" else max(values)
        label = f"{project}/{experiment}" if multi_project else experiment
        bars.append(
            ChartBar(
                label=label,
                value=value,
                low=min(values),
                high=max(values),
                note=f"{len(values)} run(s) · {spec['aggregate']}",
            )
        )
    bars = _limited(_ordered(bars, spec["mode"]), spec["max_items"])
    return BarChart(
        chart_id=spec["id"],
        title=spec["title"],
        subtitle=f"{len(groups)} experiment(s) · latest val record per run",
        mode=spec["mode"],
        unit=spec["unit"],
        bars=tuple(bars),
        footer="Derived from output/runs; rebuild with `make charts`.",
    )


def _candidate_chart(
    spec: dict[str, Any],
    candidates: list[Any],
    runs: dict[str, Run],
) -> BarChart:
    metric = spec["metric"]
    bars: list[ChartBar] = []
    for candidate in candidates:
        if candidate.status not in spec["statuses"]:
            continue
        values = [
            value
            for run_id in candidate.run_ids
            if (run := runs.get(run_id)) is not None
            if (value := _val_value(run, metric)) is not None
        ]
        if not values:
            continue
        bars.append(
            ChartBar(
                label=candidate.candidate_id,
                value=sum(values) / len(values),
                low=min(values),
                high=max(values),
                note=(
                    f"{candidate.status} · "
                    f"{len(values)}/{len(candidate.run_ids)} run(s)"
                ),
                color=_STATUS_COLORS.get(candidate.status, ""),
            )
        )
    bars = _limited(_ordered(bars, spec["mode"]), spec["max_items"])
    return BarChart(
        chart_id=spec["id"],
        title=spec["title"],
        subtitle=f"{len(bars)} candidate(s) · mean val metric across runs",
        mode=spec["mode"],
        unit=spec["unit"],
        bars=tuple(bars),
        footer="Derived from output/candidates; rebuild with `make charts`.",
    )


def _claim_chart(
    spec: dict[str, Any],
    claims: list[Any],
    run_store: RunStore,
) -> BarChart:
    metric = spec["metric"]
    bars: list[ChartBar] = []
    benchmarks: set[str] = set()
    for claim in claims:
        if claim.metric and claim.metric != metric:
            continue
        value = claim.value
        record = None
        if claim.eval_ids:
            found = run_store.find_eval(claim.eval_ids[0])
            if found is not None:
                _, record = found
        if value is None and record is not None:
            raw = record.metrics.get(metric)
            try:
                value = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                value = None
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value or value in {float("inf"), float("-inf")}:
            continue
        note_parts = []
        if record is not None:
            note_parts.append(record.split)
            if record.benchmark_id:
                note_parts.append(record.benchmark_id[:12])
                benchmarks.add(record.benchmark_id)
        else:
            experiment = claim.tags.get("experiment")
            if experiment:
                note_parts.append(str(experiment))
        bars.append(
            ChartBar(
                label=claim.claim_id,
                value=float(value),
                note=" · ".join(note_parts),
            )
        )
    bars = _limited(_ordered(bars, spec["mode"]), spec["max_items"])
    subtitle = f"{len(bars)} claim(s) · resolved from eval records"
    if len(benchmarks) > 1:
        subtitle += " · WARNING: multiple benchmark ids, not directly comparable"
    return BarChart(
        chart_id=spec["id"],
        title=spec["title"],
        subtitle=subtitle,
        mode=spec["mode"],
        unit=spec["unit"],
        bars=tuple(bars),
        footer="Derived from output/claims; rebuild with `make charts`.",
    )


def _write_index(path: Path, charts: list[BarChart]) -> None:
    lines = [
        "# Experiment charts",
        "",
        "Generated by `make charts` from `output/runs`, `output/candidates` and",
        "`output/claims`. Do not edit by hand; change `input/views/charts.yaml`",
        "and rebuild.",
        "",
    ]
    for chart in charts:
        lines.append(f"## {chart.title}")
        lines.append("")
        if not chart.bars:
            lines.append("_No data for this view._")
        else:
            best = min(chart.bars, key=lambda bar: bar.value) if chart.mode == "min" else max(
                chart.bars, key=lambda bar: bar.value
            )
            lines.append(
                f"`{len(chart.bars)}` bar(s); best `{best.label}` = "
                f"`{best.value:.4g}`."
            )
            lines.append("")
            lines.append(f"![{chart.title}]({chart.chart_id}.svg)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="output root (default: ./output, or DLEF_OUTPUT_ROOT)",
    )
    parser.add_argument(
        "--views",
        default=str(_DEFAULT_VIEWS),
        help="chart view spec (default: input/views/charts.yaml)",
    )
    parser.add_argument("--runs", default=None, help="override run store root")
    parser.add_argument("--candidates", default=None, help="override candidate store")
    parser.add_argument("--claims", default=None, help="override claim store")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args()

    output = Path(args.output)
    runs_root = Path(args.runs) if args.runs else output / "runs"
    candidates_root = Path(args.candidates) if args.candidates else output / "candidates"
    claims_root = Path(args.claims) if args.claims else output / "claims"
    charts_root = output / "charts"

    specs = _load_views(Path(args.views))
    run_store = RunStore(runs_root)
    runs = run_store.iter_runs()
    runs_by_id = {run.run_id: run for run in runs}
    candidates = (
        CandidateStore(candidates_root).list() if candidates_root.is_dir() else []
    )
    claims = ClaimStore(claims_root).list() if claims_root.is_dir() else []

    charts: list[BarChart] = []
    for spec in specs:
        if spec["source"] == "experiments":
            chart = _experiment_chart(spec, runs)
        elif spec["source"] == "candidates":
            chart = _candidate_chart(spec, candidates, runs_by_id)
        else:
            chart = _claim_chart(spec, claims, run_store)
        charts.append(chart)

    charts_root.mkdir(parents=True, exist_ok=True)
    for chart in charts:
        (charts_root / f"{chart.chart_id}.svg").write_text(
            render_bar_chart_svg(chart), encoding="utf-8"
        )
    _write_index(charts_root / "index.md", charts)

    # Drop SVGs for views removed from the spec; they are generated files.
    ids = {chart.chart_id for chart in charts}
    for path in charts_root.glob("*.svg"):
        if path.stem not in ids:
            path.unlink()

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": chart.chart_id,
                        "title": chart.title,
                        "bars": len(chart.bars),
                        "path": str(charts_root / f"{chart.chart_id}.svg"),
                    }
                    for chart in charts
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"charts: {charts_root}")
        for chart in charts:
            print(f"  {chart.chart_id}: {len(chart.bars)} bar(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
