# ADR 0008 — Static charts are derived projections

## Status

Accepted.

## Context

TensorBoard shows live training curves, but it is a mirror and is awkward as a
report artifact: it needs a running server, it cannot be pasted into a paper,
and it does not answer "which experiments are still worth looking at" at a
glance. The output layer still required opening `metrics.jsonl`, eval records
and claims to compare experiments.

At the same time, charts are easy to misread as a source of truth. If a chart
is hand-edited or becomes the only place a number exists, the evidence chain
(`claim → eval → run`) breaks.

## Decision

Add a generic, dependency-free chart layer:

```text
input/views/charts.yaml  ->  core/tools/charts.py  ->  output/charts/*.svg
                                                       output/charts/index.md
```

- `dlef/charts.py` owns only generic SVG rendering: `ChartBar` (label, value,
  low/high whisker, note, color) and `BarChart` (title, subtitle, bars, mode).
  It contains no domain metric names.
- The project declares what to draw in `input/views/charts.yaml`:
  - `experiments`: latest **val** metric per experiment, with seed spread;
  - `candidates`: mean val metric per candidate, colored by status;
  - `claims`: reported values resolved from eval records.
- Charts are written to `output/charts/` and are always rebuildable. Stale
  SVGs for views removed from the spec are deleted on the next run.
- Charts are never the record. RunStore, eval records and claims remain the
  source of truth; a chart can be deleted and regenerated at any time.
- Test metrics are charted only through claims. The `experiments` view uses the
  training val stream on purpose.
- Learning curves stay in TensorBoard; static charts are comparison snapshots
  for reports and triage.

## Consequences

- One command (`make charts`) produces report-ready SVGs plus a Markdown index,
  with no matplotlib or browser dependency.
- The project controls domain metrics without putting domain words in `dlef/`.
- Charts can go stale until rebuilt; that is acceptable because they are
  derived and the index says so.
- A chart is not evidence. Reviewers still follow the claim/eval/run chain.
