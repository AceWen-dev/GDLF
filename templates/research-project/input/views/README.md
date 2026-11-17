# Chart views

`make charts` reads this file and writes static SVGs plus `output/charts/index.md`.
Charts are a derived projection; the run store, eval records and claims remain
the record. Change this file, not the SVGs.

## Sources

| `source` | Bars | Value |
| --- | --- | --- |
| `experiments` | one per project/experiment | latest **val** metric in each run's `metrics.jsonl`; `aggregate` picks best / mean / latest across runs |
| `candidates` | one per candidate | mean val metric across the candidate's `run_ids`; whiskers show min/max |
| `claims` | one per claim | `claim.value`, or the metric from its eval record if value is empty |

`experiments` deliberately uses the training val stream. Test numbers belong to
claims; do not chart test metrics before a claim exists.

## Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `id` | required | chart id and SVG file name (`[A-Za-z0-9._-]`) |
| `title` | `source — metric` | title; `{metric}` is replaced |
| `source` | required | `experiments`, `candidates`, `claims` |
| `metric` | required | metric key, e.g. `mae` |
| `mode` | `min` | `min` = lower is better, `max` = higher is better |
| `aggregate` | `best` | experiments only: `best`, `mean`, `latest` |
| `unit` | empty | axis/legend unit, e.g. `mm` |
| `max_items` | `20` | top N bars after sorting by value; `0` = all |
| `statuses` | active | candidates only: candidate statuses to include |
| `exclude_tags` | `[]` | experiments only: skip runs carrying these tags |

`defaults` at the top are merged into every chart; per-chart values win.

## Example

```yaml
defaults:
  mode: min
  max_items: 20

charts:
  - id: experiments-val
    title: "Experiments — validation {metric}"
    source: experiments
    metric: mae
    aggregate: best

  - id: candidates-val
    title: "Candidates — validation {metric}"
    source: candidates
    metric: mae

  - id: claims-test
    title: "Reported claims — {metric}"
    source: claims
    metric: mae
```

After changing the primary metric, update the `metric:` values here. A view
whose metric is missing still renders an empty-state chart; it does not fail the
run.
