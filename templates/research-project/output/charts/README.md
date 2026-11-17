# Charts

Generated static charts for humans. Everything here except this README is
written by:

```bash
make charts
# or:
python core/tools/charts.py --output output
```

Files:

```text
output/charts/
  README.md      # this file (tracked)
  index.md       # generated index with embedded SVGs
  *.svg          # generated charts from input/views/charts.yaml
```

Charts are a **derived projection**: they come from `output/runs`,
`output/candidates` and `output/claims`, and can be rebuilt at any time. They are
not a record and must not be edited by hand. Change the view spec
(`input/views/charts.yaml`) and rerun the tool; stale SVGs for removed views are
deleted automatically.

The charts are intentionally static and dependency-free:

- `experiments`: latest validation metric per experiment, with seed spread;
- `candidates`: mean validation metric per candidate;
- `claims`: reported values, resolved from eval records.

Learning curves and live training telemetry stay in TensorBoard. See
`input/views/README.md` for the view schema.
