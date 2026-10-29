# Scripts

Peripheral scripts are grouped by **purpose**, not dumped in one directory.
`commands/` is only for core experiment execution (`train`, `eval`, `sweep`).

```text
scripts/
  data/
    acquire/     # video stream, download, import, source readers
    prepare/     # clean, label mapping, dedup, convert
    split/       # build membership, leakage checks
  analysis/      # read-only queries, plots, tables, run inspection
  ops/           # benchmark manifests, config catalog, export/migration
  _bootstrap.py  # internal: find the core when running in-repo
```

## Decision tree

1. Is the code imported by `train` / `eval` / `sweep` / tests?
   - Yes → `myproject/` (project-specific) or DLEF `adapters/` (reusable).
   - No → it is a standalone script, continue.
2. Does it acquire or prepare data?
   - → `scripts/data/acquire/`, `prepare/`, or `split/`.
3. Is it read-only analysis / query / plot?
   - → `scripts/analysis/`.
4. Is it benchmark/config/run/export maintenance?
   - → `scripts/ops/`.

Rules:

- Scripts are **thin entry points**; put reusable logic in `myproject/`.
- Scripts write data to `$DLEF_DATA_ROOT`, reports to `reports/` or stdout;
  they never write manifests/metrics/checkpoints by hand.
- No `scripts/misc/`, no top-level `utils.py`, no ad-hoc scripts in
  `commands/`.
- If a script is needed by more than one project, promote its logic to DLEF
  core/adapters; keep only the CLI wrapper here.
