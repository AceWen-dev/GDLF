# Tools

Generic project maintenance utilities shipped with the template. They are not
experiment commands and not project-specific scripts.

```text
core/tools/
  benchmark.py   # create / verify a frozen benchmark manifest
  configs.py     # list / validate experiment configs
  runs.py        # list runs with config source + latest metric
  run_info.py    # one run: config -> benchmark -> source files
  _bootstrap.py  # internal: find the core when running in-repo
```

Rules:

- `core/commands/` is only train/eval/sweep; `core/tools/` is only generic maintenance.
- Project-specific automation goes under `scripts/`, not here.
- Tools do not write run manifests, metrics or checkpoints.
