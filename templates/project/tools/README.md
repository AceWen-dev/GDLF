# Tools

Project maintenance and inspection scripts live here. They are **not** part of
the training/eval/sweep command surface.

```text
tools/
  benchmark.py   # create / verify a frozen benchmark manifest
  configs.py     # list / validate experiment configs
  runs.py        # list runs with config source + latest metric
  run_info.py    # one run: config -> benchmark -> source files
  _bootstrap.py  # internal: find the core when the template runs in-repo
```

Rules:

- `commands/` is only for core experiment entries: `train.py`, `eval.py`,
  `sweep.py`.
- A new utility goes here first. If it is reusable across projects, promote it
  to the DLEF core or an adapter instead of copying it into every project.
- Tools are read-only with respect to `outputs/runs/`; they do not write
  manifests, metrics or checkpoints.
- Do not add a script to `commands/` just because it is convenient.
