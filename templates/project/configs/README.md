# Config layout

One config file per **experiment variant**, not per run and not per seed.

```text
configs/
  base.yaml                         # shared baseline / smoke config
  experiments/
    <experiment>/
      <variant>.yaml                # one file per named variant
  sweeps/
    <sweep>.yaml                    # parameter space, optional
```

Examples:

```text
configs/experiments/baseline/adamw.yaml
configs/experiments/ablation-lr/adamw-lr-0.005.yaml
configs/experiments/ablation-loss/huber.yaml
```

Rules:

- A seed is a run dimension. Do not create `seed1.yaml`, `seed2.yaml`, ...
- A sweep describes a parameter space; generated per-trial configs belong under
  `outputs/sweeps/<sweep_id>/configs/`, not in the repo.
- Keep one variant per file; do not mutate `base.yaml` for every experiment.
- The file path and the `experiment` field should agree.
- The run manifest records `config_source` and `config_digest`; use
  `commands/runs.py` to list runs with their config, and `commands/run_info.py`
  to inspect one run.

```bash
python commands/configs.py list
python commands/runs.py --experiment ablation-lr
python commands/run_info.py --run <run_id>
```
