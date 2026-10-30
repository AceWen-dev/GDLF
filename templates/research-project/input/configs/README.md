# Config layout

One config file per **experiment variant**, not per run and not per seed.

```text
input/configs/
  base.yaml                         # shared baseline / smoke config
  experiments/
    <experiment>/
      <variant>.yaml                # one file per named variant
  sweeps/
    <sweep>.yaml                    # parameter space, optional
```

Examples:

```text
input/configs/experiments/baseline/adamw.yaml
input/configs/experiments/ablation-lr/adamw-lr-0.005.yaml
input/configs/experiments/ablation-loss/huber.yaml
```

Rules:

- A seed is a run dimension. Do not create `seed1.yaml`, `seed2.yaml`, ...
- A sweep describes a parameter space; generated per-trial configs belong under
  `output/sweeps/<sweep_id>/configs/`, not in the repo.
- Keep one variant per file; do not mutate `base.yaml` for every experiment.
- The file path and the `experiment` field should agree.
- `output_root` (default `output`) and `tensorboard` (`enabled`, `subdir`,
  `flush_secs`) are workspace/output settings, not comparison dimensions.
  Keep them in `base.yaml` or a shared variant; changing them does not create a
  new experiment, but it moves the generated run directory.
- The run manifest records `config_source` and `config_digest`; use
  `core/tools/runs.py` to list runs with their config, and `core/tools/run_info.py`
  to inspect one run.

```bash
python core/tools/configs.py list
python core/tools/runs.py --experiment ablation-lr
python core/tools/run_info.py --run <run_id>
```
