# Experiment configs

One directory per experiment, one YAML per variant:

```text
experiments/
  baseline/
    adamw.yaml
  ablation-lr/
    adamw-lr-0.005.yaml
```

Do not create per-seed files; a seed is a run. Each file is a complete config
(the template has no YAML inheritance); copy `../base.yaml` and change the
variant fields.
