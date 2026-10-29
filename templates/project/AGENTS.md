# AGENTS.md — project rules

You are in a **DLEF project**. The framework is installed as the `dlef` package;
it is not part of this repository. Read this file before editing.

This project was copied from `templates/project/`. The placeholder is synthetic
1-D regression. Replace/extend the component packages; keep `task.py` as the
composition root.

---

## 1. Layout

```text
configs/                 # one YAML per experiment/variant
myproject/
  config.py              # typed dataclass schema
  data.py                # REPLACE: dataset + loaders + DataSpec
  models/
    build.py             # name -> builder registry
    mlp.py
    linear.py
  losses/
    build.py
    mse.py
    huber.py
  optimizers/
    build.py
    adamw.py
    adam.py
    sgd.py
  schedulers/
    build.py
    cosine.py
    constant.py
  metrics/
    build.py
    mae.py
  task.py                # composition root only
  runtime.py             # policy/runner/sweep result helper
commands/
  train.py               # thin entry: train + eval
  eval.py                # thin entry: separate eval run
  sweep.py               # subprocess grid/random sweep
tests/
  test_smoke.py
  test_components.py
claims/                  # one JSON per reported claim
outputs/                 # generated; never edit by hand
AGENTS.md                # this file
```

If you rename `myproject`, update:

- `pyproject.toml` → `[tool.setuptools.packages.find] include`;
- imports in `commands/*.py` and `tests/*.py`;
- any project docs.

---

## 2. Install / run

```bash
pip install -e /path/to/DLEF   # core
pip install -e .               # project deps (torch, PyYAML)

python -m unittest discover -s tests -v
python commands/train.py --config configs/base.yaml
python commands/eval.py \
  --checkpoint outputs/runs/<project>/<experiment>/<run_id>/checkpoints/last.pt
python commands/sweep.py --config configs/base.yaml
```

Or:

```bash
make train
make eval CHECKPOINT=outputs/runs/.../checkpoints/last.pt
make sweep
make test
make report
make gc-dry
make doctor
```

---

## 3. What to replace / extend

Replace or extend these component packages:

1. `myproject/data.py`
   - Keep a `DataSpec` (name, version, schema, splits, transform).
   - Return two batch iterables: train and val.
   - Batch is a `Mapping[str, Any]`; collate/batching belongs here.

2. `myproject/models/`
   - One file per architecture.
   - `models/build.py` maps `model.name` → builder.
   - Each builder returns `TorchModel(module, forward_fn, device=...)`.
   - The model does no I/O, does not read config, does not move devices, does
     not compute loss.

3. `myproject/losses/`
   - One file per loss; `losses/build.py` maps `loss.name` → builder.
   - Return `TorchLoss` producing `LossOutput(total=..., terms={...})`.
   - Name every term; make reduction explicit.

4. `myproject/optimizers/` and `myproject/schedulers/`
   - One file per implementation; `build.py` maps config names to builders.
   - Defaults (AdamW + cosine) already work; replace only when needed.

5. `myproject/metrics/`
   - One file per metric; `metrics/build.py` maps names → builders.
   - Return `Metric` objects (`TorchMetric` is the convenience adapter).
   - Add a golden test for each metric.

Keep `myproject/task.py` as the **composition root**. It only calls
`build_model`, `build_loss`, `build_optimizer`, `build_scheduler`,
`build_metrics`; it should not contain implementations or data loading.

Put reusable, cross-project torch code in the DLEF repo's `adapters/torch/`,
not in `myproject/`; put project-specific research code in `myproject/`.

---

## 3.1 Changing model / loss mid-experiment

A running job cannot swap architecture or loss. **Start a new run; never edit a
completed run.**

The template is config-driven:

- `model.name` selects the builder in `myproject/models/build.py`;
- `loss.name` selects the builder in `myproject/losses/build.py`;
- `optim.name` / `scheduler.name` / `metrics[]` select from their registries;
- `myproject/task.py` stays unchanged.

To add a variant:

1. add `myproject/models/<name>.py` (or `losses/<name>.py` etc.) with its builder;
2. register it in the matching `build.py`;
3. allow the name in the matching `Config` validation in `config.py`;
4. add a shape/golden test;
5. commit the code.

One-off variant:

```bash
python commands/train.py --config configs/base.yaml \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=arch-linear-loss-huber
```

For evaluation, just point at the checkpoint:

```bash
python commands/eval.py --checkpoint outputs/runs/.../checkpoints/last.pt
```

`eval.py` reads the training run's `config.resolved.json` via the checkpoint
manifest, so architecture/loss match automatically and the eval run stays
grouped under the training experiment (`...-eval`). `--config` is only a
fallback when the original run directory is unavailable. Do not hand-sync
model/loss flags unless you are deliberately evaluating a different setup.

For a comparison, sweep model/loss/seed. `commands/sweep.py` already has a
`set_dotted` helper; replace the params with:

```python
params={
    "model.name": Choice(("mlp", "linear")),
    "loss.name": Choice(("mse", "huber")),
    "seed": Choice((1, 2, 3)),
},
```

Each trial is a new run with its own config/manifest. Old runs remain valid and
comparable. Commit code before official runs; a dirty worktree is exploratory
only. Report the comparison through `dlef compare` / `dlef report`.

---

## 4. Config rules

- `myproject/config.py` is the schema; `configs/*.yaml` are values.
- Use nested dataclasses (`data`, `model`, `optim`, ...) for real projects.
- Unknown keys are errors by design.
- Add validation in `validate()` (ranges, mutual exclusion).
- One config file per experiment/variant. Do not edit `base.yaml` for every run.
- Never put output paths in config; run directories are derived.
- Secrets go through environment variables; never config or logs.
- For a debug run, create a separate debug config (tiny data, 1–2 steps); do not
  add `if debug` branches in model/task code.

---

## 5. Experiment workflow

Use the IDs consistently:

```text
project      research line / repo
experiment   one question or comparison group
run          one resolved config × one seed
eval         one checkpoint × one eval spec × one split
claim        one reported number/table/figure
```

Recommended order:

1. Write the baseline config.
2. Replace data/model/loss/metrics.
3. Run the smoke test.
4. Run baseline; inspect `manifest.json` + `config.resolved.json` + checkpoint.
5. Evaluate on val; keep test for the final claim.
6. For variants, use sweep; do not write nested loops.
7. For a 3-seed result, add `seed` to the sweep params:
   ```python
   params={"optim.lr": Choice((...)), "seed": Choice((1, 2, 3))}
   # commands/sweep.py maps every param with set_dotted; one seed = one run.
   ```
8. Create a claim for each reported number.
9. Run `dlef gc` dry-run; tag runs that must survive (`reported`, `baseline`).
10. Archive manifest/config/report/key checkpoints.

Rules:

- One seed per run. Never loop seeds inside a run.
- Never tune on test.
- Failed runs are still recorded; do not delete them manually.
- Use `LocalSweep` only for lightweight in-process tasks; use `SubprocessSweep`
  for real GPU work.
- One trial = one run; one config file per trial is generated by `sweep.py`.

---

## 6. Training / policy rules

- `Task.train_step` returns `StepOutput(loss=...)`; it does **not** call
  `backward`, `optimizer.step`, or `zero_grad`.
- AMP / accumulation / clipping / DDP are configured via
  `TorchTrainingPolicy` in `runtime.py` / `train.py`.
- `custom_train_step` is only for genuine escapes (GAN, multi-optimizer,
  custom backward). Document why.
- Checkpoints are written by `Runner` + `TorchCheckpointCodec`; do not hand-write
  `.pt` files.
- Resume must restore model, optimizer, scheduler, scaler, RNG, epoch/step,
  policy state. Loading only model weights is not a resume.
- DDP training: `torchrun --nproc_per_node=N commands/train.py ...`; rank 0 writes
  the run. Multi-rank eval/resume is not built into this template.

---

## 7. Eval / claim rules

- Train/val evaluation is for model selection.
- Final test evaluation is a separate run:
  ```bash
  python commands/eval.py --checkpoint ...
  ```
  The eval reuses the training run config; `--config` is only a fallback.
- Test access is counted under `outputs/test_access/`; report shows `WARN` when
  the same data version is accessed more than once.
- A `Claim` must reference at least one eval id:
  ```python
  from dlef.claim import Claim, ClaimStore
  ClaimStore("claims").add(
      Claim(
          claim_id="main-result",
          text="...",
          eval_ids=("eval-id",),
          metric="mae",
      )
  )
  ```
- Generate the report, do not hand-copy numbers:
  ```bash
  make report
  # or: python -m dlef.cli report --runs outputs/runs --claims claims --out report.md
  ```

---

## 8. Cleanup rules

```bash
make gc-dry
# inspect the plan; then:
python -m dlef.cli gc \
  --runs outputs/runs --sweeps outputs/sweeps \
  --keep-last 3 --keep-best 1 --metric-key <your-metric> \
  --keep-tag reported --artifact-ttl-days 30 --apply
```

- Default is dry-run; `--apply` is explicit.
- Never `rm -rf outputs/runs/...` manually.
- Keep `reported`/`baseline` runs; the plan protects parent lineage and
  sweep-referenced runs.
- Artifacts are content-addressed; only unreferenced + TTL-expired ones are
  deleted.

---

## 9. Testing

Every project should have:

- `tests/test_smoke.py`: 1–2 steps on tiny data, asserts a manifest/checkpoint;
- a shape/contract test for data/model/loss;
- a golden test for each metric;
- a checkpoint round-trip test if you touch checkpoints.

Run:

```bash
python -m unittest discover -s tests -v
```

If the change touches the framework itself, stop: make the core change in the
DLEF repo (with tests), release/upgrade the core, then update this project.
Do not edit the installed `dlef` package in place.

---

## 10. Forbidden

- Do not copy `dlef` core code into this project.
- Do not call `loss.backward()` / `optimizer.step()` inside a Task.
- Do not write to `outputs/runs/...` by hand.
- Do not edit completed manifests/metrics/eval records.
- Do not delete runs/artifacts manually.
- Do not tune on test; do not report a single lucky seed.
- Do not make a tracker the only record.
- Do not put secrets in config/logs.
- Do not add project-specific code to the installed core.

---

## 11. Definition of done

1. Config validates; unknown keys fail.
2. `DataSpec` exists; one batch validated; split provenance recorded.
3. Model/loss/metrics implement the contracts; loss terms named.
4. Smoke test passes.
5. Train produces manifest + resolved config + metrics + checkpoint.
6. Eval produces a separate eval record bound to the checkpoint digest.
7. Sweep (if used) produces one run per trial and a sweep record.
8. Claim (if reported) resolves to eval → checkpoint → run.
9. `gc` dry-run shows only intended deletions.
10. No project-specific changes in the installed core.

---

## 12. Quick recipes

**New dataset** → edit `data.py`, add `DataSpec`, validate one batch, add a smoke
test. Update `configs/*.yaml` if the shape/size changes.

**New model** → add `models/<name>.py`; register it in `models/build.py`; allow
the name in `ModelConfig`; add a forward-shape test; set `model.name`.

**New loss term** → add `losses/<name>.py`; register it in `losses/build.py`;
allow the name in `LossConfig`; return `LossOutput(total, terms={...})`; add a
golden test.

**New metric** → add `metrics/<name>.py`; register it in `metrics/build.py`;
add a golden test; add the name to `metrics` in YAML.

**New optimizer/scheduler** → add a file under `optimizers/` or `schedulers/`;
register it in that package's `build.py`; allow the name in the matching config.

**New config field** → edit `config.py`, add validation, update YAML, run a
config test.

**New sweep parameter** → edit `commands/sweep.py`:
1. add it to `SweepSpec.params`;
2. map it into `trial_raw` in `command_fn`;
3. ensure `train.py` sees it in the trial config.

**New claim** → create `claims/<id>.json` via `ClaimStore`; reference eval id(s);
run `dlef report`.

**New adapter** → if reusable, add it to the DLEF repo under `adapters/<stack>/`;
if project-specific, keep it under `myproject/`.
