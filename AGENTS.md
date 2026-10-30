# AGENTS.md — DLEF

This file is for coding agents working in this repository. Read it before editing.

DLEF is a **domain-agnostic experiment framework**. It is not a model zoo and not
a training engine. The core owns contracts, run records, evaluation, sweeps,
retention and reports; training stacks live in adapters.

---

## 0. Two modes

### Mode A — start a new experiment project

Do **not** modify `dlef/` for project-specific needs.

1. Copy `templates/project/` to the new project directory.
2. `cd` into it and `git init`.
3. Read `templates/project/AGENTS.md` — it is the project-local guide.
4. Install core: `pip install -e /path/to/DLEF`.
5. Install project: `pip install -e .`.
6. Replace/extend the component packages:
   - `core/data/` (`datasets/`, `transforms/`, `splits.py`, `loaders.py`,
     `benchmark.py`)
   - `core/models/` + `models/build.py`
   - `core/losses/` + `losses/build.py`
   - `core/optimizers/`、`core/schedulers/`
   - `core/metrics/` + `metrics/build.py`
7. Keep `core/task.py` as the composition root; keep
   `core/runtime.py`, `core/config.py` and `core/commands/` as the wiring
   unless the experiment genuinely needs a new seam.

### Mode B — change the framework itself

1. Read `docs/design/contracts-v0.md` and the ADRs in `docs/adr/`.
2. Keep the core domain-agnostic (see hard rules).
3. Add or update contract tests.
4. Update `README.md` / `REPORT.md` if user-facing behavior changed.
5. Add an ADR if the decision is hard to reverse and surprising later.

---

## 1. Hard rules

1. **No domain words in `dlef/`.** No image/text/audio/video, no bbox/depth/mm,
   no classification/detection/segmentation, no RL/LLM, no case/lesion.
   Domain code belongs in `adapters/`, the project `core/`, or `examples/`.
2. **Dependency direction:** `core ← adapters ← examples / templates`.
   `dlef/` must never import `adapters/`, `examples/` or `templates/`.
3. **One run = one resolved config × one seed.** Seeds are separate runs, not an
   inner loop.
4. **Manifest before the first step.** Status is `running / completed / failed /
   killed`. Completed runs are immutable.
5. **Eval is a separate run.** A train run does not own test numbers.
6. **RunStore is the source of truth.** W&B / MLflow / TensorBoard, if added,
   are indexes/mirrors only.
7. **No writes outside the run directory or artifact store.** Scripts do not
   hand-write into `runs/`.
8. **No manual deletion.** Use `dlef gc` (dry-run first). It checks references,
   lineage, sweep references and artifact TTL.
9. **Do not edit completed `manifest.json` / metrics / eval records.** Create a
   new run or an append-only annotation.
10. **Training mechanics go through `TrainingPolicy`.** Do not call
    `loss.backward()` / `optimizer.step()` inside a Task unless you are writing a
    deliberate `custom_train_step` escape hatch.
11. **Every loss has `total` + named `terms`.** Every metric is bound to a split
    by `EvalSpec`/`EvalRecord`.
12. **Project code never goes into `dlef/`.** If it is reusable across projects,
    it belongs in `adapters/`; otherwise it stays in the project.
13. **A benchmark is frozen.** Test membership + labels + eval transform +
    metric protocol are one `benchmark_id`; changing any of them creates a new
    benchmark. Compare numbers only within the same `benchmark_id`.
14. **Config is the single declaration of choices.** Every behavior-affecting
    knob/selector is a typed config field; builders receive config instead of
    branching on names. The run manifest records config digest, config source
    and code version; `run_info` resolves config → source.
15. **One command surface; scripts are project-specific.** `core/commands/` contains
    only the core experiment entries: train, eval, sweep. `core/tools/` contains only
    generic template maintenance utilities. `scripts/` is intentionally
    unstructured for project-specific automation; promote reusable logic to
    the project `core/` package or DLEF core/adapters. Do not use
    `core/commands/` or `core/tools/` as a dump.

---

## 2. Repository map

| Path | What it is | Agent should |
|---|---|---|
| `dlef/` | domain-agnostic core | change only for framework work, with tests |
| `adapters/torch/` | optional PyTorch adapter | change only for reusable torch behavior |
| `examples/toy/` | stdlib linear regression | use for contract smoke tests |
| `examples/torch_regression/` | torch example + subprocess sweep | use as a reference, not as project code |
| `templates/project/` | copyable project skeleton | copy for new experiments |
| `tests/` | core + adapter tests | keep green; add tests with changes |
| `docs/design/contracts-v0.md` | contract map / invariants | read before changing interfaces |
| `docs/adr/` | architecture decisions | read; add one for hard-to-reverse choices |
| `REPORT.md` | final report | read for the big picture |

Useful commands:

```bash
make test                       # core tests (torch tests skip if absent)
make template-smoke             # template smoke test
python examples/toy/run.py      # stdlib end-to-end
python -m dlef.cli doctor
python -m dlef.cli report --help
```

If a torch Python is available, run the full suite and torch examples with it:

```bash
make test PYTHON=/path/to/torch/python
make template-smoke PYTHON=/path/to/torch/python
python examples/torch_regression/run.py
python examples/torch_regression/sweep.py
```

---

## 3. Starting a project: exact file placement

A project has this shape (from `templates/project/`):

```text
<project>/
  input/                   # layer 1: immutable inputs
    configs/
      base.yaml            # shared baseline / smoke
      experiments/<experiment>/<variant>.yaml
      sweeps/<sweep>.yaml  # optional parameter space
    benchmarks/            # one frozen JSON per benchmark
    README.md
  core/                    # layer 2: project source
    config.py              # typed dataclass schema
    data/
      datasets/            # one file per dataset + build.py registry
      splits.py            # split strategies (materialized indices)
      transforms/          # one file per transform
      paths.py             # $DLEF_DATA_ROOT + external layout
      spec.py              # DataSpec + recipe digest
      loaders.py           # DataLoader assembly
      benchmark.py         # frozen test set + metric protocol
      build.py             # data composition root
    models/                # one file per architecture + build.py registry
    losses/                # one file per loss + build.py registry
    optimizers/            # one file per optimizer + build.py registry
    schedulers/            # one file per scheduler + build.py registry
    metrics/               # one file per metric + build.py registry
    task.py                # composition root only
    runtime.py             # policy/runner/sweep result helper
    commands/
      train.py             # thin entry: train + eval
      eval.py              # thin entry: separate eval run (val or sealed test)
      sweep.py             # subprocess grid/random sweep
    tools/                 # generic maintenance: benchmark/config/run inspection
      benchmark.py
      configs.py
      runs.py
      run_info.py
      _bootstrap.py
    tests/
      test_smoke.py
      test_components.py
      test_benchmark.py
      test_config_contract.py
      test_command_surface.py
  scripts/                 # project-specific scripts, intentionally unstructured
  output/                  # layer 3: generated results
    runs/
    artifacts/
    sweeps/
    test_access/
    claims/                # one JSON per reported claim
    reports/
    README.md
  AGENTS.md                # project-local guide (from the template)
  README.md
  pyproject.toml
  Makefile
```

Placement rules:

- **Dataset / transforms / split / collate** → `core/data/` package:
  `datasets/<name>.py`, `transforms/<name>.py`, `splits.py`, `loaders.py`.
  Always produce a `DataSpec`; materialize splits, never `random_split` at
  runtime.
- **Architecture** → `core/models/<name>.py`; register in
  `models/build.py`.
- **Loss / named terms** → `core/losses/<name>.py`; register in
  `losses/build.py`.
- **Optimizer / scheduler** → `core/optimizers/<name>.py` /
  `core/schedulers/<name>.py`; register in their `build.py`.
- **Metrics** → `core/metrics/<name>.py`; register in
  `metrics/build.py`.
- **Wiring only** → `core/task.py`. It is the composition root; it should
  only call `build_model` / `build_loss` / `build_optimizer` /
  `build_scheduler` / `build_metrics`.
- **Config schema** → `core/config.py`; values →
  `input/configs/experiments/<experiment>/<variant>.yaml`.
- **Core experiment entries** → `core/commands/` only: `train`, `eval`, `sweep`.
- **Project-specific automation** → `scripts/` (intentionally unstructured;
  no task-specific subdirectory assumptions).
- **Generic maintenance utilities** → `core/tools/` (template-provided only).
- **Reusable script logic** → `core/` or core/adapters; keep the script a
  thin CLI. No `utils.py` at project root.
- **Project tests** → `core/tests/`.
- **Claims** → `output/claims/<claim_id>.json` via `ClaimStore`.
- **Generated artifacts** → `output/`; never edit by hand.
- **Reusable adapter** → `adapters/<stack>/` in the DLEF repo, not in a project.
- **Project-specific adapter** → project `core/`; do not force it into the DLEF core.

---

## 4. Experiment organization

Use the four IDs consistently:

| ID | Meaning | Example |
|---|---|---|
| `project` | research line / repo | `my-project` |
| `experiment` | one question or comparison group | `baseline`, `ablation-lr` |
| `run` | one resolved config × one seed | generated run_id |
| `eval` | one checkpoint × one eval spec × one split | generated eval_id |
| `claim` | one reported number/table/figure | `main-result` |

Rules:

- One config file per **experiment variant** under
  `input/configs/experiments/<experiment>/<variant>.yaml`; do not create a config per
  run or per seed. Seeds are run dimensions, not config files.
- Do not mutate `input/configs/base.yaml` for every experiment; create a variant.
- Sweeps describe a parameter space; generated per-trial configs live under
  `output/sweeps/<sweep_id>/configs/`, not in the repo.
- The run manifest records `config_source` and `config_digest`; use
  `core/tools/runs.py` to see which config produced each run and
  `core/tools/run_info.py` for one run's config → source map.
- One seed per run. For a 3-seed result, add `seed` to the sweep params and
  compare the resulting runs; do not loop seeds inside a run.
- Never tune on the test split. Use val for model selection; test belongs to a
  claim.
- Tag runs that must survive cleanup: `baseline`, `reported`, `debug`.
- A failed run/trial is still recorded (`failed` / `partial`); do not delete it.
- To compare variants, use `core/commands/sweep.py` (subprocess) or `LocalSweep`
  (thread/in-process for lightweight cases); one trial = one run.
- To report a number, create a `Claim` with the eval id(s), then run
  `dlef report`.
- To clean up, run `dlef gc` (dry-run first); keep `reported` tagged runs.

Multi-seed sweep example (edit the template’s `core/commands/sweep.py`):

```python
params={
    "optim.lr": Choice((0.02, 0.005)),
    "seed": Choice((1, 2, 3)),
}
# one trial = one run; the template's sweep maps dotted keys into trial configs.
```

Changing data/model/loss mid-experiment:

- A new dataset, split, transform, architecture or loss is a **new run**, never
  an edit to an old run.
- Core never knows datasets/architectures/losses. The project template is
  config-driven: `data.name` / `data.split` / `data.*_transform` select
  `core/data/build.py` + `data/splits.py`; `model.name` selects
  `core/models/build.py`; `loss.name` selects `core/losses/build.py`;
  optimizer/scheduler/metrics have their own registries; `task.py` stays
  unchanged.
- Materialize splits; never call `random_split` at runtime. A new data version
  or split version produces a new `DataSpec.version`, recorded in the run
  manifest; old runs keep their original version.
- One-off variant:
  ```bash
  python core/commands/train.py --config input/configs/base.yaml \
    --set data.name=regression --set data.version=synthetic \
    --set data.split=holdout --set model.name=linear --set loss.name=huber \
    --set experiment=data-linear-loss-huber
  ```
  For evaluation, point at the checkpoint; it reads the training run's resolved
  config, so architecture/loss match automatically:
  ```bash
  python core/commands/eval.py --checkpoint output/runs/.../checkpoints/last.pt
  ```
- Add the component file, register it in the matching `build.py`, extend config
  validation, add a smoke/shape test, and commit the code before official runs.
- For a comparison, sweep `model.name` / `loss.name` / `seed`; one trial = one
  run. The template’s `core/commands/sweep.py` has a `set_dotted` helper.
- Compare old vs new through `dlef compare` or `dlef report`; old runs remain
  valid because each has its own config digest and code version.

---

## 5. Config conventions

- Config is a **typed dataclass tree** (`ConfigSchema` / `TrainConfig`).
- **Every behavior-affecting choice is a config field**: data recipe, split,
  transforms, model, loss, optimizer, scheduler, metrics, benchmark, policy.
  Do not hide switches in module globals, environment variables or `if name ==`
  branches. Environment variables are only for paths (`DLEF_DATA_ROOT`) and
  secrets.
- Builders receive config objects. `task.py` / `data/build.py` / `core/commands/`
  only wire; they do not branch on implementation names.
- Unknown keys are errors. Do not add ad-hoc dict keys.
- Validation belongs in `validate()` / `__post_init__`.
- Every run dumps `config.resolved.json`; its SHA-256 and original
  `config_source` go into the manifest.
- Inspect a run from its directory: `manifest.json` + `config.resolved.json` +
  eval records. `core/tools/run_info.py --run <id>` prints the config-to-source
  map.
- Secrets go through environment variables, never config files.
- Do not put paths to outputs in config; run directories are derived.
- Do not use `if debug` branches inside model/task code; use a separate debug
  config with tiny data and few steps.
- YAML is optional; JSON always works. The template uses YAML.

---

## 6. Data conventions

- Every dataset exposes a `DataSpec`: name, version, schema, splits, transform,
  stats, provenance.
- Dataset / split / transform implementations live in `core/data/`;
  `data/build.py` selects them by config, `task.py` does not load data.
- Dataset **bytes** live outside the repo under `DLEF_DATA_ROOT`; the project
  holds code and small manifests only. Do not commit images/tables or absolute
  data paths.
- A split is a membership view over one curated copy (sample ids or relative
  paths), not a second `train/` `val/` `test/` copy of the data.
- A run pins a **data recipe** (dataset + split + transform). New data, a new
  split or a new transform is a new version and a new run; never overwrite an
  old version.
- `Dataset` is an **iterable of batches** (`Mapping[str, Any]`); batching,
  shuffling, padding and collation belong to the data adapter.
- Data is read-only. Preprocessing writes a new versioned artifact, not an
  in-place mutation.
- Split must be materialized; never do random splitting inside `__getitem__`.
  Loaders read a recorded membership (`role -> indices`), not a fresh seed.
- Test membership is frozen per benchmark and must not depend on the run seed.
  A new test set, label version, eval transform or metric protocol means a new
  benchmark version, not an edit to the old one.
- Train and eval transforms are separate declarations.
- Validate one batch before training (keys, dtypes, shapes).
- Check leakage before a new split: case/group overlap, duplicates, temporal
  leakage. DLEF does not do this automatically.
- Record data version/hash in the manifest; do not rely on directory names.

---

## 7. Model / loss / metric conventions

Model:

- Wrap an `nn.Module` with `TorchModel` and an explicit `forward_fn`.
- Models do no I/O, do not read config, do not move devices, do not compute loss.
- Device placement belongs to `TorchTask` / `TorchModel`.
- Initialization must be reproducible; record the scheme if non-standard.

Loss:

- Return `LossOutput(total=..., terms={...})`.
- Name every term; log every term.
- Make reduction/normalization explicit (per-sample / per-token / sum / mean).
- Loss weights come from config.
- Do not hide the loss inside the model.

Metric:

- Use `TorchMetric` or a custom `Metric` with `update` / `compute` / `reset`.
- Metric implementations do not know the split; `EvalSpec` binds it.
- Add a golden test: a known tiny input gives a known value.

---

## 8. Training / policy conventions

- `Task.train_step(batch)` returns `StepOutput(loss=...)`; it does not call
  `backward`/`step`.
- Standard path: Runner owns backward/step/zero; policy owns AMP/accum/clip/DDP.
- `custom_train_step` is an explicit escape hatch for GANs, multi-optimizer,
  custom backward, etc. Document why it is needed.
- AMP/accumulation/clipping/DDP are configured via `TorchTrainingPolicy`
  (`amp`, `grad_accum_steps`, `grad_clip`, `ddp`).
- Checkpoints go through `CheckpointCodec`; do not hand-write `.pt` files.
- Checkpoint must include model/optimizer/scheduler/scaler/RNG/epoch/step and
  policy state.
- Do not resume by loading only model weights; optimizer/RNG matter.

---

## 9. Eval / claim conventions

- Training eval is for model selection. Final evaluation is a separate run and
  must use a frozen benchmark.
- `EvalRecord` carries split, checkpoint digest, data version, eval spec digest,
  metrics, test access, `benchmark_id` and `test_set_id`.
- Test split access is counted per `test_set_id` in `output/test_access/`;
  report shows `WARN` when count > 1. Changing the eval transform must not reset
  the count.
- A `Claim` must reference at least one eval id and should pin its
  `benchmark_id`. Compare only records with the same `benchmark_id`.
- Report generation:

```bash
python -m dlef.cli report \
  --runs output/runs \
  --claims claims \
  --out report.md
```

- Do not hand-copy numbers into a paper/report. Let the claim/report tool
  resolve them.

---

## 10. Sweep / run / cleanup conventions

- One trial = one run. Use `SweepSpec` (`grid` or `random`).
- `LocalSweep` is thread-based and in-process; `SubprocessSweep` is the default
  for real GPU work.
- Subprocess contract:
  - `CUDA_VISIBLE_DEVICES`, `DLEF_SWEEP_ID`, `DLEF_TRIAL_ID`,
    `DLEF_TRIAL_INDEX`, `DLEF_GPU_IDS`, `DLEF_TRIAL_RESULT` are set.
  - The child writes `TrialResult.to_dict()` JSON to `DLEF_TRIAL_RESULT`.
  - Logs go to `sweeps/<sweep_id>/trials/<trial_id>.log`.
- Do not run nested loops in a single process for variants.
- RunStore is canonical; do not rely on logs alone.
- Retention: `dlef gc` (dry-run default). Protect `reported`/`baseline` tags,
  parents of kept runs, sweep references, referenced artifacts.
- Never `rm -rf runs/...` manually.

---

## 11. Testing and verification

For framework changes:

```bash
make test
make template-smoke
python -m compileall -q dlef adapters examples templates tests
```

With torch:

```bash
make test PYTHON=/path/to/torch/python
make template-smoke PYTHON=/path/to/torch/python
python examples/torch_regression/run.py
python examples/torch_regression/sweep.py
```

For project changes:

```bash
python -m unittest discover -s core/tests -v
python core/commands/train.py --config input/configs/base.yaml
```

Every project should have:

- a smoke test that runs 1–2 steps on tiny data;
- a shape/contract test for data/model/loss/metric;
- a benchmark identity test: test membership stable across train changes,
  benchmark id changes with membership/metric protocol;
- a config-contract test: every selector in the config resolves to a builder;
- a command-surface test: `core/commands/` stays train/eval/sweep only;
- a golden metric test;
- a checkpoint round-trip test if you touch checkpoints.

---

## 12. What a coding agent must not do

- Do not add domain words to `dlef/`.
- Do not copy `dlef/` into a project template.
- Do not write a bare training loop outside `Runner`/`TrainingPolicy`.
- Do not write `runs/` / `manifest.json` by hand.
- Do not edit completed metrics/manifests/eval records.
- Do not delete runs/artifacts without `dlef gc`.
- Do not tune on test; do not report a single lucky seed as the result.
- Do not make W&B/MLflow/TensorBoard the only record.
- Do not store secrets in config or logs.
- Do not commit dataset bytes or absolute data paths; use `DLEF_DATA_ROOT`.
- Do not duplicate dataset bytes per split; store one curated copy + membership.
- Do not edit a frozen benchmark or run test eval without one.
- Do not compare numbers across different `benchmark_id`s.
- Do not change a contract without updating tests and the design doc.

---

## 13. Definition of done

A project task is done when:

1. `input/configs/*.yaml` is typed, strict, and validated.
2. `DataSpec` exists; one batch is validated; split provenance is recorded.
3. Model/loss/metric contracts are implemented; loss terms are named.
4. `core/tests/` smoke test passes on tiny data.
5. `core/commands/train.py` produces a run with manifest, resolved config, metrics,
   checkpoint.
6. `core/commands/eval.py` produces an eval record bound to a checkpoint digest.
7. Sweep (if used) produces one run per trial and a sweep record.
8. A claim (if reported) resolves to eval → checkpoint → run.
9. `dlef gc` dry-run shows only intended deletions.
10. No core changes were needed, or core changes have tests + docs.

---

## 14. Where to look next

- `docs/TUTORIAL.md` — step-by-step: copy template → train → eval → sweep → report → gc.
- `REPORT.md` — the big picture and verified results.
- `README.md` — quickstart and commands.
- `templates/project/AGENTS.md` — project-local rules.
- `templates/project/TUTORIAL.md` — project-local runbook (copied with the template).
- `docs/design/contracts-v0.md` — contract map and invariants.
- `docs/adr/0001-thin-experiment-os.md` — why thin, domain-agnostic core.
- `docs/adr/0002-template-imports-core.md` — why the template does not vendor core.
- `docs/adr/0003-benchmark-identity.md` — why test set + metric protocol are a frozen benchmark.
- `examples/torch_regression/` — a complete torch + sweep example.

If a requirement is genuinely ambiguous, ask the human once, then follow the
rules above. Do not invent a new framework layer for a project-specific need.
