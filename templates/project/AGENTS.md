# AGENTS.md — project rules

You are in a **DLEF project**. The framework is installed as the `dlef` package;
it is not part of this repository. Read this file before editing.

This project was copied from `templates/project/`. The placeholder is synthetic
1-D regression. Replace/extend the component packages; keep `task.py` as the
composition root.

For the step-by-step runbook, read [`TUTORIAL.md`](TUTORIAL.md); this file is
the contract.

---

## 0. Start here (agent quickstart)

**DLEF in one paragraph.** DLEF is a thin, domain-agnostic experiment OS. It
owns config resolution, run manifests, evaluation, sweeps, retention and
reports; training stacks live in adapters (`adapters/torch` is installed with
the core). A project has three layers: `input/` (declarative configs and frozen
benchmarks), `core/` (project code), `output/` (generated evidence). The core
is installed as the `dlef` package and is never copied into the project.

If `core/` and `input/` already exist next to this file, you are in the
**existing project** case: skip the init block and go to "Existing project".

**New project from the template.** Copy only the template contents, not the
DLEF repository. If the human has not named `<project-dir>`, ask once; do not
invent a path.

```bash
mkdir -p <project-dir>
git -C <DLEF-root> archive HEAD templates/project \
  | tar -x -C <project-dir> --strip-components=2
cd <project-dir>
git init
pip install -e <DLEF-root>   # dlef + adapters
pip install -e .             # project core
export DLEF_DATA_ROOT=<data-root>
```

Do **not** copy `<DLEF-root>/dlef`, `adapters`, `examples`, `docs`, the DLEF
root `tests`, `.git`, or any `output/runs|artifacts|sweeps|test_access`.

**Existing project.** Do not re-initialize. Verify the wiring:

```bash
make test
python core/commands/train.py --config input/configs/base.yaml \
  --set epochs=1 --set experiment=smoke
```

Success means `output/runs/<project>/smoke/<run_id>/` contains
`manifest.json`, `config.resolved.json`, `metrics.jsonl`, `logs/train.log`
and `checkpoints/last.pt`.

**First response = plan, not code.** Read this file, `TUTORIAL.md`, `README.md`,
`core/README.md`, `core/data/README.md`, `core/config.py`, `core/task.py` and
`input/configs/base.yaml`. Then state:

1. the task and the comparison;
2. where the data lives (`$DLEF_DATA_ROOT`), sample ids, label units;
3. the train/val/test membership rules and the grouping that prevents leakage;
4. the frozen benchmark: test membership + label version + eval transform +
   metric protocol;
5. which files you will add/change, and which stay untouched;
6. the 1-2 step smoke test and its expected output.

Ask the human only for data/split/benchmark/metric semantics that cannot be
inferred from the repo. Do not ask "what is DLEF" — this file is the answer.
Do not write code until the plan is confirmed.

---

## 1. Layout

```text
input/                    # layer 1: immutable inputs
  configs/
    base.yaml            # shared baseline / smoke
    experiments/<experiment>/<variant>.yaml
    sweeps/<sweep>.yaml  # optional parameter space
  benchmarks/            # one frozen JSON per benchmark
  README.md

core/                     # layer 2: project source
  config.py              # typed dataclass schema
  data/
    datasets/            # REPLACE: one file per dataset
    splits.py            # split strategies (materialized indices)
    transforms/          # one file per transform
    paths.py             # $DLEF_DATA_ROOT + external layout
    spec.py              # DataSpec + recipe digest
    loaders.py           # DataLoader assembly
    benchmark.py         # frozen test set + metric protocol -> benchmark id
    build.py             # data composition root: registry + make_loaders
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
    train.py             # thin entry: train + eval
    eval.py              # thin entry: separate eval run (val or sealed test)
    sweep.py             # subprocess grid/random sweep
  tools/
    benchmark.py         # freeze / verify a benchmark manifest
    configs.py           # list / validate experiment configs
    runs.py              # list runs with config source + latest metric
    run_info.py          # run card: config -> benchmark -> source files
    _bootstrap.py        # internal: find the core when running in-repo
    README.md
  tests/
    test_smoke.py
    test_components.py
    test_benchmark.py
    test_config_contract.py
    test_command_surface.py
    test_paths.py
  README.md

scripts/                 # project-specific scripts, intentionally unstructured
  README.md

output/                   # layer 3: generated results
  runs/                   # includes logs/train.log per run
  artifacts/
  sweeps/
  test_access/
  claims/                 # one JSON per reported claim
  reports/
  README.md

AGENTS.md                # project-local guide
TUTORIAL.md              # step-by-step runbook
README.md
pyproject.toml
Makefile
```

If you rename `core`, update:

- `pyproject.toml` → `[tool.setuptools.packages.find] include`;
- imports in `core/commands/*.py`, `core/tools/*.py` and `core/tests/*.py`;
- source-path prefixes in `core/tools/run_info.py`;
- any project docs.

### 1.1 Placement matrix

| You are adding... | Put it here | Also update |
|---|---|---|
| dataset reader / acquisition library code | `core/data/` | `data/build.py` if selectable |
| project-specific automation (acquisition, cleaning, query, export) | `scripts/` | no fixed structure; promote reusable logic |
| transform | `core/data/transforms/<name>.py` | `data/build.py`, `DataConfig.validate` |
| split strategy | `core/data/splits.py` | `SPLITTERS`, `DataConfig.validate` |
| model | `core/models/<name>.py` | `models/build.py`, `ModelConfig.validate` |
| loss | `core/losses/<name>.py` | `losses/build.py`, `LossConfig.validate` |
| optimizer / scheduler | `core/optimizers/<name>.py`, `core/schedulers/<name>.py` | matching `build.py` + config |
| metric | `core/metrics/<name>.py` | `metrics/build.py` + `metrics` list |
| benchmark manifest | `input/benchmarks/` | create via `core/tools/benchmark.py` |
| experiment config | `input/configs/experiments/<experiment>/<variant>.yaml` | `core/tools/configs.py list --check` |
| generic maintenance utility (provided by template) | `core/tools/` | keep generic; do not dump project scripts |
| test | `core/tests/test_*.py` | run `python -m unittest discover -s core/tests` |
| claim | `output/claims/` | `ClaimStore` |
| reusable cross-project code | DLEF core or `adapters/` | not in `core/` |
| generated run output | `output/` | never edit by hand |

**Rules:** no ad-hoc scripts in `core/commands/`; `scripts/` is intentionally
unstructured and project-specific (no video/data assumptions); `core/tools/` holds
only generic maintenance utilities. If a script grows reusable logic, move the
logic into `core/` (or core/adapters) and leave a thin CLI. No `utils.py`
at the project root.

---

## 2. Install / run

```bash
pip install -e /path/to/DLEF   # core
pip install -e .               # project deps (torch, PyYAML)

python -m unittest discover -s core/tests -v
python core/commands/train.py --config input/configs/base.yaml
python core/commands/eval.py \
  --checkpoint output/runs/<project>/<experiment>/<run_id>/checkpoints/last.pt
python core/tools/run_info.py --run <run_id>
python core/tools/runs.py
python core/tools/configs.py list
python core/commands/sweep.py --config input/configs/base.yaml
```

Or:

```bash
make train
make eval CHECKPOINT=output/runs/.../checkpoints/last.pt
make inspect RUN=<run_id>
make runs
make configs
make sweep
make test
make report
make gc-dry
make doctor
```

---

## 3. What to replace / extend

Replace or extend these component packages:

1. `core/data/`
   - One dataset per file in `data/datasets/`; register it in `data/build.py`.
   - One transform per file in `data/transforms/`; register it in `data/build.py`.
   - `data/splits.py` materializes membership (`role -> indices`); loaders never
     call `random_split`.
   - Each dataset builder returns a `DataBundle(dataset=..., spec=DataSpec(...))`.
   - `DataSpec` carries name, version, schema, splits, transform and provenance.
   - Dataset bytes live outside the repo under `$DLEF_DATA_ROOT`; resolve them
     through `data/paths.py` (`curated_dir(name, version)`). `data.version` is
     the dataset content version; never commit images/tables or absolute paths.
   - `data.seed` controls dataset content; the top-level `seed` controls the run.
     Do not use the run seed for the test membership.
   - `data/benchmark.py` freezes `test membership + labels + eval transform +
     metric protocol` into a benchmark id; the config pins benchmark versions.
   - Batch is a `Mapping[str, Any]`; collate/batching belongs in `data/loaders.py`.
   - Data is an ablation axis: `data.name`, `data.version`, `data.split`,
     `data.train_transform`, `data.eval_transform` are config-selected.

2. `core/models/`
   - One file per architecture.
   - `models/build.py` maps `model.name` → builder.
   - Each builder returns `TorchModel(module, forward_fn, device=...)`.
   - The model does no I/O, does not read config, does not move devices, does
     not compute loss.

3. `core/losses/`
   - One file per loss; `losses/build.py` maps `loss.name` → builder.
   - Return `TorchLoss` producing `LossOutput(total=..., terms={...})`.
   - Name every term; make reduction explicit.

4. `core/optimizers/` and `core/schedulers/`
   - One file per implementation; `build.py` maps config names to builders.
   - Defaults (AdamW + cosine) already work; replace only when needed.

5. `core/metrics/`
   - One file per metric; `metrics/build.py` maps names → builders.
   - Return `Metric` objects (`TorchMetric` is the convenience adapter).
   - Add a golden test for each metric.

Keep `core/task.py` as the **task composition root**. It only calls
`build_model`, `build_loss`, `build_optimizer`, `build_scheduler`,
`build_metrics`; it should not contain implementations or data loading.
`core/data/build.py` is the **data composition root**; `core/commands/` wires
the data loaders and the task into one run. Both are config-selected and part of
the ablation surface.

Put reusable, cross-project torch code in the DLEF repo's `adapters/torch/`,
not in `core/`; put project-specific research code in `core/`.

---

## 3.1 Changing data / model / loss mid-experiment

A running job cannot swap data, architecture or loss. **Start a new run; never
edit a completed run.**

The template is config-driven:

- `data.name` selects the dataset builder in `core/data/build.py`;
- `data.version` selects the curated dataset version under `$DLEF_DATA_ROOT`;
- `data.split` selects the splitter in `core/data/splits.py`;
- `data.train_transform` / `data.eval_transform` select transforms in
  `core/data/transforms/`;
- `model.name` selects the builder in `core/models/build.py`;
- `loss.name` selects the builder in `core/losses/build.py`;
- `optim.name` / `scheduler.name` / `metrics[]` select from their registries;
- `core/task.py` stays unchanged.

Data rule: **new data or a new split is a new run.** The loader records the
materialized `DataSpec.version` in the run manifest; never re-split at runtime
and never overwrite a previous split or dataset version. Old runs stay valid.

Benchmark rule: **the test set + eval transform + metric protocol are frozen
together.** If any of them changes, create a new benchmark manifest/version;
never edit an old one in place. Results are comparable only within the same
`benchmark_id`; `dlef compare` / `report` warn across benchmarks.

To add a variant:

1. add `data/datasets/<name>.py` (or `models/<name>.py`, `losses/<name>.py`,
   etc.) with its builder;
2. register it in the matching registry (`data/build.py`, `models/build.py`, ...);
3. allow the name in the matching `Config` validation in `config.py`;
4. add a shape/golden test;
5. commit the code.

One-off variant:

```bash
python core/commands/train.py --config input/configs/base.yaml \
  --set data.name=regression \
  --set data.version=synthetic \
  --set data.split=holdout \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=data-linear-loss-huber
```

For evaluation, just point at the checkpoint:

```bash
python core/commands/eval.py --checkpoint output/runs/.../checkpoints/last.pt
```

`eval.py` reads the training run's `config.resolved.json` via the checkpoint
manifest, so architecture/loss match automatically and the eval run stays
grouped under the training experiment (`...-eval`). `--config` is only a
fallback when the original run directory is unavailable. Do not hand-sync
model/loss flags unless you are deliberately evaluating a different setup.

For a comparison, sweep data/model/loss/seed. `core/commands/sweep.py` already has a
`set_dotted` helper; replace the params with:

```python
params={
    "data.name": Choice(("regression",)),
    "data.split": Choice(("holdout",)),
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

- `core/config.py` is the schema; `input/configs/*.yaml` are values.
- Training loop knobs (`seed`, `epochs`, `max_steps`, `log_every`,
  `checkpoint_every`, `early_stopping`) are top-level; loader knobs
  (`batch_size`, `n_train`, ...) are under `data`.
- Early stopping monitors a validation metric (`monitor`), with `mode=min|max`,
  `patience`, `min_delta` and `min_epochs`. It is disabled by default; log lines
  show the best value and why it stopped.
- **Every behavior-affecting choice is a typed config field.** Data
  name/version/seed/split/transforms, model, loss, optimizer, scheduler,
  metrics, benchmark and training policy all live in the config. No hidden
  module-level constants or `if name == ...` switches.
- Builders receive config objects; `task.py`, `data/build.py` and `core/commands/`
  only wire. They do not read environment-specific overrides.
- Use nested dataclasses (`data`, `model`, `optim`, ...) for real projects.
- Unknown keys are errors by design.
- Add validation in `validate()` (ranges, mutual exclusion).
- One config file per **experiment variant**, not per run or seed:
  `input/configs/experiments/<experiment>/<variant>.yaml`.
- A seed is a run dimension; two seeds reuse the same config. A sweep describes
  a parameter space; generated trial configs stay under
  `output/sweeps/<sweep_id>/configs/`, not in the repo.
- Do not edit `base.yaml` for every experiment; copy/create a variant.
- Discover configs with `python core/tools/configs.py list`; map runs to configs
  with `python core/tools/runs.py`.
- Never put output paths in config; run directories are derived.
- Dataset bytes live under `$DLEF_DATA_ROOT`; config selects logical names and
  versions, not absolute data paths.
- Secrets go through environment variables; never config or logs.
- For a debug run, create a separate debug config (tiny data, 1–2 steps); do not
  add `if debug` branches in model/task code.
- A run is inspected from its run directory: `manifest.json` +
  `config.resolved.json` + eval records. `core/tools/run_info.py --run <id>` prints
  the config-to-source map. The manifest records the original config source and
  code version, so a name cannot silently drift to different source.

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
2. Replace `data/` + model/loss/metrics.
3. Run the smoke test.
4. Run baseline; inspect `manifest.json` + `config.resolved.json` + checkpoint.
5. Evaluate on val; keep test for the final claim.
6. For variants, use sweep; do not write nested loops.
7. For a 3-seed result, add `seed` to the sweep params:
   ```python
   params={"optim.lr": Choice((...)), "seed": Choice((1, 2, 3))}
   # core/commands/sweep.py maps every param with set_dotted; one seed = one run.
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
- DDP training: `torchrun --nproc_per_node=N core/commands/train.py ...`; rank 0 writes
  the run. Multi-rank eval/resume is not built into this template.

---

## 7. Eval / claim rules

- Train/val evaluation is for model selection.
- Final test evaluation is a separate run. It must use a frozen benchmark:
  ```bash
  python core/commands/eval.py \
    --checkpoint ... \
    --split test \
    --benchmark input/benchmarks/synthetic-test-v1.json
  ```
  `eval.py` verifies the current test membership/metric protocol against the
  manifest and refuses to run on a mismatch. `--config` is only a fallback when
  the original run directory is gone.
- Test access is counted under `output/test_access/` by `test_set_id`; report
  shows `WARN` when the same test set is accessed more than once.
- A `Claim` must reference at least one eval id and should pin the benchmark
  id it belongs to:
  ```python
  from dlef.claim import Claim, ClaimStore
  ClaimStore("output/claims").add(
      Claim(
          claim_id="main-result",
          text="...",
          eval_ids=("eval-id",),
          metric="mae",
          benchmark_id="sha256:...",
      )
  )
  ```
- Generate the report, do not hand-copy numbers:
  ```bash
  make report
  # or: python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md
  ```

---

## 8. Cleanup rules

```bash
make gc-dry
# inspect the plan; then:
python -m dlef.cli gc \
  --runs output/runs --sweeps output/sweeps \
  --keep-last 3 --keep-best 1 --metric-key <your-metric> \
  --keep-tag reported --artifact-ttl-days 30 --apply
```

- Default is dry-run; `--apply` is explicit.
- Never `rm -rf output/runs/...` manually.
- Keep `reported`/`baseline` runs; the plan protects parent lineage and
  sweep-referenced runs.
- Artifacts are content-addressed; only unreferenced + TTL-expired ones are
  deleted.

---

## 9. Testing

Every project should have:

- `core/tests/test_smoke.py`: 1–2 steps on tiny data, asserts a manifest/checkpoint;
- a shape/contract test for data/model/loss;
- `core/tests/test_benchmark.py`: test membership stable across train changes;
  benchmark id changes with membership/metric protocol;
- `core/tests/test_config_contract.py`: every config selector resolves to a builder;
- `core/tests/test_command_surface.py`: `core/commands/` stays train/eval/sweep only;
- a golden test for each metric;
- a checkpoint round-trip test if you touch checkpoints.

Run:

```bash
python -m unittest discover -s core/tests -v
```

If the change touches the framework itself, stop: make the core change in the
DLEF repo (with tests), release/upgrade the core, then update this project.
Do not edit the installed `dlef` package in place.

---

## 10. Forbidden

- Do not copy `dlef` core code into this project.
- Do not call `loss.backward()` / `optimizer.step()` inside a Task.
- Do not write to `output/runs/...` by hand.
- Do not edit completed manifests/metrics/eval records.
- Do not delete runs/artifacts manually.
- Do not tune on test; do not report a single lucky seed.
- Do not edit a frozen benchmark manifest; create a new benchmark version.
- Do not run `--split test` without `--benchmark`.
- Do not compare numbers from different `benchmark_id`s as if they were one result.
- Do not make a tracker the only record.
- Do not put secrets in config/logs.
- Do not commit dataset bytes or absolute data paths.
- Do not add ad-hoc scripts to `core/commands/`; `core/commands/` is only train/eval/sweep.
- Project-specific automation goes under `scripts/` (intentionally unstructured).
  `core/tools/` is only for the template's generic maintenance utilities.
  Promote reusable logic to `core/` or core/adapters.
- Do not add project-specific code to the installed core.

---

## 11. Definition of done

1. Config validates; unknown keys fail.
2. `DataSpec` exists; train/val/test membership materialized; one batch validated.
3. Model/loss/metrics implement the contracts; loss terms named.
4. Smoke test + benchmark identity test pass.
5. Train produces manifest + resolved config + metrics + checkpoint.
6. Test eval uses a frozen benchmark and produces an eval record bound to the
   checkpoint digest + benchmark id.
7. Sweep (if used) produces one run per trial and a sweep record.
8. Claim (if reported) resolves to eval → checkpoint → run and names its
   benchmark.
9. `gc` dry-run shows only intended deletions.
10. No project-specific changes in the installed core.

---

## 12. Quick recipes

**New dataset** → add `data/datasets/<name>.py` returning
`DataBundle(dataset=..., spec=DataSpec(...))`; register it in `data/build.py`;
allow the name in `DataConfig`; validate one batch; add a smoke test. Resolve
bytes with `data/paths.py` (`$DLEF_DATA_ROOT`), never with a path in config.
Update `input/configs/*.yaml`.

**New transform** → add `data/transforms/<name>.py` returning a named
`Transform`; register it in `data/build.py`; allow the name in `DataConfig`;
keep train and eval transforms separate; add a test.

**New split strategy** → add a function in `data/splits.py` returning
`role -> list[int]`; register it in `SPLITTERS`; allow the name in
`DataConfig.split`; record the membership digest in `DataSpec.splits`; add a
test that the same split name + seed yields the same membership.

**New benchmark** → change the test membership, label version, eval transform
or metric protocol, then create a new manifest:
```bash
python core/tools/benchmark.py create \
  --config input/configs/base.yaml \
  --out input/benchmarks/my-benchmark-v2.json
```
Never edit the old manifest. Test eval must pass `--benchmark`; `compare` /
`report` show the benchmark id.

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

**New sweep parameter** → edit `core/commands/sweep.py`:
1. add it to `SweepSpec.params`;
2. map it into `trial_raw` in `command_fn`;
3. ensure `train.py` sees it in the trial config.

**New claim** → create `output/claims/<id>.json` via `ClaimStore`; reference eval id(s);
run `dlef report`.

**New adapter** → if reusable, add it to the DLEF repo under `adapters/<stack>/`;
if project-specific, keep it under `core/`.
