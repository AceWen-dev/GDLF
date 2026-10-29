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
6. Replace the four placeholder modules:
   - `myproject/data.py`
   - `myproject/model.py`
   - `myproject/loss.py`
   - `myproject/metrics.py`
7. Keep `myproject/task.py`, `myproject/runtime.py`, `myproject/config.py` and
   `commands/` as the wiring unless the experiment genuinely needs a new seam.

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
   Domain code belongs in `adapters/`, `myproject/`, or `examples/`.
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
  configs/
    base.yaml              # one config per baseline / experiment variant
  myproject/
    config.py              # typed dataclass schema
    data.py                # dataset + loaders + DataSpec
    model.py               # architecture
    loss.py                # loss + named terms
    metrics.py             # metrics
    task.py                # wire model/loss/optimizer/scheduler/metrics
    runtime.py             # policy/runner/sweep result helper
  commands/
    train.py               # thin entry: train + eval
    eval.py                # thin entry: separate eval run
    sweep.py               # subprocess grid/random sweep
  tests/
    test_smoke.py
  claims/                  # one JSON per reported claim (created by ClaimStore)
  outputs/                 # generated: runs / artifacts / sweeps / test_access
  AGENTS.md                # project-local guide (from the template)
```

Placement rules:

- **Dataset / transforms / collate** → `myproject/data.py` (or a `myproject/data/`
  package if it grows). Always produce a `DataSpec`.
- **Architecture** → `myproject/model.py` (or `myproject/models/`).
- **Loss / named terms** → `myproject/loss.py` (or `myproject/losses/`).
- **Metrics** → `myproject/metrics.py` (or `myproject/metrics/`).
- **Wiring only** → `myproject/task.py`. It is the only place that knows the
  domain-specific combination.
- **Config schema** → `myproject/config.py`; values → `configs/*.yaml`.
- **Entry points** → `commands/`; keep them thin.
- **Project tests** → `tests/`.
- **Claims** → `claims/<claim_id>.json` via `ClaimStore`.
- **Generated artifacts** → `outputs/`; never edit by hand.
- **Reusable adapter** → `adapters/<stack>/` in the DLEF repo, not in a project.
- **Project-specific adapter** → `myproject/`; do not force it into core.

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

- One config file per experiment/variant. Do not mutate `configs/base.yaml` for
  every run; copy it or let a sweep generate per-trial configs.
- One seed per run. For a 3-seed result, add `seed` to the sweep params and
  compare the resulting runs; do not loop seeds inside a run.
- Never tune on the test split. Use val for model selection; test belongs to a
  claim.
- Tag runs that must survive cleanup: `baseline`, `reported`, `debug`.
- A failed run/trial is still recorded (`failed` / `partial`); do not delete it.
- To compare variants, use `commands/sweep.py` (subprocess) or `LocalSweep`
  (thread/in-process for lightweight cases); one trial = one run.
- To report a number, create a `Claim` with the eval id(s), then run
  `dlef report`.
- To clean up, run `dlef gc` (dry-run first); keep `reported` tagged runs.

Multi-seed sweep example (edit the template’s `commands/sweep.py`):

```python
params={
    "optim.lr": Choice((0.02, 0.005)),
    "seed": Choice((1, 2, 3)),
}
# one trial = one run; the template's sweep maps dotted keys into trial configs.
```

Changing model/loss mid-experiment:

- A new architecture or loss is a **new run**, never an edit to an old run.
- Core never knows architectures/losses. The project template is config-driven:
  `model.name` selects `myproject/model.py`; `loss.name` selects
  `myproject/loss.py`; `task.py` stays unchanged.
- One-off variant:
  ```bash
  python commands/train.py --config configs/base.yaml \
    --set model.name=linear --set loss.name=huber \
    --set experiment=arch-linear-loss-huber
  ```
  For evaluation, point at the checkpoint; it reads the training run's resolved
  config, so architecture/loss match automatically:
  ```bash
  python commands/eval.py --checkpoint outputs/runs/.../checkpoints/last.pt
  ```
- Add the builder branch, extend config validation, add a smoke/shape test, and
  commit the code before official runs.
- For a comparison, sweep `model.name` / `loss.name` / `seed`; one trial = one
  run. The template’s `commands/sweep.py` has a `set_dotted` helper.
- Compare old vs new through `dlef compare` or `dlef report`; old runs remain
  valid because each has its own config digest and code version.

---

## 5. Config conventions

- Config is a **typed dataclass tree** (`ConfigSchema` / `TrainConfig`).
- Unknown keys are errors. Do not add ad-hoc dict keys.
- Validation belongs in `validate()` / `__post_init__`.
- Every run dumps `config.resolved.json`; its SHA-256 goes into the manifest.
- Secrets go through environment variables, never config files.
- Do not put paths to outputs in config; run directories are derived.
- Do not use `if debug` branches inside model/task code; use a separate debug
  config with tiny data and few steps.
- YAML is optional; JSON always works. The template uses YAML.

---

## 6. Data conventions

- Every dataset exposes a `DataSpec`: name, version, schema, splits, transform,
  stats, provenance.
- `Dataset` is an **iterable of batches** (`Mapping[str, Any]`); batching,
  shuffling, padding and collation belong to the data adapter.
- Data is read-only. Preprocessing writes a new versioned artifact, not an
  in-place mutation.
- Split must be materialized; never do random splitting inside `__getitem__`.
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

- Training eval is for model selection. Final evaluation is a separate run.
- `EvalRecord` carries split, checkpoint digest, data version, eval spec digest,
  metrics, test access.
- Test split access is counted per data version in
  `outputs/test_access/`; report shows `WARN` when count > 1.
- A `Claim` must reference at least one eval id.
- Report generation:

```bash
python -m dlef.cli report \
  --runs outputs/runs \
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
python -m unittest discover -s tests -v
python commands/train.py --config configs/base.yaml
```

Every project should have:

- a smoke test that runs 1–2 steps on tiny data;
- a shape/contract test for data/model/loss/metric;
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
- Do not change a contract without updating tests and the design doc.

---

## 13. Definition of done

A project task is done when:

1. `configs/*.yaml` is typed, strict, and validated.
2. `DataSpec` exists; one batch is validated; split provenance is recorded.
3. Model/loss/metric contracts are implemented; loss terms are named.
4. `tests/` smoke test passes on tiny data.
5. `commands/train.py` produces a run with manifest, resolved config, metrics,
   checkpoint.
6. `commands/eval.py` produces an eval record bound to a checkpoint digest.
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
- `docs/design/contracts-v0.md` — contract map and invariants.
- `docs/adr/0001-thin-experiment-os.md` — why thin, domain-agnostic core.
- `docs/adr/0002-template-imports-core.md` — why the template does not vendor core.
- `examples/torch_regression/` — a complete torch + sweep example.

If a requirement is genuinely ambiguous, ask the human once, then follow the
rules above. Do not invent a new framework layer for a project-specific need.
