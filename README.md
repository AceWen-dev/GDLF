# GDLF / DLEF

General deep learning experiment framework. The Python package is `dlef`.

DLEF is domain-agnostic: it owns contracts, run records, evaluation, sweeps and
reports; training stacks live in adapters.

**Tutorial:** [`docs/TUTORIAL.md`](docs/TUTORIAL.md). **Final report:** [`REPORT.md`](REPORT.md). **Agent guide:** [`AGENTS.md`](AGENTS.md).

> **Hard rule:** the core never contains domain words.
> image / text / audio / video, bbox / depth / mm, classification / detection /
> segmentation, RL / LLM, case / lesion — all of these belong in adapters or
> examples, never in `dlef/`.

## What this is

A thin experiment OS skeleton:

- **Contracts**: Data, Model, Loss, Optimizer, Scheduler, Callback, Metric, Task/Runner, TrainingPolicy.
- **Services**: Config, Run/Artifact, Eval, Sweep, Retention, CLI/Launcher, Registry.
- **Invariants**: one run = one resolved config × one seed; manifest before the
  first step; named loss terms; eval as a separate run; resumable checkpoints;
  local-first run store; tracker optional.

The core has **zero runtime dependencies**. The toy example uses only the
standard library.

## Layout

```text
dlef/
  contracts.py     # Protocol + dataclass contracts
  config.py        # typed schema, strict loader, digest
  registry.py      # name/class_path -> implementation
  run.py           # manifest, run store, content-addressed artifacts
  eval.py          # EvalSpec / EvalRecord
  benchmark.py     # test set + metric protocol -> benchmark id
  claim.py         # Claim / ClaimStore / TestAccessLog
  runner.py        # minimal train/eval loop
  policy.py        # TrainingPolicy: AMP / accumulation / clipping / DDP seam
  checkpoint.py    # CheckpointCodec: JSON default, binary adapters
  sweep.py         # SweepSpec, trials, local GPU-aware queue
  retention.py     # reference-checked retention / gc
  report.py        # comparison table from eval records
  cli.py           # doctor / compare / gc
  launcher.py      # local launcher stub
adapters/torch/    # optional PyTorch adapter (not imported by core)
examples/toy/      # synthetic linear regression; proves the chain
examples/torch_regression/  # synthetic tensor example via the adapter
templates/project/ # copyable project skeleton; imports dlef, does not vendor it
tests/             # unittest suite (works with pytest too)
docs/              # ADR, contract draft
AGENTS.md          # agent guide: project setup, file placement, experiment rules
REPORT.md          # final report
```

## Quick start

```bash
python -m unittest discover -s tests -v
python examples/toy/run.py
python examples/toy/sweep.py
python -m dlef.cli doctor

# optional torch adapter
pip install -e .[torch]
python examples/torch_regression/run.py

# project template smoke test
make template-smoke
```

The toy walks:

```text
Config -> Run -> Eval -> Compare
```

and writes:

```text
examples/toy/output/
  runs/<project>/<experiment>/<run_id>/
    manifest.json
    metrics.jsonl
    checkpoints/last.*      # .json by default, .pt with the torch codec
    evals/<eval_id>.json
  artifacts/<sha[:2]>/<sha>/<name>
  comparison.md
```

## What the toy proves

- The core is domain-agnostic: a batch is an opaque mapping.
- Config is typed, strict, resolved, hashed, and dumped into the run directory.
- The manifest is written before the first step; status is explicit.
- Each run keeps `metrics.jsonl` plus a human-readable `logs/train.log`; sweep
  child stdout/stderr is captured in the sweep trial log.
- Loss terms are named and logged.
- Eval is a separate record linked to a checkpoint digest.
- `compare` reads eval records, not hand-copied numbers.
- Checkpoints round-trip model/optimizer/RNG state.
- The Runner supports epoch-based or step-limited training (`epochs`, `max_steps`)
  and configurable early stopping on a validation metric.
- Callbacks and step/epoch schedulers are lifecycle-tested.
- Grid/random sweeps run one trial per run, with a local GPU-aware queue.
- A failed trial is recorded as `failed` without killing the sweep.
- Retention plans protect lineage, sweep references, and referenced artifacts;
  deletion is dry-run by default and reference-checked.
- The optional torch adapter implements the same contracts against real tensors,
  with a binary checkpoint codec; the core never imports torch.
- The training policy seam supports AMP, gradient accumulation, clipping and
  optional DDP without teaching the core about torch.
- Test-split evaluations are counted per test set; claims link reported numbers
  to eval records and surface test reuse as `WARN`.
- A benchmark pins test membership + labels + eval transform + metric protocol;
  different `benchmark_id`s are reported as not directly comparable.
- Config is the single declaration of choices; the manifest records config
  digest, original config source and code version. The template's
  `core/tools/run_info.py` resolves a run's config to its source files.

It does **not** prove a good model, a real dataset pipeline, or distributed
training. Those are adapter concerns.

## Sweep (one trial = one run)

```python
from dlef.sweep import Choice, LocalSweep, SweepSpec, SweepStore

spec = SweepSpec(
    name="lr-batch",
    params={"lr": Choice((1e-3, 3e-4)), "batch_size": Choice((32, 64))},
    method="grid",
)
record = LocalSweep(
    SweepStore("output/sweeps"),
    gpu_ids=(0, 1),
    max_parallel=2,
).run(spec, trial_fn)
```

`trial_fn(trial, context)` builds the Task/Dataset from `trial.params` and
`context.gpu_ids`, runs `Runner.fit`, and returns `TrialResult`.

Two backends:

- `LocalSweep` + `LocalQueue`: thread-based, in-process `trial_fn`; good for
  the stdlib toy and lightweight tasks.
- `SubprocessSweep` + `SubprocessQueue`: one subprocess per trial,
  `CUDA_VISIBLE_DEVICES` isolation, captured logs, timeout support.

```python
from dlef.sweep import SubprocessSweep, TrialCommand, SweepStore

def command_fn(trial, context):
    return TrialCommand(
        argv=[sys.executable, "train.py", "--config", trial.params["config"]],
        cwd="/path/to/project",
    )

record = SubprocessSweep(
    SweepStore("output/sweeps"),
    gpu_ids=(0, 1),
    max_parallel=2,
).run(spec, command_fn)
```

The child receives:

```text
CUDA_VISIBLE_DEVICES
DLEF_SWEEP_ID
DLEF_TRIAL_ID
DLEF_TRIAL_INDEX
DLEF_GPU_IDS
DLEF_TRIAL_RESULT   # absolute path; write TrialResult.to_dict() JSON here
```

Logs go to `sweeps/<sweep_id>/trials/<trial_id>.log`; result files go next to
them. If the child writes no result file, exit code 0 means `completed` and a
non-zero exit means `failed`.

Run the torch subprocess demo:

```bash
python examples/torch_regression/sweep.py
```

## Retention / gc

```bash
python -m dlef.cli gc \
  --runs runs --sweeps sweeps \
  --keep-last 3 --keep-best 1 --metric-key val_loss \
  --keep-tag reported \
  --artifact-ttl-days 30
```

Default is a dry-run; pass `--apply` to delete. The plan protects:

- last N per experiment;
- best N by an eval metric;
- every Nth run;
- runs with protected tags/status;
- parents of kept runs;
- runs referenced by sweeps (unless `--no-protect-sweep-runs`);
- artifacts referenced by kept runs, or younger than TTL.

The plan is printed before anything is deleted. `--verbose` shows kept items too.

## Optional PyTorch adapter

`adapters/torch/` implements:

- `TorchDataLoader` — `DataLoader` → core `Dataset` (iterable of batches).
- `TorchModel` — `nn.Module` + explicit `forward_fn` → `ModelOutput`.
- `TorchLoss` — callable → `LossOutput`.
- `TorchOptimizer` / `TorchScheduler` — standard PyTorch wrappers.
- `TorchMetric` — accumulate predictions/targets → `MetricResult`.
- `TorchTask` — supervised-style Task; moves the batch to a device.
- `TorchTrainingPolicy` — AMP (`autocast` + `GradScaler`), gradient
  accumulation, gradient clipping, optional DDP.
- `TorchCheckpointCodec` — `torch.save`/`torch.load` for the core `Checkpoint`.

The core has no torch import. Tests skip when torch is absent; in an env with
torch, run the full suite.

## Training policy (AMP / accumulation / clipping / DDP)

The Runner delegates backward/step mechanics to a `TrainingPolicy`. The default
is a no-op; the torch adapter provides:

```python
from adapters.torch import TorchTrainingPolicy

policy = TorchTrainingPolicy(
    amp=True,
    grad_accum_steps=2,
    grad_clip=1.0,
    ddp=False,
)
runner = Runner(..., training_policy=policy)
```

DDP smoke test on CPU:

```bash
CUDA_VISIBLE_DEVICES="" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 examples/torch_regression/run.py
```

Multi-GPU: `torchrun --nproc_per_node=N examples/torch_regression/run.py`.
Rank 0 writes the run, logs and checkpoint; other ranks train without writing.
DDP evaluation and distributed checkpoint resume are v0 gaps.

## Project template

`templates/project/` is the copyable skeleton. It imports `dlef`; it does not
vendor the core. The project is organized into three layers: `input/` (configs
and benchmarks), `core/` (project source), and `output/` (runs, claims,
reports).

```bash
cp -r templates/project ~/my-experiment
cd ~/my-experiment
git init
pip install -e /path/to/DLEF
pip install -e .
python core/commands/train.py --config input/configs/base.yaml
```

See `templates/project/README.md` for the replace points and commands. The
template contains a synthetic placeholder; replace the data/model/loss/metrics
modules with your own.

## Claims / report

A `Claim` links a reported number/table to one or more `EvalRecord`s.

```python
from dlef.claim import Claim, ClaimStore

ClaimStore("claims").add(
    Claim(
        claim_id="main-result",
        text="Our method improves the main metric",
        eval_ids=("eval-id-from-a-run",),
        metric="mae",
        benchmark_id="sha256:...",
    )
)
```

```bash
python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md
```

Test-set access is logged under `runs/../test_access/`, keyed by `test_set_id`,
and shown as `test_access` in the report; access count > 1 is marked `WARN`.
Claims and comparison rows also carry the `benchmark_id`; rows from different
benchmarks are marked as not directly comparable. The report is file-based
Markdown; there is no web UI.

## What is intentionally missing

- Lightning / Accelerate adapters (the torch adapter is the reference).
- YAML configs — JSON works; YAML is optional via PyYAML.
- Tracker integrations (W&B / MLflow / TensorBoard) — the Run Store is canonical.
- HPO adapters (Optuna / Ray Tune).
- Multi-rank DDP evaluation and distributed checkpoint resume.
- A full kill-resume test (checkpoint round-trip is covered).
- A hard claim policy (block vs warn) and a claim/report UI.

## Design docs

- `docs/design/contracts-v0.md` — contract map, invariants, boundaries.
- `docs/adr/0001-thin-experiment-os.md` — why a thin, domain-agnostic core.
- `docs/adr/0002-template-imports-core.md` — template imports the versioned core; no vendoring.
- `docs/adr/0003-benchmark-identity.md` — frozen benchmark: test set + metric protocol.
