# Project template

A copyable project skeleton for DLEF. It imports the `dlef` core; it does **not**
vendor or copy the core.

**Agent guide:** [`AGENTS.md`](AGENTS.md) — read this before editing, or before
asking a coding agent to edit.

**Tutorial:** [`TUTORIAL.md`](TUTORIAL.md) — copy → install → smoke → first
claim, written for the project directory (it travels with the template).

**Project memory:** [`PROJECT.yaml`](PROJECT.yaml) is the stable brief;
[`STATE.md`](STATE.md) is the current snapshot (read and update it);
[`DECISIONS.md`](DECISIONS.md) is the append-only decision log.

## Start a new project

Install the core once, then let `dlef init` copy the template and fill metadata:

```bash
pip install -e /path/to/DLEF
dlef init ~/code/my-project --project my-project
cd ~/code/my-project
pip install -e .               # this project
export DLEF_DATA_ROOT=/path/to/data
make test
```

`dlef init` copies only the template contents, runs `git init`, and writes
project metadata into `PROJECT.yaml`, `STATE.md`, `pyproject.toml` and
`base.yaml`. It does not install packages or fill the research fields
(question/data/split/benchmark/metric).

Fallback (older core, or if you want a custom copy):

```bash
mkdir -p ~/code/my-project
git -C /path/to/DLEF archive HEAD templates/research-project \
  | tar -x -C ~/code/my-project --strip-components=2
cd ~/code/my-project
git init
pip install -e /path/to/DLEF
pip install -e .
export DLEF_DATA_ROOT=/path/to/data
make test
```

Then read [`AGENTS.md`](AGENTS.md), [`STATE.md`](STATE.md),
[`DECISIONS.md`](DECISIONS.md), [`PROJECT.yaml`](PROJECT.yaml) and
[`TUTORIAL.md`](TUTORIAL.md), or give them to your coding agent.
`AGENTS.md` contains the framework orientation, the one-time init/verify steps,
the placement matrix, and the "plan before code" protocol; `TUTORIAL.md` is the
step-by-step runbook. Fill `PROJECT.yaml` and `STATE.md` with the real project
question/data before the first real run; do not leave placeholders.

`dlef/` and `adapters/` are installed, not copied. Data bytes stay outside the
repo under `$DLEF_DATA_ROOT`.

## What is here

```text
input/                    # layer 1: immutable inputs
  configs/base.yaml       # shared baseline / smoke config
  configs/experiments/    # one YAML per experiment variant
  configs/sweeps/         # optional parameter-space specs
  benchmarks/             # one frozen JSON per benchmark
  README.md

core/                     # layer 2: project source
  config.py               # typed config schema
  data/                   # datasets, splits, transforms, loaders
    datasets/
    splits.py
    transforms/
    paths.py              # $DLEF_DATA_ROOT
    spec.py
    loaders.py
    benchmark.py
    build.py
  models/                 # one file per architecture + build.py
  losses/                 # one file per loss + build.py
  optimizers/
  schedulers/
  metrics/
  task.py                 # composition root
  runtime.py              # policy / runner / sweep result helper
  tensorboard.py          # optional TensorBoard mirror callback
  commands/
    train.py
    eval.py
    sweep.py
  tools/                  # generic maintenance utilities
  tests/
    test_smoke.py
    test_components.py
    test_benchmark.py
    test_config_contract.py
    test_command_surface.py
  README.md

scripts/                  # project-specific scripts, intentionally free-form
  README.md

output/                   # layer 3: generated results
  runs/
  artifacts/
  sweeps/
  test_access/
  claims/
  reports/
  README.md

AGENTS.md                 # project-local agent guide
PROJECT.yaml              # stable project brief (what we are doing)
STATE.md                  # current project snapshot (read + update)
DECISIONS.md              # append-only decision log
TUTORIAL.md               # step-by-step runbook for this project
README.md
pyproject.toml
Makefile
```

The placeholder is synthetic 1-D regression. Replace/extend the component
packages; keep `task.py` as the composition root and keep the entry points thin.

## Components

Each component is one file plus a small registry:

| Registry | Selects by | Implementation |
|---|---|---|
| `data/build.py` | `data.name` | `data/datasets/<name>.py` |
| `data/build.py` | `data.train_transform` / `data.eval_transform` | `data/transforms/<name>.py` |
| `data/splits.py` | `data.split` | `data/splits.py` |
| `models/build.py` | `model.name` | `models/<name>.py` |
| `losses/build.py` | `loss.name` | `losses/<name>.py` |
| `optimizers/build.py` | `optim.name` | `optimizers/<name>.py` |
| `schedulers/build.py` | `scheduler.name` | `schedulers/<name>.py` |
| `metrics/build.py` | `metrics[]` | `metrics/<name>.py` |

`data/build.py` is the data composition root; `task.py` composes the trainable
task. `core/commands/train.py` wires both into a run, so data, model, loss and
metrics are all selectable from the same config.

To add a dataset:

```python
# core/data/datasets/my_dataset.py
def build_my_dataset(config, role, transform):
    ...
    return DataBundle(dataset=..., spec=DataSpec(...))

# core/data/build.py
from .datasets.my_dataset import build_my_dataset
DATASETS = {"regression": build_regression, "my_dataset": build_my_dataset}
```

Then allow `"my_dataset"` in `DataConfig.validate` and set
`data.name: my_dataset`. Splits are materialized in `data/splits.py`; the
runtime loader never calls `random_split`.

To add a model:

```python
# core/models/resnet.py
def build_resnet(config, device):
    return TorchModel(ResNet(config.depth), lambda m, b: m(b["x"]), device=device)

# core/models/build.py
from .resnet import build_resnet
BUILDERS = {"mlp": build_mlp, "linear": build_linear, "resnet": build_resnet}
```

Then allow `"resnet"` in `ModelConfig.validate` and set `model.name: resnet`
in the config. The run's `config.resolved.json` records exactly which name was
used; `models/build.py` gets you to the file.

## External data

Do not put dataset bytes in this repository. Set `DLEF_DATA_ROOT` to a
directory outside the repo:

```bash
export DLEF_DATA_ROOT=/path/to/datasets
```

Standard layout:

```text
$DLEF_DATA_ROOT/
  raw/<source>/...
  curated/<dataset>/<version>/...
  splits/<dataset>/<version>/<split>.json
```

`core/data/paths.py` resolves these paths. A real dataset builder uses
`curated_dir(config.name, config.version)` and
`split_manifest_path(config.name, config.version, split)`. The project keeps
code and small manifests; the images/tables themselves stay outside git. A
split manifest stores membership (sample ids or relative paths), not a copied
`train/` `val/` `test/` tree — one curated copy serves every split.

The project `.gitignore` blocks `/data/`, `/datasets/` and `/raw/` at the
project root, so accidental copies do not get committed. `core/data/`
(code) is not ignored.

## Frozen benchmark / test set

A benchmark is the unit of comparison:

```text
benchmark = test membership + labels
          + eval transform
          + metric implementation/version + aggregation + thresholds
```

Freeze it once and commit the small manifest:

```bash
python core/tools/benchmark.py create \
  --config input/configs/base.yaml \
  --out input/benchmarks/synthetic-test-v1.json

python core/tools/benchmark.py verify \
  --config input/configs/base.yaml \
  --benchmark input/benchmarks/synthetic-test-v1.json
```

Test evaluation must go through the frozen manifest:

```bash
python core/commands/eval.py \
  --checkpoint output/runs/.../checkpoints/last.pt \
  --split test \
  --benchmark input/benchmarks/synthetic-test-v1.json
```

If the test membership, labels, eval transform or metric protocol changes, the
benchmark id changes. Create a new benchmark file; do not edit or overwrite the
old one. Old runs keep their original benchmark id, and `dlef compare` warns
when records come from different benchmarks.

The template config has `data.seed` (dataset content) separate from the
top-level `seed` (run seed), so changing the training seed does not move the
test set.

## Install

The core is a separate package. Install it first; the project can live anywhere
on disk.

```bash
pip install -e /path/to/DLEF   # core + adapters
pip install -e .               # this project
```

Special case: if you are running the template **in place** inside the DLEF repo
(`DLEF/templates/research-project`), the relative core path is `../..`:

```bash
pip install -e ../..
pip install -e .
```

For a copied project, always use the absolute path (or a wheel/git URL) for the
core.

## Run

```bash
python core/commands/train.py --config input/configs/base.yaml
python core/commands/train.py --config input/configs/base.yaml \
  --tag baseline --tag reported   # tags protect runs from gc
python core/commands/eval.py \
  --checkpoint output/runs/<project>/<experiment>/<run_id>/checkpoints/last.pt
python core/tools/runs.py                    # experiment summary
python core/tools/runs.py --experiment <name>
python core/tools/run_info.py --run <run_id>
python core/tools/configs.py list
python core/commands/sweep.py --config input/configs/base.yaml
make audit                                  # data / split / leakage first pass
make check                                  # initialization + config gate
python -m unittest discover -s core/tests -v
python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md
tensorboard --logdir output/runs --port 6006
```

Or `make train`, `make runs`, `make runs-all`, `make configs`, `make sweep`,
`make report`, `make gc-dry`, `make audit`, `make check`, `make test`,
`make doctor`.
`OUTPUT=<path>` overrides the output root for the tools (`make runs OUTPUT=...`).
`make eval CHECKPOINT=...` needs the checkpoint path explicitly;
`make inspect RUN=<run_id>` prints the run card. `make gc-dry` is a dry-run;
change `--metric-key` to your primary metric before using `--apply`.

## Command surface

Keep the command surface small:

```text
core/commands/   train.py / eval.py / sweep.py   # core experiment execution
core/tools/      benchmark / configs / runs / run_info  # generic project maintenance
scripts/    project-specific scripts        # intentionally unstructured
```

`scripts/` is the escape hatch for project-specific automation: organize it
however the current task needs. `core/tools/` is for the template's generic
maintenance utilities; don't turn it into a project script dump. If a script's
logic becomes reusable, move it to `core/` or DLEF core/adapters.
`core/tests/test_command_surface.py` enforces the small `core/commands/` surface.

## Config organization

One config file per **experiment variant**, not per run or seed:

```text
input/configs/
  base.yaml
  experiments/<experiment>/<variant>.yaml
  sweeps/<sweep>.yaml          # optional parameter space
```

See [`input/configs/README.md`](input/configs/README.md). Use `core/tools/configs.py list` to
validate the catalog and `core/tools/runs.py` to see which config produced each
run. Seeds and sweep trials are runs, not extra config files.

Training knobs are in two places:

```yaml
seed: 7
epochs: 10
max_steps: null
log_every: 10
checkpoint_every: 0

early_stopping:
  enabled: false  # set true to stop on plateau
  monitor: mae
  mode: min
  patience: 5
  min_delta: 0.0
  min_epochs: 0

# Workspace path: where runs/artifacts/sweeps are written.
# Relative paths resolve against the project root.
output_root: output

tensorboard:
  enabled: true
  subdir: tensorboard
  flush_secs: 10

data:
  batch_size: 32
```

The resolved run config is `output/runs/.../config.resolved.json`; the training
log header prints the same summary.

`output_root` is a workspace path, not an experiment variable. It defaults to
`output`; a relative value resolves against the project root. If you change it,
set `OUTPUT=<path>` for `make runs`/`make gc-dry` or export
`DLEF_OUTPUT_ROOT=<path>` for the tools' defaults.

## TensorBoard

When `tensorboard.enabled` is true, each run writes an event file under:

```text
output/runs/<project>/<experiment>/<run_id>/tensorboard/
```

Start TensorBoard:

```bash
tensorboard --logdir output/runs --port 6006
# open http://localhost:6006
```

Remote machine:

```bash
tensorboard --logdir output/runs --port 6006 --bind_all
# on your laptop:
ssh -L 6006:localhost:6006 user@host
```

The callback logs:

- `train/loss` and `train/<term>` per step;
- `val/<metric>` per epoch;
- `run/config` and `run/manifest` as text.

TensorBoard is a **mirror**, not the record: `manifest.json`,
`config.resolved.json`, `metrics.jsonl` and `logs/train.log` remain the source
of truth. Set `tensorboard.enabled: false` to disable it, or change `subdir` if
you do not want `tensorboard/` inside each run.

## Project brief, audit and check

`PROJECT.yaml` is the stable brief: question, unit, data contract, splits,
benchmark, primary metric, success criterion. Fill it before a real baseline.
If it changes, append a `DECISIONS.md` entry.

```bash
make audit    # data/split/leakage first pass
make check    # brief + config + benchmark + STATE/DECISIONS + tests
```

`make audit` is generic: it reports split sizes, numeric stats, NaNs/infs,
duplicate sample ids and group overlap when the batches expose the keys named
in `PROJECT.yaml`. If `sample_id` / `group_by` are missing it warns; add those
keys or audit the membership manifest manually.

`make check` is intentionally stricter than `make test`: it fails while
`PROJECT.yaml`, the config or the benchmark still contain template placeholders.
The template's tests pass before initialization; `make check` should not.

## Inspect a run

Start with the experiment-level index; it keeps a long run list readable:

```bash
make runs                                      # one row per experiment
python core/tools/runs.py --experiment <name>  # runs inside one experiment
python core/tools/runs.py --status completed --since 2026-10-01
python core/tools/runs.py --tag baseline
```

Then open one run; the run directory is the entry point and the config is the
first file to read:

```bash
python core/tools/run_info.py --run <run_id>
# or: make inspect RUN=<run_id>
```

The run card prints:

- `manifest.json`: status, code version, config digest, seed, data versions;
- `config.resolved.json`: data recipe, model/loss/optimizer/scheduler/metrics,
  benchmark config;
- eval records: benchmark id, test set id, metrics, test access;
- source files: `core/data/datasets/<name>.py`,
  `core/models/<name>.py`, `core/losses/<name>.py`, etc.

The config selects implementations; the `build.py` registries map names to
files. The manifest records the exact code version and data versions, so a
config name cannot silently drift to different source.

## Changing data / model / loss mid-experiment

A running job cannot swap data, architecture or loss. Start a new run; old runs
stay immutable. The template is config-driven:

```bash
python core/commands/train.py --config input/configs/base.yaml \
  --set data.name=regression \
  --set data.version=synthetic \
  --set data.split=holdout \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=data-linear-huber
```

`data.name` / `data.split` / `data.*_transform`, `model.name` and `loss.name`
select builders in `data/build.py`, `data/splits.py`, `models/build.py` and
`losses/build.py`. Add a new component file, register it in the corresponding
registry, extend the config validation, add a shape/golden test, and commit the
code.

For data, the key rule is: **new data or a new split is a new run, not an edit
to a previous run.** The loader reads a materialized membership and records the
resulting `DataSpec.version` in the run manifest, so old runs remain
reproducible even after the dataset grows.

For evaluation, just point at the checkpoint. `eval.py` reads the training
run's `config.resolved.json` via the checkpoint manifest, so the architecture
and loss match automatically; the eval run stays grouped under the training
experiment. `--config` is only a fallback if the original run directory is gone.

For a comparison, sweep `data.name`, `data.split`, `data.*_transform`,
`model.name`, `loss.name`, and `seed` in `core/commands/sweep.py`; one trial = one
new run. Commit code before official runs; dirty runs are exploratory.

## What you get

Each run writes:

```text
output/runs/<project>/<experiment>/<run_id>/
  manifest.json
  config.resolved.json
  metrics.jsonl
  logs/train.log
  checkpoints/last.pt
  evals/<eval_id>.json
```

Each sweep writes:

```text
output/sweeps/<sweep_id>/
  sweep.json
  configs/<trial_id>.json
  trials/<trial_id>.log
  trials/<trial_id>.result.json
```

`core/commands/train.py` writes a `TrialResult` when it runs under
`SubprocessSweep`; that is how the sweep gets `run_id` and metrics.

## Rules

- Core stays domain-agnostic; all project-specific words live here.
- One run = one resolved config × one seed.
- Eval is a separate run; every number links to a checkpoint.
- Use the sweep backend for variants; don't write nested loops in a single run.
- Keep failed runs; retention is reference-checked (`dlef gc`).

## DDP

`core/commands/train.py` supports `torchrun`:

```bash
torchrun --nproc_per_node=N core/commands/train.py --config input/configs/base.yaml
```

Rank 0 writes the run; other ranks train without writing. DDP evaluation and
distributed checkpoint resume are outside this template's v0 scope.
