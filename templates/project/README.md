# Project template

A copyable project skeleton for DLEF. It imports the `dlef` core; it does **not**
vendor or copy the core.

**Agent guide:** [`AGENTS.md`](AGENTS.md) — read this before editing, or before
asking a coding agent to edit.

## What is here

```text
configs/base.yaml        # experiment config
myproject/config.py      # typed config schema
myproject/data/          # REPLACE/EXTEND: datasets, splits, transforms, loaders
  build.py               # data.name / transform registry + make_loaders
  datasets/              # one file per dataset
  transforms/            # one file per transform
  splits.py              # split strategies (materialized indices)
  paths.py               # $DLEF_DATA_ROOT + external dataset layout
  loaders.py             # DataLoader assembly
  spec.py                # DataSpec + recipe digest
myproject/models/        # REPLACE/EXTEND: one file per architecture + build.py
myproject/losses/        # REPLACE/EXTEND: one file per loss + build.py
myproject/optimizers/    # one file per optimizer + build.py
myproject/schedulers/    # one file per scheduler + build.py
myproject/metrics/       # one file per metric + build.py
myproject/task.py        # composition root: build model/loss/optimizer/scheduler/metrics
myproject/runtime.py     # policy / runner / sweep result helper
commands/train.py        # thin entry: train + eval
commands/eval.py         # thin entry: separate eval run
commands/sweep.py        # subprocess grid sweep
tests/test_smoke.py      # one tiny end-to-end test
tests/test_components.py # component shape/golden tests
claims/                  # one JSON per reported claim
AGENTS.md                # project-local agent guide
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
task. `commands/train.py` wires both into a run, so data, model, loss and
metrics are all selectable from the same config.

To add a dataset:

```python
# myproject/data/datasets/my_dataset.py
def build_my_dataset(config, role, seed, transform):
    ...
    return DataBundle(dataset=..., spec=DataSpec(...))

# myproject/data/build.py
from .datasets.my_dataset import build_my_dataset
DATASETS = {"regression": build_regression, "my_dataset": build_my_dataset}
```

Then allow `"my_dataset"` in `DataConfig.validate` and set
`data.name: my_dataset`. Splits are materialized in `data/splits.py`; the
runtime loader never calls `random_split`.

To add a model:

```python
# myproject/models/resnet.py
def build_resnet(config, device):
    return TorchModel(ResNet(config.depth), lambda m, b: m(b["x"]), device=device)

# myproject/models/build.py
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

`myproject/data/paths.py` resolves these paths. A real dataset builder uses
`curated_dir(config.name, config.version)` and
`split_manifest_path(config.name, config.version, split)`. The project keeps
code and small manifests; the images/tables themselves stay outside git. A
split manifest stores membership (sample ids or relative paths), not a copied
`train/` `val/` `test/` tree — one curated copy serves every split.

The project `.gitignore` blocks `/data/`, `/datasets/` and `/raw/` at the
project root, so accidental copies do not get committed. `myproject/data/`
(code) is not ignored.

## Install

The core is a separate package. Install it first; the project can live anywhere
on disk.

```bash
pip install -e /path/to/DLEF   # core + adapters
pip install -e .               # this project
```

Special case: if you are running the template **in place** inside the DLEF repo
(`DLEF/templates/project`), the relative core path is `../..`:

```bash
pip install -e ../..
pip install -e .
```

For a copied project, always use the absolute path (or a wheel/git URL) for the
core.

## Run

```bash
python commands/train.py --config configs/base.yaml
python commands/eval.py \
  --checkpoint outputs/runs/<project>/<experiment>/<run_id>/checkpoints/last.pt
python commands/sweep.py --config configs/base.yaml
python -m unittest discover -s tests -v
python -m dlef.cli report --runs outputs/runs --claims claims --out report.md
```

Or `make train`, `make sweep`, `make report`, `make gc-dry`, `make test`,
`make doctor`. `make eval CHECKPOINT=...` needs the checkpoint path explicitly.
`make gc-dry` is a dry-run; change `--metric-key` to your primary metric before
using `--apply`.

## Changing data / model / loss mid-experiment

A running job cannot swap data, architecture or loss. Start a new run; old runs
stay immutable. The template is config-driven:

```bash
python commands/train.py --config configs/base.yaml \
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
`model.name`, `loss.name`, and `seed` in `commands/sweep.py`; one trial = one
new run. Commit code before official runs; dirty runs are exploratory.

## What you get

Each run writes:

```text
outputs/runs/<project>/<experiment>/<run_id>/
  manifest.json
  config.resolved.json
  metrics.jsonl
  checkpoints/last.pt
  evals/<eval_id>.json
```

Each sweep writes:

```text
outputs/sweeps/<sweep_id>/
  sweep.json
  configs/<trial_id>.json
  trials/<trial_id>.log
  trials/<trial_id>.result.json
```

`commands/train.py` writes a `TrialResult` when it runs under
`SubprocessSweep`; that is how the sweep gets `run_id` and metrics.

## Rules

- Core stays domain-agnostic; all project-specific words live here.
- One run = one resolved config × one seed.
- Eval is a separate run; every number links to a checkpoint.
- Use the sweep backend for variants; don't write nested loops in a single run.
- Keep failed runs; retention is reference-checked (`dlef gc`).

## DDP

`commands/train.py` supports `torchrun`:

```bash
torchrun --nproc_per_node=N commands/train.py --config configs/base.yaml
```

Rank 0 writes the run; other ranks train without writing. DDP evaluation and
distributed checkpoint resume are outside this template's v0 scope.
