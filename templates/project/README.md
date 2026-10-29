# Project template

A copyable project skeleton for DLEF. It imports the `dlef` core; it does **not**
vendor or copy the core.

**Agent guide:** [`AGENTS.md`](AGENTS.md) — read this before editing, or before
asking a coding agent to edit.

## What is here

```text
configs/base.yaml        # experiment config
myproject/config.py      # typed config schema
myproject/data.py        # REPLACE: dataset + loaders
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
| `models/build.py` | `model.name` | `models/<name>.py` |
| `losses/build.py` | `loss.name` | `losses/<name>.py` |
| `optimizers/build.py` | `optim.name` | `optimizers/<name>.py` |
| `schedulers/build.py` | `scheduler.name` | `schedulers/<name>.py` |
| `metrics/build.py` | `metrics[]` | `metrics/<name>.py` |

`task.py` only composes; it never contains model/loss code.

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

## Changing model / loss mid-experiment

A running job cannot swap architecture/loss. Start a new run; old runs stay
immutable. The template is config-driven:

```bash
python commands/train.py --config configs/base.yaml \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=arch-linear-loss-huber
```

`model.name` and `loss.name` select the builders in `models/build.py` /
`losses/build.py`; `task.py` stays unchanged. Add a new component file, register
it in the corresponding `build.py`, extend the config validation, add a shape/
golden test, and commit the code.

For evaluation, just point at the checkpoint. `eval.py` reads the training
run's `config.resolved.json` via the checkpoint manifest, so the architecture
and loss match automatically; the eval run stays grouped under the training
experiment. `--config` is only a fallback if the original run directory is gone.

For a comparison, sweep `model.name`, `loss.name`, and `seed` in
`commands/sweep.py`; one trial = one new run. Commit code before official runs;
dirty runs are exploratory.

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
