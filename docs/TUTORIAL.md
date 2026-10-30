# 从零跑通一个 DLEF 项目（手把手）

这份教程只做一件事：**让你在自己的机器上，从复制模板到跑出第一个可追溯的实验结果**。

你不需要先读源码。

> **TL;DR（给 coding agent）**
>
> ```bash
> pip install -e /path/to/DLEF
> dlef init ~/code/my-project --project my-project
> cd ~/code/my-project
> pip install -e .
> make test
> ```
>
> 填 `PROJECT.yaml` / `STATE.md`，跑 `make audit`、`make check`；
> 然后让它读 `AGENTS.md` 第 0 节：先出计划，不要先写代码。

> **项目内版本**：复制完成后，项目根目录里会有一份 `TUTORIAL.md`
> （来自 `templates/research-project/TUTORIAL.md`），后续步骤可以直接看它。
> 这份 `docs/TUTORIAL.md` 是 DLEF 仓库的完整版，额外包含 examples / 框架改动 / FAQ。

---

## 0. 先选对入口

| 你想做什么 | 用哪里 |
|---|---|
| 开始一个新实验项目 | 复制 `templates/research-project/` |
| 学习框架怎么跑 | `examples/toy/` 或 `examples/torch_regression/` |
| 改框架本身 | `dlef/` + `adapters/`，并跑测试 |

**不要**把项目代码写进 `dlef/`。你的数据/模型/loss 都在项目目录里。

---

## 1. 准备 Python 环境

先确认有 torch：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果没有 torch，创建一个环境：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
```

> 你机器上已经有一个带 torch 2.7.1 的环境；demo 可以直接用它，正式项目建议自己建 venv。

---

## 2. 初始化项目

只复制 `templates/research-project/` 的**内容**，不要复制整个 DLEF 仓库。
`dlef/` 和 `adapters/` 是安装进来的依赖，不是项目文件。

推荐用 `dlef init`（复制 + `git init` + 写入 project 元数据到
`PROJECT.yaml` / `STATE.md` / `pyproject.toml` / `base.yaml`；研究字段仍要你填）：

```bash
pip install -e /path/to/DLEF
dlef init ~/dl-projects/demo --project demo
cd ~/dl-projects/demo
```

如果 core 较旧、或你想自己控制复制，使用 `git archive` 回退：

```bash
mkdir -p ~/dl-projects/demo
git -C /path/to/DLEF archive HEAD templates/research-project \
  | tar -x -C ~/dl-projects/demo --strip-components=2
cd ~/dl-projects/demo
git init
```

如果你改了模板工作区、想连未提交修改一起带走，用 rsync：

```bash
mkdir -p ~/dl-projects/demo
rsync -a \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'output/runs' --exclude 'output/artifacts' \
  --exclude 'output/sweeps' --exclude 'output/test_access' \
  /path/to/DLEF/templates/research-project/ ~/dl-projects/demo/
```

现在这个 `demo/` 就是你的项目。`core/` 是项目包名。项目根目录里还有
`PROJECT.yaml`、`TUTORIAL.md`、`STATE.md`、`DECISIONS.md`（都随模板复制）：
`PROJECT.yaml` 是稳定契约，`STATE.md` 是当前项目快照，
`DECISIONS.md` 是 append-only 决策日志。先填 `PROJECT.yaml` / `STATE.md`，
再开始真实实验。

> **不要复制**：`dlef/`、`adapters/`、`examples/`、DLEF 根目录的 `tests/`、
> `docs/`、`.git/`、`output/runs|artifacts|sweeps|test_access`。
> 数据字节放 `$DLEF_DATA_ROOT`，不进项目。

---

## 3. 安装 core 和项目

```bash
pip install -e /path/to/DLEF
pip install -e .
```

检查：

```bash
python -c "import dlef, torch; print('ok')"
```

`dlef` 是框架 core；`core` 是你的项目包。

填完 `PROJECT.yaml` / `STATE.md` 后，先跑初始化和数据门禁：

```bash
make audit    # 数据量、标签、NaN、重复 id、split/group overlap
make check    # brief + config + benchmark + STATE/DECISIONS + tests
```

`make check` 在 `PROJECT.yaml` / config / benchmark 还有占位符时会失败；
它不过，不要开始真实 baseline。

### 让 coding agent 接手

环境装好之后（模板 smoke 见下一节），把项目目录交给 agent，第一段话可以只有：

```text
读 AGENTS.md、PROJECT.yaml、STATE.md、DECISIONS.md，按 AGENTS.md 的 0. Start here 执行。
先给我计划，不要先写代码：
任务 / 数据 / split / benchmark / metric / 要改的文件 / smoke test。
我确认后再实现；先让 make audit / make check 过，结束时更新 STATE.md，
硬决策追加到 DECISIONS.md。
```

`AGENTS.md` 已经写了框架是什么、三层结构、文件放置、配置规则、
run/eval/claim 生命周期和禁止事项；`PROJECT.yaml` 是稳定契约，
`STATE.md` 是当前快照，`DECISIONS.md` 是不可改写的决策日志。
agent 不需要你再解释 DLEF。
如果它问“这个框架怎么用”，让它重新读 `AGENTS.md` 第 0 节。

---

## 4. 先跑 smoke test

```bash
python -m unittest discover -s core/tests -v
```

只跑 1 个测试：用很小的数据走一遍 train + eval。
**这一步通过，说明接线没问题。**

---

## 5. 跑第一次训练

```bash
python core/commands/train.py --config input/configs/base.yaml
```

你会看到类似：

```text
run: 20261029-..._s7_...
metrics: {'mae': 0.13...}
manifest: .../output/runs/my-project/baseline/.../manifest.json
checkpoint: .../checkpoints/last.pt
```

这一步做了：

1. 读 `input/configs/base.yaml`；
2. 建 data/model/loss/metrics；
3. 训练；
4. 在 val 上评测；
5. 写 manifest / config / metrics / checkpoint / eval record。

---

## 6. 看 run：先看 experiment 摘要，再看单个 run

`output/runs` 很快就会有很多 run；不要一上来 `find`。先看按 experiment
聚合的摘要：

```bash
make runs
python core/tools/runs.py                  # 默认 summary
python core/tools/runs.py --experiment baseline
python core/tools/runs.py --status completed --since 2026-10-01
python core/tools/runs.py --tag baseline
python core/tools/runs.py --json
```

`--experiment` 会自动切到 run 视图；`--all` 看所有 run；`--limit 0` 不限。
摘要里能看到每个 experiment 的 run 数、最新时间、状态、last/best metric、
tags 和 last config。

然后看一个 run：

```bash
python core/tools/runs.py --experiment <experiment>
python core/tools/run_info.py --run <run_id>
```

run 目录是：

```text
output/runs/<project>/<experiment>/<run_id>/
```

需要逐个文件看时再 `find` 这个目录，不要从 `output/runs` 根目录开始扫。

你会看到：

```text
manifest.json          # 身份证：代码版本、config 指纹、数据版本、seed、状态
config.resolved.json   # 这次实验完整的配置
metrics.jsonl          # 训练/验证过程（机器可读）
logs/train.log         # 训练文本日志（epoch/step/eval/checkpoint）
checkpoints/last.pt    # 模型
evals/xxx.json         # 评测成绩单
```

手动看：

```bash
cat "$RUN_DIR/config.resolved.json"
tail -5 "$RUN_DIR/metrics.jsonl"
```

更快的入口：

```bash
make runs                                      # experiment summary
make runs-all                                  # one row per run
python core/tools/runs.py --experiment baseline
python core/tools/runs.py --status completed --since 2026-10-01
python core/tools/runs.py --tag baseline
python core/tools/run_info.py --run <run_id>
python core/tools/configs.py list
```

`input/configs/` 按 variant 组织，不按 run/seed：

```text
input/configs/
  base.yaml
  experiments/<experiment>/<variant>.yaml
  sweeps/<sweep>.yaml
```

一个 seed 是一个 run，不是一份 config。`runs.py` 直接显示每个 run 的
`config_source`；`run_info.py` 再把 config 映射到源码文件。

### TensorBoard

`tensorboard.enabled: true` 时，每个 run 写：

```text
output/runs/<project>/<experiment>/<run_id>/tensorboard/
```

启动：

```bash
tensorboard --logdir output/runs --port 6006
# 远程：tensorboard --logdir output/runs --port 6006 --bind_all
# 本地：ssh -L 6006:localhost:6006 user@host
```

会看到 `train/loss`、`train/<term>`、`val/<metric>`，以及 `run/config` /
`run/manifest` 文本。TensorBoard 只是镜像；`manifest.json`、
`config.resolved.json`、`metrics.jsonl`、`logs/train.log` 才是记录。

建议养成习惯：**每跑完一次，先看 manifest 和 `config.resolved.json`，再看指标。**

---

## 7. 做一次独立评测

```bash
python core/commands/eval.py --checkpoint "$RUN_DIR/checkpoints/last.pt"
```

输出：

```text
config: training run ...
eval run: ...
metrics: {'mae': ...}
eval record: .../evals/...json
```

注意：**eval 不需要你重新传 config**。
它从 checkpoint 的 manifest 找到训练 run，读回训练时的
`config.resolved.json`，所以模型结构/loss 不会配错。

---

## 8. 跑一次 sweep（多个变体）

```bash
python core/commands/sweep.py --config input/configs/base.yaml
```

默认跑 4 个 trial（lr × grad accumulation），每个 trial 一个 subprocess。

你会看到类似：

```text
sweep: ... status=completed

| trial | optim.lr | accum | status | mae | run | log |
...
```

每个 trial 的日志和结果在：

```text
output/sweeps/<sweep_id>/trials/
```

一个 trial = 一个新 run；旧 run 不动。

---

## 9. 写一个 claim，生成报告

先拿一个 eval id：

```bash
EVAL_ID=$(python - <<'PY'
from pathlib import Path
import json
paths = sorted(Path("output/runs").rglob("evals/*.json"))
print(json.loads(paths[-1].read_text())["eval_id"])
PY
)
echo "$EVAL_ID"
```

写 claim：

```bash
python - <<PY
from dlef.claim import Claim, ClaimStore
ClaimStore("output/claims").add(Claim(
    claim_id="demo",
    text="Demo baseline MAE",
    eval_ids=("$EVAL_ID",),
    metric="mae",
))
PY
```

生成报告：

```bash
python -m dlef.cli report \
  --runs output/runs \
  --claims output/claims \
  --out output/reports/report.md
cat output/reports/report.md
```

报告里的每一行都能追到 eval → checkpoint → run。

---

## 10. 清理旧实验

先 dry-run：

```bash
python -m dlef.cli gc \
  --runs output/runs \
  --sweeps output/sweeps \
  --keep-last 2 \
  --keep-best 1 \
  --metric-key mae \
  --keep-tag reported \
  --keep-tag baseline \
  --sweep-ttl-days 30
```

它只会**打印计划**，不会删。启动训练时用 `--tag baseline` /
`--tag reported` 标记要保留的 run；`--keep-tag` 会保护它们。

确认后加：

```bash
--apply
```

`gc` 会保护：

- 最近 N 个 run；
- 最好的 N 个 run；
- 带 `reported` / `baseline` tag 的 run；
- 其他 run 的 parent；
- sweep 引用的 run；
- 被 kept run 引用的 artifact。

不要手动 `rm -rf output/runs`。

---

## 11. 换成你自己的数据/模型/loss/指标

改数据 + 组件包：

```text
core/data/          # 数据集 + split + transform + loader
  datasets/              # 每个数据集一个文件
  splits.py              # 物化的 role -> indices
  transforms/            # 每个 transform 一个文件
  paths.py               # $DLEF_DATA_ROOT + 外部目录布局
  spec.py                # DataBundle + DataSpec + recipe digest
  loaders.py             # DataLoader 组装
  benchmark.py           # 冻结 test set + metric protocol -> benchmark id
  build.py               # 数据侧 composition root
core/models/        # 每个模型一个文件 + models/build.py 注册
core/losses/        # 每个 loss 一个文件 + losses/build.py 注册
core/optimizers/    # 需要换优化器时
core/schedulers/    # 需要换调度器时
core/metrics/       # 每个指标一个文件 + metrics/build.py 注册
```

`task.py` 是 trainable task 的 composition root；`data/build.py` 是数据侧的
composition root；`core/commands/train.py` 把两者接到同一个 run。

### 外部数据根目录

实体数据（图片、音频、表格）不进 repo，放到项目外的目录：

```bash
export DLEF_DATA_ROOT=/path/to/datasets
```

标准布局：

```text
$DLEF_DATA_ROOT/
  raw/<source>/...
  curated/<dataset>/<version>/...
  splits/<dataset>/<version>/<split>.json
```

`core/data/paths.py` 负责解析这些路径；真实 dataset builder 用
`curated_dir(config.name, config.version)` 和
`split_manifest_path(config.name, config.version, split)`。项目里只放代码和
小 manifest。split 文件只存 membership（sample id 或相对路径），不复制
`train/` / `val/` / `test/` 三份图片。这样一份 curated 数据可以服务多个
split，也不会把项目目录撑乱。

`data.seed` 控制数据内容，顶层 `seed` 控制 run；改训练 seed 不应改变
test membership。

项目的 `.gitignore` 会忽略根目录下的 `/data/`、`/datasets/`、`/raw/`，
但不会忽略 `core/data/` 代码包。

### data/

一个数据集一个文件，返回 `DataBundle(dataset=..., spec=DataSpec(...))`：

```python
# data/datasets/my_dataset.py
def build_my_dataset(config, role, transform):
    ...
    return DataBundle(
        dataset=...,
        spec=DataSpec(
            name="my-dataset",
            version="sha256:...",
            schema={...},
            splits=...,
            transform=transform.name,
        ),
    )
```

在 `data/build.py` 注册：

```python
from .datasets.my_dataset import build_my_dataset

DATASETS = {
    "regression": build_regression,
    "my_dataset": build_my_dataset,
}
```

split 在 `data/splits.py` 物化，loader 只读 `role -> indices`，不在训练时
`random_split`：

```python
SPLITTERS = {"holdout": holdout, "my_split": my_split}
```

transform 一个文件，train/eval 分开选：

```python
# data/transforms/my_transform.py
def build_my_transform() -> Transform:
    return Transform(name="my_transform", fn=lambda sample: ...)
```

注册到 `TRANSFORMS`，然后在 YAML 里写 `data.train_transform` /
`data.eval_transform`。

对外保持这个接口：

```python
def make_loaders(config, seed, *, distributed=False):
    ...
    return train_loader, val_loader
```

每个 loader 是 **batch iterable**，batch 是 `Mapping[str, Any]`；每个 loader
带一个 `DataSpec`，其 `version` 会写进 run manifest。

### benchmark 固定测试集

benchmark = 测试集 membership + 标签版本 + eval transform + 指标实现/聚合/
阈值。把这份 manifest 冻结并提交：

```bash
python core/tools/benchmark.py create \
  --config input/configs/base.yaml \
  --out input/benchmarks/synthetic-test-v1.json

python core/tools/benchmark.py verify \
  --config input/configs/base.yaml \
  --benchmark input/benchmarks/synthetic-test-v1.json
```

test 评测必须带 benchmark：

```bash
python core/commands/eval.py \
  --checkpoint output/runs/.../checkpoints/last.pt \
  --split test \
  --benchmark input/benchmarks/synthetic-test-v1.json
```

- `test_set_id` 用于 test access 计数，改 eval transform 不会重置；
- `benchmark_id` 用于比较，改测试集/指标/eval transform 都会产生新 id；
- 旧 benchmark 不覆盖；要改 test 就新建 `v2`，报告里两个 benchmark 不合并排名。

### models/

一个模型一个文件：

```python
# models/my_model.py
def build_my_model(config, device="cpu") -> TorchModel:
    module = MyModel(...)
    return TorchModel(module, lambda module, batch: module(batch["x"]), device=device)
```

在 `models/build.py` 注册：

```python
from .my_model import build_my_model

BUILDERS = {
    "mlp": build_mlp,
    "linear": build_linear,
    "my_model": build_my_model,
}
```

模型只做 forward；不做 I/O、不读 config、不搬 device、不算 loss。

### losses/

一个 loss 一个文件，返回 `LossOutput(total, terms)`：

```python
# losses/my_loss.py
def build_my_loss(config) -> TorchLoss:
    def loss_fn(output, batch):
        total = ...
        return LossOutput(total=total, terms={"main": total})
    return TorchLoss(loss_fn)
```

然后在 `losses/build.py` 注册。

### metrics/

```python
# metrics/my_metric.py
def build_my_metric() -> Metric:
    return TorchMetric(lambda preds, targets: {"my_metric": ...})
```

在 `metrics/build.py` 注册，并把名字加进 YAML 的 `metrics` 列表。

### config.py + input/configs/base.yaml

把新组件名字加进对应 Config 的 `validate()`，再更新 YAML。
`data.name` / `data.version` / `data.split` / `data.*_transform` 和
`model.name` / `loss.name` 都是同一类选择器。unknown key 会报错；这是故意的。

早停配置在顶层：

```yaml
early_stopping:
  enabled: true
  monitor: mae
  mode: min
  patience: 5
  min_delta: 0.0
  min_epochs: 0
```

`monitor` 必须是 val metrics 里的名字；`patience` 是连续多少个 epoch 没有
改善后停止。日志会记录 best 值、bad epoch 计数和触发原因。

输出根目录和 TensorBoard 也在顶层：

```yaml
output_root: output

tensorboard:
  enabled: true
  subdir: tensorboard
  flush_secs: 10
```

`output_root` 是 workspace path（默认 `output`），相对路径相对项目根；
它不是实验变量。改了就 `make runs OUTPUT=<path>`，或 export
`DLEF_OUTPUT_ROOT=<path>`。`tensorboard` 只写镜像事件，不替代 RunStore。

### task.py

通常不用改。它只负责：

```python
model = build_model(config.model, device)
loss = build_loss(config.loss)

optimizer_fn = lambda model: build_optimizer(config.optim, model)
scheduler_fn = lambda opt: build_scheduler(config.scheduler, opt, epochs=config.epochs)

metrics = build_metrics(config.metrics)
```

---

## 12. 换数据/模型/loss 怎么办

**新数据集、新 split、新 transform、新模型、新 loss = 新 run，不改旧 run。**

模板已经配置驱动：

```bash
python core/commands/train.py --config input/configs/base.yaml \
  --set data.name=regression \
  --set data.split=holdout \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=data-linear-loss-huber
```

val 评测只给 checkpoint：

```bash
python core/commands/eval.py --checkpoint .../last.pt
```

它会自动读回训练配置。

最终 test 评测必须走冻结 benchmark：

```bash
python core/commands/eval.py \
  --checkpoint .../last.pt \
  --split test \
  --benchmark input/benchmarks/synthetic-test-v1.json
```

要比较多个组合，改 `core/commands/sweep.py`：

```python
params={
    "data.name": Choice(("regression",)),
    "data.split": Choice(("holdout",)),
    "model.name": Choice(("mlp", "linear")),
    "loss.name": Choice(("mse", "huber")),
    "seed": Choice((1, 2, 3)),
}
```

一个 trial = 一个新 run。旧 run 的 `DataSpec.version` 和 `config.resolved.json`
保持不变，所以新增数据不会让旧结果失效。

### 跑完一个 run 怎么查

入口是 run 目录，不是原始 YAML：

```text
output/runs/<project>/<experiment>/<run_id>/
  manifest.json            # config digest、config_source、code_version、data_versions
  config.resolved.json     # 权威配置：包含 --set / sweep 后的最终值
  evals/<eval_id>.json     # benchmark_id、test_set_id、metrics、test_access
  checkpoints/last.pt
```

```bash
python core/tools/run_info.py --run <run_id>
# 或 make inspect RUN=<run_id>
```

它会打印：

- run 状态、seed、代码版本、config digest、原始 config 路径；
- 数据：`data.name` / `version` / `seed` / `split` / transforms，以及 manifest
  里的 `data_versions`；
- 模型、loss、optimizer、scheduler、metrics、benchmark 的最终配置；
- 每条 eval 的 `benchmark_id` / `test_set_id` / metrics / test access；
- 配置名对应的源码文件，例如 `core/models/<name>.py`。

注意：**原始 `input/configs/*.yaml` 不是权威**。如果用了 `--set` 或 sweep，最终值在
`config.resolved.json`；源码由 `build.py` 注册表 + manifest 里的
`code_version` 决定。所以“config → 源码”的完整链路是：

```text
run dir
  → manifest.json (config_source, code_version, data_versions)
  → config.resolved.json (最终选择)
  → core/{models,losses,optimizers,schedulers,metrics,data}/build.py
  → <name>.py
```

---

## 13. 每天的工作流

```text
写/改 config
  → 跑 smoke
  → 跑 baseline
  → 看 manifest + config + metrics
  → val 上选模
  → 变体走 sweep
  → 最终 test 单独 eval
  → claim + report
  → 给要保留的 run 打 tag / 用 gc 清理
```

四条硬规则：

1. 一个 run = 一份 config × 一个 seed。
2. test 只用于 claim，不用于调参；test eval 必须走冻结 benchmark。
3. 旧 run 不改；新想法开新 run。
4. 旧 benchmark 不改；改测试集/指标/eval transform 就新建 benchmark，旧结果继续标旧 id。

---

## 14. 常用命令

```bash
# 训练
python core/commands/train.py --config input/configs/base.yaml \
  --tag baseline --tag reported

# 查看 experiment 摘要 / 单个 run 的 config / benchmark / 源码
make runs
make runs-all
python core/tools/runs.py --experiment baseline
python core/tools/run_info.py --run <run_id>

# 初始化门禁 / 数据审计
make check
make audit

# TensorBoard
tensorboard --logdir output/runs --port 6006

# 配置目录
python core/tools/configs.py list

# 换模型/loss
python core/commands/train.py --config input/configs/base.yaml \
  --set model.name=linear --set loss.name=huber

# val 评测
python core/commands/eval.py --checkpoint output/runs/.../checkpoints/last.pt

# 冻结 benchmark
python core/tools/benchmark.py create \
  --config input/configs/base.yaml \
  --out input/benchmarks/synthetic-test-v1.json

# 最终 test 评测（必须带 benchmark）
python core/commands/eval.py \
  --checkpoint output/runs/.../checkpoints/last.pt \
  --split test \
  --benchmark input/benchmarks/synthetic-test-v1.json

# 扫参
python core/commands/sweep.py --config input/configs/base.yaml

# 报告
python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md

# 清理（先 dry-run）
python -m dlef.cli gc --runs output/runs --sweeps output/sweeps \
  --keep-last 2 --keep-best 1 --metric-key mae

# 环境检查
python -m dlef.cli doctor

# DDP
torchrun --nproc_per_node=N core/commands/train.py --config input/configs/base.yaml
```

---

## 15. 常见问题

**torch 找不到**
用带 torch 的 Python/venv；`python -c "import torch"` 确认。

**unknown key**
配置写错或 schema 没加字段；看报错里的 key。

**eval 报结构不匹配**
eval 会自动读训练 run 的 config；如果原 run 目录被删，才用
`--config` 兜底，并确保 model/loss 与 checkpoint 一致。

**sweep 失败**
看 `output/sweeps/<sweep_id>/trials/<trial>.log` 和 `.result.json`。

**磁盘满了**
先 `dlef gc` dry-run，确认后再 `--apply`；不要手动 rm。

**test 怎么评**
模板 demo 只有 train/val；真实项目在 `data/splits.py` 里加 test membership，
然后用 `make_loader(config, "test", seed)` 取 test loader，在 `eval.py` 里用
`split="test"`。test 访问会被计数。

**examples 可以删吗**
`examples/` 不是 core，但测试会 import 它；删了 `make test` 会挂。
安装包不包含 examples；不用为了“安装包小”去删。

---

## 16. 不要做

- 不要改 `dlef/` 来放你的模型/loss。
- 不要手写 `output/runs/...` 或 `manifest.json`。
- 不要改已完成的 metrics/eval record。
- 不要在一个 run 里循环 seed。
- 不要用 test 调参。
- 不要手动删 run；用 `gc`。
- 不要用 dirty worktree 跑要进报告的结果；先 commit。

---

## 17. 一句话

> **复制模板 → 改数据 + 组件包 → 跑 train/eval/sweep → claim/report → gc。**

框架的职责是让每个数字有来源；你的职责是决定模型、数据和问题。
