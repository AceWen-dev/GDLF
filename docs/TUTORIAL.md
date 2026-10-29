# 从零跑通一个 DLEF 项目（手把手）

这份教程只做一件事：**让你在自己的机器上，从复制模板到跑出第一个可追溯的实验结果**。

你不需要先读源码。

---

## 0. 先选对入口

| 你想做什么 | 用哪里 |
|---|---|
| 开始一个新实验项目 | 复制 `templates/project/` |
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

## 2. 复制模板

```bash
cd /path/to/DLEF
mkdir -p ~/dl-projects
cp -r templates/project ~/dl-projects/demo
cd ~/dl-projects/demo
git init
```

现在这个 `demo/` 就是你的项目。`myproject/` 是占位包名；demo 先不用改。

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

`dlef` 是框架 core；`myproject` 是你的项目包。

---

## 4. 先跑 smoke test

```bash
python -m unittest discover -s tests -v
```

只跑 1 个测试：用很小的数据走一遍 train + eval。
**这一步通过，说明接线没问题。**

---

## 5. 跑第一次训练

```bash
python commands/train.py --config configs/base.yaml
```

你会看到类似：

```text
run: 20261029-..._s7_...
metrics: {'mae': 0.13...}
manifest: .../outputs/runs/my-project/baseline/.../manifest.json
checkpoint: .../checkpoints/last.pt
```

这一步做了：

1. 读 `configs/base.yaml`；
2. 建 data/model/loss/metrics；
3. 训练；
4. 在 val 上评测；
5. 写 manifest / config / metrics / checkpoint / eval record。

---

## 6. 看一次 run 到底有什么

```bash
RUN_DIR=$(find outputs/runs -name manifest.json | head -1 | xargs dirname)
echo "$RUN_DIR"
find "$RUN_DIR" -maxdepth 2 -type f | sort
```

你会看到：

```text
manifest.json          # 身份证：代码版本、config 指纹、数据版本、seed、状态
config.resolved.json   # 这次实验完整的配置
metrics.jsonl          # 训练/验证过程
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
python scripts/analysis/run_info.py --run <run_id>
python scripts/analysis/runs.py --experiment baseline
python scripts/ops/configs.py list
```

`configs/` 按 variant 组织，不按 run/seed：

```text
configs/
  base.yaml
  experiments/<experiment>/<variant>.yaml
  sweeps/<sweep>.yaml
```

一个 seed 是一个 run，不是一份 config。`runs.py` 直接显示每个 run 的
`config_source`；`run_info.py` 再把 config 映射到源码文件。

建议养成习惯：**每跑完一次，先看 manifest 和 `config.resolved.json`，再看指标。**

---

## 7. 做一次独立评测

```bash
python commands/eval.py --checkpoint "$RUN_DIR/checkpoints/last.pt"
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
python commands/sweep.py --config configs/base.yaml
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
outputs/sweeps/<sweep_id>/trials/
```

一个 trial = 一个新 run；旧 run 不动。

---

## 9. 写一个 claim，生成报告

先拿一个 eval id：

```bash
EVAL_ID=$(python - <<'PY'
from pathlib import Path
import json
paths = sorted(Path("outputs/runs").rglob("evals/*.json"))
print(json.loads(paths[-1].read_text())["eval_id"])
PY
)
echo "$EVAL_ID"
```

写 claim：

```bash
python - <<PY
from dlef.claim import Claim, ClaimStore
ClaimStore("claims").add(Claim(
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
  --runs outputs/runs \
  --claims claims \
  --out report.md
cat report.md
```

报告里的每一行都能追到 eval → checkpoint → run。

---

## 10. 清理旧实验

先 dry-run：

```bash
python -m dlef.cli gc \
  --runs outputs/runs \
  --sweeps outputs/sweeps \
  --keep-last 2 \
  --keep-best 1 \
  --metric-key mae
```

它只会**打印计划**，不会删。

确认后加：

```bash
--apply
```

`gc` 会保护：

- 最近 N 个 run；
- 最好的 N 个 run；
- 其他 run 的 parent；
- sweep 引用的 run；
- 被 kept run 引用的 artifact。

不要手动 `rm -rf outputs/runs`。

---

## 11. 换成你自己的数据/模型/loss/指标

改数据 + 组件包：

```text
myproject/data/          # 数据集 + split + transform + loader
  datasets/              # 每个数据集一个文件
  splits.py              # 物化的 role -> indices
  transforms/            # 每个 transform 一个文件
  paths.py               # $DLEF_DATA_ROOT + 外部目录布局
  spec.py                # DataBundle + DataSpec + recipe digest
  loaders.py             # DataLoader 组装
  benchmark.py           # 冻结 test set + metric protocol -> benchmark id
  build.py               # 数据侧 composition root
myproject/models/        # 每个模型一个文件 + models/build.py 注册
myproject/losses/        # 每个 loss 一个文件 + losses/build.py 注册
myproject/optimizers/    # 需要换优化器时
myproject/schedulers/    # 需要换调度器时
myproject/metrics/       # 每个指标一个文件 + metrics/build.py 注册
```

`task.py` 是 trainable task 的 composition root；`data/build.py` 是数据侧的
composition root；`commands/train.py` 把两者接到同一个 run。

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

`myproject/data/paths.py` 负责解析这些路径；真实 dataset builder 用
`curated_dir(config.name, config.version)` 和
`split_manifest_path(config.name, config.version, split)`。项目里只放代码和
小 manifest。split 文件只存 membership（sample id 或相对路径），不复制
`train/` / `val/` / `test/` 三份图片。这样一份 curated 数据可以服务多个
split，也不会把项目目录撑乱。

`data.seed` 控制数据内容，顶层 `seed` 控制 run；改训练 seed 不应改变
test membership。

项目的 `.gitignore` 会忽略根目录下的 `/data/`、`/datasets/`、`/raw/`，
但不会忽略 `myproject/data/` 代码包。

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
python scripts/ops/benchmark.py create \
  --config configs/base.yaml \
  --out benchmarks/synthetic-test-v1.json

python scripts/ops/benchmark.py verify \
  --config configs/base.yaml \
  --benchmark benchmarks/synthetic-test-v1.json
```

test 评测必须带 benchmark：

```bash
python commands/eval.py \
  --checkpoint outputs/runs/.../checkpoints/last.pt \
  --split test \
  --benchmark benchmarks/synthetic-test-v1.json
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

### config.py + configs/base.yaml

把新组件名字加进对应 Config 的 `validate()`，再更新 YAML。
`data.name` / `data.version` / `data.split` / `data.*_transform` 和
`model.name` / `loss.name` 都是同一类选择器。unknown key 会报错；这是故意的。

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
python commands/train.py --config configs/base.yaml \
  --set data.name=regression \
  --set data.split=holdout \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=data-linear-loss-huber
```

val 评测只给 checkpoint：

```bash
python commands/eval.py --checkpoint .../last.pt
```

它会自动读回训练配置。

最终 test 评测必须走冻结 benchmark：

```bash
python commands/eval.py \
  --checkpoint .../last.pt \
  --split test \
  --benchmark benchmarks/synthetic-test-v1.json
```

要比较多个组合，改 `commands/sweep.py`：

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
outputs/runs/<project>/<experiment>/<run_id>/
  manifest.json            # config digest、config_source、code_version、data_versions
  config.resolved.json     # 权威配置：包含 --set / sweep 后的最终值
  evals/<eval_id>.json     # benchmark_id、test_set_id、metrics、test_access
  checkpoints/last.pt
```

```bash
python scripts/analysis/run_info.py --run <run_id>
# 或 make inspect RUN=<run_id>
```

它会打印：

- run 状态、seed、代码版本、config digest、原始 config 路径；
- 数据：`data.name` / `version` / `seed` / `split` / transforms，以及 manifest
  里的 `data_versions`；
- 模型、loss、optimizer、scheduler、metrics、benchmark 的最终配置；
- 每条 eval 的 `benchmark_id` / `test_set_id` / metrics / test access；
- 配置名对应的源码文件，例如 `myproject/models/<name>.py`。

注意：**原始 `configs/*.yaml` 不是权威**。如果用了 `--set` 或 sweep，最终值在
`config.resolved.json`；源码由 `build.py` 注册表 + manifest 里的
`code_version` 决定。所以“config → 源码”的完整链路是：

```text
run dir
  → manifest.json (config_source, code_version, data_versions)
  → config.resolved.json (最终选择)
  → myproject/{models,losses,optimizers,schedulers,metrics,data}/build.py
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
python commands/train.py --config configs/base.yaml

# 查看 run 的 config / benchmark / 源码
python scripts/analysis/run_info.py --run <run_id>
python scripts/analysis/runs.py

# 配置目录
python scripts/ops/configs.py list

# 换模型/loss
python commands/train.py --config configs/base.yaml \
  --set model.name=linear --set loss.name=huber

# val 评测
python commands/eval.py --checkpoint outputs/runs/.../checkpoints/last.pt

# 冻结 benchmark
python scripts/ops/benchmark.py create \
  --config configs/base.yaml \
  --out benchmarks/synthetic-test-v1.json

# 最终 test 评测（必须带 benchmark）
python commands/eval.py \
  --checkpoint outputs/runs/.../checkpoints/last.pt \
  --split test \
  --benchmark benchmarks/synthetic-test-v1.json

# 扫参
python commands/sweep.py --config configs/base.yaml

# 报告
python -m dlef.cli report --runs outputs/runs --claims claims --out report.md

# 清理（先 dry-run）
python -m dlef.cli gc --runs outputs/runs --sweeps outputs/sweeps \
  --keep-last 2 --keep-best 1 --metric-key mae

# 环境检查
python -m dlef.cli doctor

# DDP
torchrun --nproc_per_node=N commands/train.py --config configs/base.yaml
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
看 `outputs/sweeps/<sweep_id>/trials/<trial>.log` 和 `.result.json`。

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
- 不要手写 `outputs/runs/...` 或 `manifest.json`。
- 不要改已完成的 metrics/eval record。
- 不要在一个 run 里循环 seed。
- 不要用 test 调参。
- 不要手动删 run；用 `gc`。
- 不要用 dirty worktree 跑要进报告的结果；先 commit。

---

## 17. 一句话

> **复制模板 → 改数据 + 组件包 → 跑 train/eval/sweep → claim/report → gc。**

框架的职责是让每个数字有来源；你的职责是决定模型、数据和问题。
