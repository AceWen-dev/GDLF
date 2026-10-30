# DLEF 最终报告

> **一句话**：DLEF 是一个领域无关的深度学习实验框架。它不替你写模型，而是保证每次实验的**数据、配置、指标、产物、结论**都能对上号。

---

## 0. 先看结论

- 核心 `dlef/` **零运行时依赖**，不含任何领域词（没有 image / text / bbox / mm / classification / RL）。
- 8 个实验环节都有明确契约；PyTorch adapter 是可选的参考实现。
- 每次 run 都有 **manifest**：代码版本、config hash、数据版本、环境、seed、状态、产物。
- **eval 是独立 run**；每个数字能追到 checkpoint，再追到 run。
- 支持 **grid / random sweep**，有 thread 和 subprocess 两种后端；一个 trial = 一个 run。
- 支持 **AMP、梯度累积、梯度裁剪、可选 DDP**。
- 支持 **retention / gc**：默认 dry-run，引用检查、血缘保护、artifact TTL。
- 支持 **Claim / Markdown report**：报告里的数字可以反向解析到 eval record；test split 按 `test_set_id` 计数，重复访问标 `WARN`；跨 benchmark 标为不可比。
- 项目模板 `templates/research-project/` 可以直接复制，**依赖 core 包，不复制 core**。
- 测试：主套件 **61 个**；模板 smoke **23 个**（torch 环境全过）。
- 真实跑通：CUDA AMP、2 进程 CPU DDP、subprocess sweep、模板 train/eval/sweep。

---

## 1. 这个框架解决什么问题

做大量深度学习实验时，最常见的混乱不是“模型写不出来”，而是：

1. 看到一个数字，**不知道它从哪来**：哪份代码、哪份 config、哪份数据、哪个 seed、哪台机器？
2. 单次 run 看起来很好，**换成 3 个 seed 就变差**；比较靠手抄表格。
3. 数据、checkpoint、日志、评测结果散在不同目录，**过一个星期就找不到**。
4. 每开一个新项目就复制一套代码，**越复制越漂移**。
5. 磁盘只增不减，**旧 run 和 checkpoint 不敢删**，因为不知道谁引用了谁。
6. 论文/报告里的数字是手抄的，**审稿人或未来的自己无法复核**。

DLEF 的目标不是做“更强的训练框架”，而是做**实验操作系统**：

- 训练循环、AMP、DDP、checkpoint 这些“商品化能力”交给现有引擎；
- 框架只拥有**契约、记录、血缘、评测、清理**；
- 让“数据流”和“数字流”都有唯一入口和唯一出口。

---

## 2. 核心模型：一条 spine

```text
Config ──► Run ──► Eval ──► Claim
             │        │
             └── Artifact
```

| 对象 | 含义 | 不变量 |
|---|---|---|
| **Config** | 一次 run 的完整输入声明 | 类型化、unknown key 报错、运行前 dump + hash |
| **Run** | 一份 resolved config × 一个 seed 的一次执行 | manifest 先写；完成后不可变；失败也保留 |
| **Eval** | 一个 checkpoint × 一份 eval spec × 一个 benchmark | 独立 run；有自己的 manifest；test access 计数；benchmark id 可比性 |
| **Claim** | 报告/论文里的一个数字、表或图 | 必须引用 eval；report 反向解析到 run |
| **Artifact** | checkpoint、预测、指标表、图 | 内容寻址；被 kept run 引用则保护 |

一句话规则：

> **没有 run 就没有数字；没有 manifest 就没有 run；没有 eval 就没有 claim。**

---

## 3. 8 个环节怎么落地

你最早给的 8 个环节，现在都有对应的抽象和代码：

| # | 环节 | 核心对象 | 工程约束 | 代码 |
|---|---|---|---|---|
| 1 | 数据 | `DataSpec` + `Dataset` + `Batch` | 只读；split/transform 版本化；batch 是 typed mapping | `dlef/contracts.py` |
| 2 | 模型 | `Model` + `ModelOutput` | 不做 I/O、不读 config、不碰 device、不含 loss | `dlef/contracts.py` |
| 3 | 损失 | `Loss` + `LossOutput` | `total` + 命名 `terms`；reduction/normalization 显式 | `dlef/contracts.py` |
| 4 | 训练 | `Runner` + `TrainingPolicy` | loop 归 Runner；AMP/accum/clip/resume 可插拔 | `dlef/runner.py`、`dlef/policy.py` |
| 5 | 配置 | `Config` + dataclass schema | 类型化；unknown key 报错；resolved + digest | `dlef/config.py` |
| 6 | 指标/评测 | `Metric` + `EvalSpec` + `EvalRecord` | eval 独立 run；split 命名空间；test access 计数 | `dlef/eval.py`、`dlef/claim.py` |
| 7 | 运行/输出 | `RunManifest` + `RunStore` + `ArtifactStore` | one run = config × seed；artifact 内容寻址；tracker 只是索引 | `dlef/run.py` |
| 8 | 脚本 | `CLI` + `Launcher` + 项目模板 | 薄入口；一条命令一个工作流；preflight + smoke | `dlef/cli.py`、`templates/research-project/` |

补上的两个横切层：

- **TrainingPolicy**：AMP / 梯度累积 / 梯度裁剪 / DDP 的接缝，core 不 import torch。
- **Claim / TestAccess**：把“报告里的数字”接回 eval，并记录 test split 访问次数。

---

## 4. 架构

```text
core (dlef/)
  contracts.py    # Data / Model / Loss / Optimizer / Scheduler / Callback / Metric / Task
  config.py       # dataclass schema + strict loader + digest
  runner.py       # train/eval loop
  policy.py       # TrainingPolicy: AMP / accum / clip / DDP seam
  checkpoint.py   # CheckpointCodec: JSON default
  run.py          # manifest / run store / artifact store
  eval.py         # EvalSpec / EvalRecord
  benchmark.py    # test set + metric protocol -> benchmark id
  claim.py        # Claim / ClaimStore / TestAccessLog
  sweep.py        # SweepSpec / trials / thread + subprocess queues
  retention.py    # plan / apply / reference checks
  report.py       # comparison + claim report
  registry.py     # name / class_path -> implementation
  cli.py          # doctor / compare / gc / report

adapters/torch/   # optional PyTorch adapter
  components.py   # DataLoader / Model / Loss / Optimizer / Scheduler / Metric / Task
  policy.py       # AMP / accumulation / clipping / DDP
  checkpoint.py   # .pt checkpoint codec

examples/toy/               # stdlib linear regression
examples/torch_regression/  # CUDA toy + subprocess sweep
templates/research-project/          # copyable project skeleton (imports core, no vendoring)
```

依赖方向：

```text
core ← adapters ← examples / templates
```

硬规则：

> **core 里不出现领域词。** 领域代码只能出现在 adapter 或项目模板里。

---

## 5. 已实现功能

### 5.1 配置

- dataclass schema，支持嵌套（`data / model / optim / benchmark / train`）。
- **每个影响行为的选项都是 config 字段**；builder 接收 config，不在代码里隐藏开关。
- unknown key 报错；类型不匹配报错；`validate()` 做范围/互斥检查。
- 运行前 dump `config.resolved.json`，SHA-256 digest 进 manifest。
- manifest 记录 `config_source` 和 `code_version`；`run_info.py` 从 config 反查源码。
- 支持 `early_stopping`：`monitor` / `mode` / `patience` / `min_delta` /
  `min_epochs`；默认关闭，触发原因写进训练日志。
- `output_root` 控制 workspace 输出路径（默认 `output`，相对项目根）；
  `tensorboard` 是可选镜像（`enabled` / `subdir` / `flush_secs`）。
- JSON 永远可用；YAML 在安装 PyYAML 后可用。

### 5.2 Run / Artifact

- `RunManifest`：run_id、project、experiment、config_digest、config_source、code_version、data_versions、env、seed、status、artifacts、parent_run_id、tags、error。
- `RunStore`：`runs/<project>/<experiment>/<run_id>/`。
- `ArtifactStore`：内容寻址 `<sha[:2]>/<sha>/<name>`。
- checkpoint 默认 JSON；torch adapter 用 `.pt`（`torch.save`）。
- run 目录同时有 `metrics.jsonl`（机器可读）和 `logs/train.log`（epoch/step/eval/checkpoint 文本日志）；
  `tensorboard.enabled=true` 时还有 `tensorboard/` 事件文件（可选镜像）。
- manifest 在第一个 step 之前写；失败/中断也保留。

### 5.3 Eval / benchmark

- `EvalSpec`：checkpoint、split、metrics、aggregation、ci_method、test_access_policy、benchmark_id、test_set_id。
- `EvalRecord`：eval_id、split、checkpoint_digest、data_version、eval_spec_digest、metrics、predictions、test_access、benchmark_id、test_set_id。
- `BenchmarkSpec`：test set（dataset + version + membership + label version）+ eval transform + metric versions + aggregation + thresholds + evaluator version。
- `test_set_id` 用于防泄漏和 test access；`benchmark_id` 用于可比性；只有同一个 `benchmark_id` 的 eval 才能直接比较。
- eval 是独立 run；`compare` 从 eval records 生成对比表，跨 benchmark 会提示不可比。
- test split 每次 eval 都按 `test_set_id` 写入 `TestAccessLog`；report 里访问次数 > 1 标 `WARN`。

### 5.4 Sweep

- `SweepSpec`：grid / random；`Choice / Uniform / LogUniform / IntUniform`。
- `LocalSweep` + `LocalQueue`：thread-based，in-process trial。
- `SubprocessSweep` + `SubprocessQueue`：一 trial 一进程，GPU 池分配，日志和 result file，timeout，失败隔离。
- 子进程 env：`CUDA_VISIBLE_DEVICES / DLEF_SWEEP_ID / DLEF_TRIAL_ID / DLEF_TRIAL_INDEX / DLEF_GPU_IDS / DLEF_TRIAL_RESULT`。
- 一个 trial = 一个 run；失败 trial 记 `failed`，sweep 变 `partial`。

### 5.5 Retention / gc

- 保护规则：per-experiment last N、best N、every-N、tags、status、age、parent lineage、sweep references。
- artifact：被 kept run 引用，或 age < TTL，才保留；否则删除。
- 默认 dry-run；`--apply` 才删。
- 计划里每个 action 都有 reason 和 size。

### 5.6 Training policy

- `NoOpTrainingPolicy`：默认单进程行为。
- `TorchTrainingPolicy`：
  - AMP（`autocast` + `GradScaler`）；
  - gradient accumulation（loss 缩放 + 边界 step + epoch 末 flush）；
  - gradient clipping（unscale 后）；
  - 可选 DDP（rank 0 写 run/log/checkpoint）。
- checkpoint 里保存 policy state（例如 scaler）。

### 5.7 Claim / report

- `Claim`：claim_id、text、eval_ids、metric、value、unit、notes、tags、benchmark_id。
- `ClaimStore`：一个 claim 一个 JSON。
- `dlef report`：claim + eval records → Markdown。
- 缺失 eval 会显示 `MISSING`；test access > 1 显示 `WARN`；claim 与 eval 的 benchmark 不一致显示 `MISMATCH`。

### 5.8 CLI

```bash
python -m dlef.cli doctor
python -m dlef.cli compare <eval-json-or-run-dir> ...
python -m dlef.cli gc --runs runs --keep-last 3 --keep-best 1 --metric-key mae
python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md
```

---

## 6. 怎么用

### 6.1 从模板开始

```bash
pip install -e /path/to/DLEF
dlef init ~/my-experiment --project my-experiment
cd ~/my-experiment
pip install -e .
export DLEF_DATA_ROOT=/path/to/data
make test
```

模板按三层组织：`input/`（配置 + benchmark）、`core/`（项目源码）、`output/`
（run / claim / report）。项目根目录还有 `AGENTS.md`、`PROJECT.yaml`、
`STATE.md`、`DECISIONS.md`、`TUTORIAL.md`：`PROJECT.yaml` 是稳定契约，
`STATE.md` 是当前快照，`DECISIONS.md` 是 append-only 决策日志。
填完 brief 后先跑 `make audit` + `make check`；`make check` 在
`PROJECT.yaml` / config / benchmark 还有占位符时会失败，这是初始化门禁。

配置按 **variant** 组织，不按 run/seed：

```text
input/configs/
  base.yaml
  experiments/<experiment>/<variant>.yaml
  sweeps/<sweep>.yaml
```

一个 seed 是一个 run，不是一份 config。`core/tools/configs.py list` 检查配置目录，
`core/tools/runs.py` 默认按 experiment 汇总（run 数、最新、状态、last/best
metric、tags、config source），`--experiment` 看单个实验、`--tag` 过滤标签；
细节看 `core/tools/run_info.py --run <id>`。

模板里需要替换/扩展的是数据 + 组件包：

```text
core/data/          # 数据集 + split + transform + loader
  datasets/              # 每个数据集一个文件
  splits.py              # 物化的 role -> indices
  transforms/            # 每个 transform 一个文件
  paths.py               # $DLEF_DATA_ROOT + 外部目录布局
  spec.py                # DataSpec + recipe digest
  loaders.py             # DataLoader 组装
  benchmark.py           # 冻结 test set + metric protocol
  build.py               # 数据侧 composition root
core/models/        # 每个模型一个文件 + models/build.py 注册
core/losses/        # 每个 loss 一个文件 + losses/build.py 注册
core/optimizers/    # 需要换优化器时
core/schedulers/    # 需要换调度器时
core/metrics/       # 每个指标一个文件 + metrics/build.py 注册
```

数据也是消融轴：`data.name` / `data.split` / `data.train_transform` /
`data.eval_transform` 都可以进 sweep；一个新数据集、新 split 或新 transform
就是一个新 run。

实体数据（图片/表格等）不进项目目录，放到项目外的 `$DLEF_DATA_ROOT`：

```text
$DLEF_DATA_ROOT/
  raw/<source>/...
  curated/<dataset>/<version>/...
  splits/<dataset>/<version>/<split>.json
```

`core/data/paths.py` 负责解析；split 文件只存 membership，不复制
train/val/test 图片。项目里只保留代码和小 manifest。

`config.py / task.py / runtime.py / core/commands/` 是接线；`task.py` 是 composition
root，只负责把组件拼成 `TorchTask`。

### 6.2 训练 + 评测

```bash
python core/commands/train.py --config input/configs/base.yaml
```

输出：

```text
output/runs/<project>/<experiment>/<run_id>/
  manifest.json
  config.resolved.json
  metrics.jsonl
  logs/train.log
  checkpoints/last.pt
  evals/<eval_id>.json
```

独立评测：

```bash
python core/commands/eval.py \
  --checkpoint output/runs/<project>/<experiment>/<run_id>/checkpoints/last.pt
```

查一个 run 的全部信息：

```bash
python core/tools/run_info.py --run <run_id>
```

它从 run 目录读取 `manifest.json` + `config.resolved.json` + eval records，打印
数据/模型/loss/指标/benchmark 以及对应源码文件。`config.resolved.json` 是权威；
原始 YAML 在用了 `--set` 或 sweep 后不是最终值。

### 6.3 实验中途换模型 / 损失函数

正在跑的任务不能热换架构或 loss；**换模型/loss = 新 run**，不要改旧 run。

模板是配置驱动的：

```bash
python core/commands/train.py --config input/configs/base.yaml \
  --set data.name=regression \
  --set data.version=synthetic \
  --set data.split=holdout \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=data-linear-loss-huber
```

- `data.name` / `data.version` / `data.split` / `data.*_transform` 选择
  `data/build.py` 和 `data/splits.py`；
- `model.name` 选择 `models/build.py` 里的 builder；
- `loss.name` 选择 `losses/build.py` 里的 builder；
- `optim.name` / `scheduler.name` / `metrics[]` 分别从各自 registry 选择；
- `task.py` 不变；
- `eval.py` 会通过 checkpoint manifest 找到训练 run 的
  `config.resolved.json`，自动复用同一份结构/超参，所以不需要手动重复
  `--set`；只有原 run 目录丢了才用 `--config` 兜底。
- 最终 test eval 必须带冻结 benchmark：
  `python core/commands/eval.py --checkpoint ... --split test --benchmark input/benchmarks/<id>.json`。

比较多个数据/架构/loss 时，用 sweep 把 `data.name`、`data.split`、
`data.*_transform`、`model.name`、`loss.name`、`seed` 作为参数；一个 trial =
一个新 run。旧 run 保留，`dlef compare` / `dlef report` 负责比较。
换代码后先 commit 再跑正式实验；dirty run 只能算探索。

### 6.4 扫参

```bash
python core/commands/sweep.py --config input/configs/base.yaml
```

每个 trial：

- 自动写一份 trial config；
- 一个 subprocess；
- 日志在 `output/sweeps/<sweep_id>/trials/<trial_id>.log`；
- result 在 `trials/<trial_id>.result.json`；
- `train.py` 会自动把 run_id / metrics 写回 result。

### 6.5 报告

```python
from dlef.claim import Claim, ClaimStore

ClaimStore("output/claims").add(
    Claim(
        claim_id="main-result",
        text="Our method improves the main metric",
        eval_ids=("eval-id-from-a-run",),
        metric="mae",
    )
)
```

```bash
python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md
```

### 6.6 清理

```bash
# 默认 dry-run
python -m dlef.cli gc \
  --runs output/runs --sweeps output/sweeps \
  --keep-last 3 --keep-best 1 --metric-key mae \
  --keep-tag reported --artifact-ttl-days 30

# 确认后
python -m dlef.cli gc ... --apply
```

### 6.7 不用 torch 的 toy

```bash
python examples/toy/run.py
python examples/toy/sweep.py
```

### 6.8 用 torch 的 toy

```bash
pip install -e .[torch]
python examples/torch_regression/run.py
python examples/torch_regression/sweep.py
```

### 6.9 DDP

```bash
torchrun --nproc_per_node=N core/commands/train.py --config input/configs/base.yaml
```

本地 CPU smoke：

```bash
CUDA_VISIBLE_DEVICES="" python -m torch.distributed.run \
  --standalone --nproc_per_node=2 examples/torch_regression/run.py
```

---

## 7. 验证结果

### 7.1 测试

```text
# 主套件，无 torch
Ran 48 tests ... OK (skipped=4)

# 主套件，torch 2.7.1
Ran 48 tests ... OK

# 项目模板
make template-smoke                  # base：1 skipped
make template-smoke PYTHON=...       # torch：1 passed
```

### 7.2 stdlib toy

```text
| eval_id | split | checkpoint | mae |
| ...-val-... | val | ... | 0.0824712 |
| ...-val-... | val | ... | 0.0802664 |
```

### 7.3 CUDA AMP + accumulation + clipping

```text
run: 20261027-151843_...
device: cuda
amp=True accum=2 clip=1.0 ddp=False
metrics: {'mae': 0.07590167969465256}
checkpoint: .../checkpoints/last.pt
```

### 7.4 2 进程 CPU DDP

```text
rank 1: training complete (non-main)
run: 20261027-151847_...
device: cpu
amp=False accum=2 clip=1.0 ddp=True
metrics: {'mae': 0.09482796490192413}
checkpoint: .../checkpoints/last.pt
```

### 7.5 模板 train / eval / sweep

训练：

```text
run: 20261027-152412_...
metrics: {'mae': 0.13193866610527039}
manifest: .../manifest.json
checkpoint: .../checkpoints/last.pt
```

评测：

```text
eval run: 20261027-152419_...
metrics: {'mae': 0.13193866610527039}
eval record: .../baseline-eval/.../evals/...json
```

Sweep：

```text
sweep: 20261027-152422_... status=completed

| trial | lr | accum | status | mae | run |
|---|---|---|---|---|---|
| t000 | 0.02  | 1 | completed | 0.0761794 | ... |
| t001 | 0.02  | 2 | completed | 0.0816824 | ... |
| t002 | 0.005 | 1 | completed | 0.0784605 | ... |
| t003 | 0.005 | 2 | completed | 0.376686  | ... |
```

### 7.6 Retention

```text
reclaimable: 186.8 KiB (6 runs, 12 artifacts)
dry-run: would delete 6 run(s), 12 artifact(s)
```

### 7.7 Claim report

```text
| claim | text | metric | value | split | benchmark | eval | run | test_access |
| c1 | Baseline comparison | mae | ... | test | 647b946221bf | ... | ... | WARN 2 |
```

---

## 8. 关键设计决策

### 8.1 做“实验操作系统”，不做“训练框架”

训练循环、AMP、DDP、checkpoint 是商品化能力。重造它们会花掉 6–12 个月，而且不是研究贡献。

DLEF 拥有的是：

- typed config；
- run manifest；
- artifact store；
- eval record；
- claim / report；
- sweep / retention / policy 这些接缝。

### 8.2 core 领域无关

core 不出现任何领域词。领域代码只在 adapter / template。

这不是“抽象洁癖”，而是为了让 CV / NLP / RL / 多模态都能用同一套记录和评测协议。项目审计只作为 case study，不进入 core 设计。

### 8.3 本地优先，tracker 只是索引

RunStore 是事实来源；W&B / MLflow 将来只是可选索引。项目模板已包含可选的
TensorBoard 镜像（`tensorboard.enabled`），但事件文件不替代 RunStore。

原因很现实：Neptune 在 2025-12 被 OpenAI 收购，SaaS 在 2026-03 关停。把实验记录绑在某个 SaaS 上是危险的。

### 8.4 Config 用 dataclass，不用 Hydra

- 类型化、unknown key 报错、IDE 支持好；
- resolved dump + hash 更适合审计；
- Hydra 成熟但低维护，且 YAML 里的 interpolation / `_target_` 容易把配置变成代码。

### 8.5 Eval 独立于训练

训练里算的 val 只用于选模；真正的评测是独立 run：

- 独立 manifest；
- checkpoint digest；
- data version + test set id；
- benchmark id + eval spec digest；
- test access 计数。

### 8.6 模板依赖 core，不复制 core

早期想过“纯模板复制”，但项目一多就会 drift。现在：

- `templates/research-project/` 只放项目代码；
- core 通过安装引入；
- 修 bug / 加能力通过升级 core 传播；
- 每个项目仍可在 adapter 层自由分叉。

ADR：`docs/adr/0002-template-imports-core.md`、`docs/adr/0003-benchmark-identity.md`。

### 8.7 失败也保留

失败的 run、失败的 trial 都保留：

- run 状态 `failed`；
- trial 状态 `failed`；
- sweep 状态 `partial`；
- retention 默认保护 running / 有 tag / 年轻 / 被引用的 run。

“只保留成功”会丢失最重要的信息。

---

## 9. 目录导览

```text
DLEF/
  README.md                 # 快速上手
  REPORT.md                 # 本报告
  GLOSSARY.md               # 领域词汇
  dlef/                     # core，零运行时依赖
  adapters/torch/           # 可选 PyTorch adapter
  examples/toy/             # stdlib 线性回归
  examples/torch_regression/# torch 玩具 + sweep
  templates/research-project/        # 项目模板（AGENTS / STATE / DECISIONS / TUTORIAL）
  tests/                    # 主测试套件
  docs/
    adr/                    # 架构决策
    design/contracts-v0.md  # 契约草案
```

### 阅读顺序建议

1. `docs/TUTORIAL.md` — 手把手从复制模板跑到 report。
2. `README.md` — 先跑起来。
3. 本报告 — 理解整体设计。
4. `AGENTS.md` — 建新项目时 agent 必须读的规则（文件放哪、实验怎么组织）。
5. `templates/research-project/AGENTS.md` — 项目复制后的本地规则。
6. `docs/design/contracts-v0.md` — 看 18 个 contract。
7. `templates/research-project/` — 看一个真实项目怎么接。
8. `dlef/runner.py` / `dlef/policy.py` — 看训练循环和 AMP/DDP。
9. `dlef/sweep.py` / `dlef/retention.py` — 看大量实验和清理。

---

## 10. 已知边界

这些不是 bug，是 v0 明确没做的：

- **HPO**：只有 grid / random；没有 Optuna / Ray Tune adapter。
- **Tracker**：没有 W&B / MLflow adapter；模板有可选 TensorBoard 镜像，
  RunStore 仍是唯一的真相。
- **远程执行**：只有本地 thread / subprocess；没有 Slurm / K8s launcher。
- **DDP 完整性**：训练已跑通；多 rank eval、分布式 checkpoint resume 没做。
- **Claim 强制**：test access > 1、跨 benchmark 只标 `WARN`，不阻塞；没有 claim/report UI。
- **Benchmark 自动检查**：模板会校验 benchmark 的 id；duplicate/group/time/label 泄漏检查仍需项目自己实现。
- **kill-resume 测试**：checkpoint round-trip 有测试；完整 kill-resume 还没自动化。
- **模板占位符**：数据/模型/loss/metric 是合成回归，需要替换。
- **数据审计**：没有内建 leakage/duplicate 检查；DataSpec 只提供字段。
- **多 seed**：支持多个 run；模板 sweep 只给了一个 lr/accum 示例，数据/模型/loss/seed 都可加进 `SweepSpec.params`。
- **模型 zoo / 训练引擎**：不提供；这是 adapter 的事。

---

## 11. 开始一个新实验的 checklist

1. 复制 `templates/research-project/`，`git init`。
2. 安装 core 和项目依赖。
3. 写 `input/configs/base.yaml`：project / experiment / seed / epochs / 数据 / 模型 / 优化器。
4. 设置 `DLEF_DATA_ROOT`，替换 `data/` 下的数据集/transform/split，并按需扩展 `models/`、`losses/`、`metrics/`、`optimizers/`、`schedulers/`。
5. 冻结 benchmark 并提交 manifest：
   ```bash
   python core/tools/benchmark.py create --config input/configs/base.yaml --out input/benchmarks/<name>.json
   ```
6. 跑 smoke test：
   ```bash
   python -m unittest discover -s core/tests -v
   ```
7. 跑 baseline；确认 manifest / config / checkpoint 都对。
8. 需要多 seed 时，把 `seed` 加进 sweep 的 params；一个 seed 一个 run。
9. val 上选模；test 只用冻结 benchmark 评一次。
10. 写 claim（带 benchmark_id）；`dlef report` 生成报告。
11. 给要保留的 run 打 `reported` tag；用 `dlef gc` 清理其余。
12. 归档：把 manifest、config、benchmark、报告、关键 checkpoint 放到长期存储。

---

## 12. 术语小词典

| 词 | 意思 |
|---|---|
| **Run** | 一份 config × 一个 seed 的一次执行 |
| **Manifest** | Run 的身份证：代码、config、数据、环境、seed、状态、产物 |
| **Config** | 一次 run 的完整输入声明；resolved + hash |
| **Artifact** | checkpoint、预测、指标表、图；内容寻址 |
| **Eval** | 独立评测 run：checkpoint × eval spec × benchmark |
| **Benchmark** | 测试集 + 标签 + eval transform + 指标协议；可比较性的单位 |
| **Claim** | 报告里的一个数字/表/图，链接到 eval |
| **Sweep** | 一组 trial；一个 trial = 一个 run |
| **Trial** | sweep 里的一个 run |
| **Policy** | 训练接缝：AMP / accum / clip / DDP |
| **Adapter** | 领域实现：数据、模型、loss、metric、tracker、launcher |
| **Core** | 领域无关的契约 + 服务 |

---

## 13. 结语

这套东西的目标不是让你多写代码，而是让你**少解释**：

- 看到一个数字，能自己追到 eval、checkpoint、run、config、数据、代码；
- 换一个项目，不需要重新发明记录方式；
- 清理磁盘时，知道哪些能删、哪些被引用；
- 论文/报告里的每个数字，都能反向复核。

如果以后继续做，优先级建议是：

1. **Optuna / Ray Tune adapter**（HPO）；
2. **W&B / MLflow / TensorBoard adapter**（可选索引）；
3. **Slurm / K8s launcher**（远程执行）；
4. **多 rank eval / 分布式 checkpoint resume**；
5. **Claim 强制策略和报告 UI**。

但即使停在这里，`templates/research-project/` + `dlef` core + torch adapter 已经可以支撑一个真实项目的完整实验闭环。
