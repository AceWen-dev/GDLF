# 从零跑通这个 DLEF 项目（项目内教程）

这份文件随 `templates/project/` 一起复制，目的是让你在**项目目录内**完成：

```text
装依赖 → 跑通模板 → 换成自己的数据/模型/loss/metric
→ train → eval val → 冻结 benchmark → eval test → sweep → claim → report
```

框架规则以 [`AGENTS.md`](AGENTS.md) 为准；这份教程只讲操作顺序。
如果你还没有项目目录，先看 DLEF 仓库的 `docs/TUTORIAL.md` 或 `README.md`
里的复制命令；复制时**只复制模板内容**，不要复制整个 DLEF 仓库。

---

## 0. 这是什么框架

DLEF 是一个薄薄的、领域无关的实验操作系统：

- `input/`：声明式输入——config、冻结的 benchmark；
- `core/`：你的项目代码——data / model / loss / optimizer / scheduler / metric；
- `output/`：生成的结果——runs、artifacts、sweeps、test_access、claims、reports。

DLEF core 以 `dlef` 包的形式安装，不复制进项目；`adapters/torch` 随 core 一起安装。
一个 run = 一个 resolved config × 一个 seed；manifest 在第一个 step 之前写；
completed run 不可修改；eval 是独立 run；RunStore 是唯一事实源。

---

## 1. 安装与验证

```bash
pip install -e /path/to/DLEF   # dlef + adapters
pip install -e .               # 本项目 core

export DLEF_DATA_ROOT=/path/to/data
python -c "import dlef, torch; print('ok')"
```

数据字节放在 `$DLEF_DATA_ROOT`，不要放进项目。标准布局：

```text
$DLEF_DATA_ROOT/
  raw/<source>/...
  curated/<dataset>/<version>/...
  splits/<dataset>/<version>/<split>.json
```

`core/data/paths.py` 负责解析这些路径；config 只写逻辑名字和版本，不写绝对路径。

---

## 2. 先跑通模板

不要先改代码。先确认模板能跑：

```bash
make test
python core/commands/train.py --config input/configs/base.yaml \
  --set epochs=1 --set experiment=smoke
```

成功时你会看到：

```text
output/runs/<project>/smoke/<run_id>/
  manifest.json
  config.resolved.json
  metrics.jsonl
  logs/train.log
  checkpoints/last.pt
  evals/...
```

模板占位是 synthetic 1-D regression；它只证明接线正确，不是你的实验。
给这个 run 打 `smoke` / `debug` 标签，以后用 `dlef gc` 清理。

---

## 3. 看 run：先看 experiment 摘要，再看单个 run

实验多了不要一上来列所有 run。先看按 experiment 聚合的摘要：

```bash
make runs
python core/tools/runs.py --limit 20
python core/tools/runs.py --experiment <name>
python core/tools/runs.py --status completed --since 2026-10-01
python core/tools/runs.py --tag baseline
python core/tools/runs.py --json
```

`--experiment` 会自动切到 run 视图；`--all` 看所有 run；`--limit 0` 不限。
摘要列：run 数、最新时间、状态、last/best metric、tags、last config。

然后看一个 run 的详情：

```bash
python core/tools/run_info.py --runs output/runs --run <run_id>
```

关键文件：

| 文件 | 作用 |
|---|---|
| `manifest.json` | 身份证：status、seed、config digest、config source、code version、data versions |
| `config.resolved.json` | 这次实验的完整配置；`--set`/sweep 后的最终值在这里 |
| `metrics.jsonl` | 机器可读的训练/验证指标 |
| `logs/train.log` | 人类可读日志：epoch/step、lr、loss terms、val、checkpoint、早停 |
| `checkpoints/last.pt` | 模型 + optimizer/scheduler/scaler/RNG/policy state |
| `evals/*.json` | eval 记录：split、checkpoint digest、benchmark id、metrics、test access |

`config.resolved.json` 是权威；原始 `input/configs/*.yaml` 在用了 `--set` 或
sweep 后不一定等于最终值。

---

## 4. 换成自己的数据/模型/loss/metric

保持 `core/task.py` 是组合根，只负责：

```python
model = build_model(config.model, device)
loss = build_loss(config.loss)
optimizer_fn = lambda model: build_optimizer(config.optim, model)
scheduler_fn = lambda opt: build_scheduler(config.scheduler, opt, epochs=config.epochs)
metrics = build_metrics(config.metrics)
```

按组件放文件：

| 你要加的东西 | 放哪里 | 还要改 |
|---|---|---|
| 数据集 | `core/data/datasets/<name>.py` | `core/data/build.py` 注册、`DataConfig.validate` |
| split | `core/data/splits.py` | `SPLITTERS`、`DataConfig.split` |
| transform | `core/data/transforms/<name>.py` | `core/data/build.py`、`DataConfig` |
| 模型 | `core/models/<name>.py` | `core/models/build.py`、`ModelConfig` |
| loss | `core/losses/<name>.py` | `core/losses/build.py`、`LossConfig` |
| optimizer / scheduler | 对应目录 | 对应 `build.py` 和 config |
| metric | `core/metrics/<name>.py` | `core/metrics/build.py`、`metrics` 列表 |

规则：

- 数据集 builder 返回 `DataBundle(dataset=..., spec=DataSpec(...))`；
- split 必须是 materialized membership（`role -> indices`），不要在
  `__getitem__` 或 loader 里 `random_split`；
- 每个 loss 返回 `LossOutput(total=..., terms={...})`，每个 term 都要命名；
- 每个 metric 加一个 golden test；
- 每个 dataset/transform 至少验证一个 batch 的 keys、dtypes、shapes；
- 数据/模型/loss 的实现不读 config、不做 I/O、不移动 device、不算 loss。

细节见 [`AGENTS.md`](AGENTS.md) 第 3、4、12 节。

---

## 5. 配置组织

```text
input/configs/
  base.yaml                         # 共享 baseline / smoke
  experiments/<experiment>/<variant>.yaml
  sweeps/<sweep>.yaml               # 可选参数空间
```

- 一个实验变体一个 config，不按 run、不按 seed 建 config；
- seed 是 run 的维度，3 个 seed = 3 个 run（sweep 参数里加 `seed`）；
- 不要为了每个实验改 `base.yaml`，复制成 variant；
- 正式 run 用 config 文件，不要靠一长串 `--set`；
- `--set` 只适合 smoke / 临时排查；
- 所有影响行为的选项都进 typed config，unknown key 报错；
- 不要在 config 里写输出路径，run 目录是派生的；
- 秘密走环境变量，不进 config/log。

查看/校验配置：

```bash
python core/tools/configs.py list
python core/tools/configs.py list --check
```

---

## 6. 训练 + val

```bash
python core/commands/train.py \
  --config input/configs/experiments/baseline/<variant>.yaml \
  --tag baseline

python core/commands/eval.py \
  --checkpoint output/runs/<project>/<experiment>/<run_id>/checkpoints/last.pt \
  --split val
```

`--tag` 会写进 manifest，`dlef gc --keep-tag` 会保护这些 run。常用 tag：
`baseline`、`reported`、`debug`。

val 用于模型选择。不要用 test 调参。

如果训练中断，先看 `logs/train.log` 的 traceback 和 `manifest.json` 的 status；
失败 run 也是记录，不要手工删。

---

## 7. 冻结 benchmark + test

test 只能通过冻结的 benchmark 访问。benchmark 的组成是：

```text
test membership + labels + eval transform + metric protocol
```

冻结一次并提交小 manifest：

```bash
python core/tools/benchmark.py create \
  --config input/configs/base.yaml \
  --out input/benchmarks/<name>.json

python core/tools/benchmark.py verify \
  --config input/configs/base.yaml \
  --benchmark input/benchmarks/<name>.json
```

最终 test：

```bash
python core/commands/eval.py \
  --checkpoint output/runs/.../checkpoints/last.pt \
  --split test \
  --benchmark input/benchmarks/<name>.json
```

规则：

- test membership、label version、eval transform、metric protocol 任一变化，
  就是新的 benchmark；不要改旧 manifest；
- 只在同一个 `benchmark_id` 内比较数字；
- test 访问次数按 `test_set_id` 计数，重复访问会 `WARN`；
- 不要在 test 上选模型或调参。

---

## 8. Sweep

一个 trial = 一个 run：

```bash
python core/commands/sweep.py --config input/configs/base.yaml
```

参数空间在 `core/commands/sweep.py` 里编辑（或按模板的 sweep spec 写）。
多 seed 示例：

```python
params={
    "optim.lr": Choice((0.02, 0.005)),
    "seed": Choice((1, 2, 3)),
}
```

生成的 trial config 在 `output/sweeps/<sweep_id>/configs/`，不写回
`input/configs/`。真实 GPU 工作用 subprocess sweep；轻量任务才用 in-process。

---

## 9. Claim + report

一个 claim 引用至少一个 eval id，并建议 pin 住 `benchmark_id`：

```python
from dlef.claim import Claim, ClaimStore

ClaimStore("output/claims").add(
    Claim(
        claim_id="main-result",
        text="Baseline MAE on frozen test set",
        eval_ids=("eval-id",),
        metric="mae",
        benchmark_id="sha256:...",
    )
)
```

生成报告：

```bash
make report
# 或：
python -m dlef.cli report \
  --runs output/runs \
  --claims output/claims \
  --out output/reports/report.md
```

不要手抄数字；让 claim/report 从 eval → checkpoint → run 解析。

---

## 10. 清理

先 dry-run：

```bash
make gc-dry
```

启动时用 `--tag baseline` / `--tag reported` 标记必须保留的 run；
`--keep-tag reported` 会保护它们。确认计划后再 apply：

```bash
python -m dlef.cli gc \
  --runs output/runs --sweeps output/sweeps \
  --keep-last 3 --keep-best 1 --metric-key <your-metric> \
  --keep-tag reported --keep-tag baseline \
  --artifact-ttl-days 30 --sweep-ttl-days 30 --apply
```

不要 `rm -rf output/runs/...`。`reported` / `baseline` tag 的 run 会被保护；
被 sweep 引用、被 claim 引用的 run 也会被保护。

---

## 11. 把项目交给 coding agent

环境装好之后，第一段话可以只有：

```text
读 AGENTS.md，按第 0 节执行。
先给我计划，不要先写代码：
任务 / 数据 / split / benchmark / metric / 要改的文件 / smoke test。
我确认后再实现。
```

`AGENTS.md` 已经写了框架是什么、三层结构、文件放置、配置规则、
run/eval/claim 生命周期和禁止事项；agent 不需要你再解释 DLEF。
如果它问“这个框架怎么用”，让它重新读 `AGENTS.md` 第 0 节。

---

## 12. 常见错误

- 把整个 DLEF 仓库复制进项目：不要；只复制模板内容，core 用 pip 安装。
- 改已安装的 `dlef/` 来适配项目：不要；项目代码放本项目 `core/`。
- 在 `Task.train_step` 里调用 `backward` / `optimizer.step`：不要；走
  `Runner` / `TrainingPolicy`。
- 手写 `output/runs/` 或 `manifest.json`：不要；RunStore 是唯一事实源。
- 手动删 run：不要；用 `dlef gc`。
- 把数据路径写进 config 或提交数据：不要；用 `$DLEF_DATA_ROOT`。
- 每个 seed 建一个 config：不要；seed 是 sweep 维度。
- 每个 smoke test 建一个新 experiment：不要；用一个 `debug`/`smoke`
  experiment，或给 run 打 `--tag debug`，否则 `output/runs` 很快会失控。
- 直接翻 `output/runs` 目录找实验：不要；用 `runs.py` 的 experiment 摘要，
  再 `--experiment` / `run_info.py` 钻取。
- 在 test 上调参，或看完 test 结果再改 benchmark：不要。
- 从 `train.log` 手抄数字：不要；用 eval + claim + report。
- 在项目根目录加 `utils.py`：不要；逻辑放 `core/`，一次性脚本放 `scripts/`。

---

## 13. 完成标准

1. config 通过校验，unknown key 报错；
2. `DataSpec` 存在，train/val/test membership 已 materialize，一个 batch 已验证；
3. model/loss/metric 实现契约，loss terms 有名字；
4. smoke test + benchmark identity test 通过；
5. train 产生 manifest + resolved config + metrics + checkpoint；
6. test eval 使用冻结 benchmark，eval 记录绑定 checkpoint digest + benchmark id；
7. sweep（如使用）一个 trial 一个 run，有 sweep 记录；
8. claim（如报告）能解析到 eval → checkpoint → run，并写清 benchmark；
9. `dlef gc` dry-run 只显示预期删除；
10. 没有把项目代码写进已安装的 core。
