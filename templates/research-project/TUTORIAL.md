# 从零跑通这个 DLEF 项目（项目内教程）

这份文件随 `templates/research-project/` 一起复制，目的是让你在**项目目录内**完成：

```text
装依赖 → 跑通模板 → 换成自己的数据/模型/loss/metric
→ train → eval val → 冻结 benchmark → eval test → sweep → claim → report
```

框架规则以 [`AGENTS.md`](AGENTS.md) 为准；这份教程只讲操作顺序。
项目契约在 [`PROJECT.yaml`](PROJECT.yaml)（稳定的 what），当前快照在
[`STATE.md`](STATE.md)，历史在 [`DECISIONS.md`](DECISIONS.md)
（append-only）；开工前读，收工前更新。
如果你还没有项目目录，推荐：

```bash
pip install -e /path/to/DLEF
dlef init ~/code/my-project --project my-project
```

`dlef init` 只复制模板内容、`git init`、写入 project 元数据到
`PROJECT.yaml` / `STATE.md` / `pyproject.toml` / `base.yaml`，不会替你填研究字段
或决定研究问题。不要复制整个 DLEF 仓库。

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

跑通后先填项目契约和记忆，再开始真实实验：

- `PROJECT.yaml`：问题/单位/输入输出/主指标/数据契约/split/benchmark；
- `STATE.md`：项目/问题/数据/split/benchmark/当前 baseline/下一步；
- `DECISIONS.md`：读模板条目 D-0001/D-0002，第一个项目决策从 D-0003 开始。

然后先跑数据和初始化门禁：

```bash
make audit    # 数据量、标签、NaN、重复 id、split/group overlap
make check    # brief + config + benchmark + STATE/DECISIONS + tests
```

`make check` 在 `PROJECT.yaml` / config / benchmark 还有占位符时会失败；
**它不过，不要开始真实 baseline。** `make audit` 缺 `sample_id` /
`group_by` 时会警告；真实项目应把这两个 key 放进 batch。

`make check` 还按 `STATE.md.phase` 做阶段门禁：

- `setup` / `data`：brief、config、数据根、benchmark 元数据；
- `baseline`：至少一个 completed 且带 `baseline` tag 的 run；
- `ablation`：至少一个 completed、非 baseline/debug 的 run；
- `benchmark`：冻结 manifest + `STATE.md` 里的 benchmark id；
- `report`：至少一个 claim + `output/reports/report.md`。

阶段没到就别改 `phase`。

---

## 3. 看 run：先看 status 和 experiment 摘要，再看单个 run

每次开工先看一屏状态：

```bash
make status
# 收工：make handoff
```

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
- `output_root` 是 **workspace path**（默认 `output`），控制
  runs/artifacts/sweeps 写到哪里；相对路径相对项目根。它不是实验变量，
  不要用它区分 run；
- `tensorboard` 是可选镜像；事件写 `<run_dir>/tensorboard/`，
  RunStore / `metrics.jsonl` / `logs/train.log` 仍是记录；
- 秘密走环境变量，不进 config/log。

```yaml
output_root: output

tensorboard:
  enabled: true
  subdir: tensorboard
  flush_secs: 10
```

如果改了 `output_root`，`make runs` / `make gc-dry` 用
`OUTPUT=<path>`，工具也可以 export `DLEF_OUTPUT_ROOT=<path>` 或传
`--runs <path>/runs`。

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

### TensorBoard 可视化

`tensorboard.enabled: true` 时，每个 run 会写：

```text
output/runs/<project>/<experiment>/<run_id>/tensorboard/
```

启动：

```bash
tensorboard --logdir output/runs --port 6006
# 浏览器打开 http://localhost:6006
```

远程机器：

```bash
tensorboard --logdir output/runs --port 6006 --bind_all
# 本地终端：
ssh -L 6006:localhost:6006 user@host
```

会看到：

- `train/loss`、`train/<loss_term>`（每 step）；
- `val/<metric>`（每 epoch）；
- `run/config`、`run/manifest`（文本）。

TensorBoard 只是**镜像**：真正记录仍是 `manifest.json`、
`config.resolved.json`、`metrics.jsonl`、`logs/train.log`。不想用就设
`tensorboard.enabled: false`；想换目录就改 `tensorboard.subdir`。

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

先 dry-run（如果改了 `output_root`，加 `OUTPUT=<path>`）：

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
读 AGENTS.md、PROJECT.yaml、STATE.md、DECISIONS.md，按 AGENTS.md 第 0 节执行。
先跑 make status，再给我计划，不要先写代码：
任务 / 数据 / split / benchmark / metric / 要改的文件 / smoke test。
我确认后再实现；先让 make audit / make check 过，结束时更新 STATE.md，
硬决策追加到 DECISIONS.md。
```

`AGENTS.md` 已经写了框架是什么、三层结构、文件放置、配置规则、
run/eval/claim 生命周期和禁止事项；`PROJECT.yaml` 是稳定契约，
`STATE.md` 是当前状态，`DECISIONS.md` 是不可改写的决策日志。
agent 不需要你再解释 DLEF。
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
- 跳过 `make audit` 直接 baseline：不要；先看数据量/标签/NaN/重复/overlap。
- `PROJECT.yaml` 和 `base.yaml` / benchmark 不一致：不要；`make check` 会挡住。
- 在 test 上调参，或看完 test 结果再改 benchmark：不要。
- 从 `train.log` 手抄数字：不要；用 eval + claim + report。
- 把 `STATE.md` 当日志、一直追加：不要；它是快照，结束时替换过期值。
- 改旧的 `DECISIONS.md` 条目：不要；追加新条目并标 `superseded by`。
- 在 `STATE.md` / `DECISIONS.md` 写绝对数据路径或秘密：不要。
- 在项目根目录加 `utils.py`：不要；逻辑放 `core/`，一次性脚本放 `scripts/`。

---

## 13. 完成标准

1. `PROJECT.yaml` 填完；`make check` 通过；
2. config 通过校验，unknown key 报错；
3. `DataSpec` 存在，train/val/test membership 已 materialize，一个 batch 已验证；
4. `make audit` 已看：样本量、标签、NaN、重复 id、split/group overlap；
5. model/loss/metric 实现契约，loss terms 有名字；
6. smoke test + benchmark identity test 通过；
7. train 产生 manifest + resolved config + metrics + checkpoint；
8. test eval 使用冻结 benchmark，eval 记录绑定 checkpoint digest + benchmark id；
9. sweep（如使用）一个 trial 一个 run，有 sweep 记录；
10. claim（如报告）能解析到 eval → checkpoint → run，并写清 benchmark；
11. `dlef gc` dry-run 只显示预期删除；
12. 没有把项目代码写进已安装的 core；
13. `STATE.md` 已更新，`DECISIONS.md` 有硬决策条目；
14. 项目记忆里没有秘密或绝对数据路径。
