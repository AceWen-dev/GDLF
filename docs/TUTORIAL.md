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

建议养成习惯：**每跑完一次，先看 manifest 和 config，再看指标。**

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

只改这四个文件：

```text
myproject/data.py
myproject/model.py
myproject/loss.py
myproject/metrics.py
```

### data.py

保持这个接口：

```python
def make_loaders(config, seed, *, distributed=False):
    ...
    return train_loader, val_loader
```

每个 loader 是 **batch iterable**，batch 是 `Mapping[str, Any]`。
给每个数据集一个 `DataSpec`（name / version / schema / splits / transform）。

### model.py

```python
def build_model(config, device="cpu") -> TorchModel:
    module = MyModel(...)
    return TorchModel(module, lambda module, batch: module(batch["x"]), device=device)
```

模型只做 forward；不做 I/O、不读 config、不搬 device、不算 loss。

### loss.py

```python
def build_loss(config) -> TorchLoss:
    def loss_fn(output, batch):
        total = ...
        return LossOutput(total=total, terms={"main": total})
    return TorchLoss(loss_fn)
```

每个 term 都命名；reduction 要明确。

### metrics.py

```python
def build_metrics(config):
    return {
        "mae": TorchMetric(lambda preds, targets: {"mae": (preds - targets).abs().mean()}),
    }
```

### config.py + configs/base.yaml

把新字段加进 dataclass，再更新 YAML。
unknown key 会报错；这是故意的。

### task.py

通常不用改。它只负责把 model/loss/optimizer/scheduler/metrics 接起来。

---

## 12. 换模型/换 loss 怎么办

**新模型/loss = 新 run，不改旧 run。**

模板已经配置驱动：

```bash
python commands/train.py --config configs/base.yaml \
  --set model.name=linear \
  --set loss.name=huber \
  --set experiment=arch-linear-loss-huber
```

评测只给 checkpoint：

```bash
python commands/eval.py --checkpoint .../last.pt
```

它会自动读回训练配置。

要比较多个组合，改 `commands/sweep.py`：

```python
params={
    "model.name": Choice(("mlp", "linear")),
    "loss.name": Choice(("mse", "huber")),
    "seed": Choice((1, 2, 3)),
}
```

一个 trial = 一个新 run。

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

三条硬规则：

1. 一个 run = 一份 config × 一个 seed。
2. test 只用于 claim，不用于调参。
3. 旧 run 不改；新想法开新 run。

---

## 14. 常用命令

```bash
# 训练
python commands/train.py --config configs/base.yaml

# 换模型/loss
python commands/train.py --config configs/base.yaml \
  --set model.name=linear --set loss.name=huber

# 评测
python commands/eval.py --checkpoint outputs/runs/.../checkpoints/last.pt

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
模板 demo 只有 train/val；真实项目在 `data.py` 里加 test loader，
在 `eval.py` 里用 `split="test"`。test 访问会被计数。

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

> **复制模板 → 改四个文件 → 跑 train/eval/sweep → claim/report → gc。**

框架的职责是让每个数字有来源；你的职责是决定模型、数据和问题。
