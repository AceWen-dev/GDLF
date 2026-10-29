# ADR 0002: Project template imports the versioned core

- **Status:** Accepted
- **Date:** 2026-10-27
- **Deciders:** 项目所有者

## Context

单人研究，多个项目并行。早期选择（Q7a）是把框架作为模板复制到每个项目。随着 core 长出 Config / Manifest / RunStore / Eval / Sweep / Retention / TrainingPolicy，复制 core 会让每个项目各自漂移，修 bug 需要手工 diff。

同时，新实验需要一个足够小的起点：配置、数据、模型、损失、指标、入口脚本，而不是再读一遍框架源码。

## Decision

`templates/project/` 是一个可复制的项目骨架，**依赖 `dlef` 包**，不 vendor core。

模板只包含项目特定内容：

- `myproject/config.py`、`data/`、`models/`、`losses/`、`optimizers/`、`schedulers/`、`metrics/`、`task.py`；
- `commands/train.py`、`eval.py`、`sweep.py`；
- `tools/`（模板自带的通用维护工具）和 `scripts/`（项目特定脚本，无固定结构）；
- `configs/base.yaml`、`benchmarks/`、`tests/test_smoke.py`、`tests/test_benchmark.py`。

安装时先安装 core（`pip install -e /path/to/DLEF`），再安装项目（`pip install -e .`）。真实部署时项目 pin 一个 core 版本。

## Consequences

**正面：**

- core 的修 bug / 新能力通过升级或 pin 版本传播，不靠手工 diff。
- 模板保持小而可读；新实验从拷贝模板开始。
- 项目仍可在自己的 adapter 层自由分叉。

**负面：**

- 需要安装/版本管理纪律；core 与项目的兼容性要明确。
- 多机器复现必须安装同一个 core 版本。

## Alternatives considered

- **纯复制模板（Q7a）**：拒绝。会 drift，修 bug 成本随项目数线性增长。
- **单 monorepo**：拒绝。会把不相关项目及其依赖耦合在一起。
- **把 core 放进模板**：拒绝。模板会变成第二份框架源码。
