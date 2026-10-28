# ADR 0001: Thin, domain-agnostic experiment OS

- **Status:** Accepted
- **Date:** 2026-10-27
- **Deciders:** 项目所有者

## Context

单人研究，需要一个覆盖广泛深度学习实验的通用框架。核心问题是：大量实验时，数据流、配置、指标、产物和血缘容易失控。

CV 只是第一个验证领域，不是框架的核心假设。RL、NLP、多模态、音频、图数据都应当能通过 adapter 接入，而不需要改 core。

从零构建完整训练框架，大部分工作会花在商品化基础设施上：AMP、DDP/FSDP、checkpoint/resume、日志、profiling。这些不是研究贡献。

## Decision

构建一个薄的、领域无关的 **experiment OS**：

- core 只定义通用契约：`DataSpec` / `Dataset` / `Batch`、`Model` / `ModelOutput`、`Loss` / `LossOutput`、`Metric` / `MetricResult`、`Runner`、`Config`、`RunManifest`、`RunStore`、`ArtifactStore`、`Launcher`。
- core 不包含任何领域词：没有 image / text / audio / video，没有 bbox / depth / mm，没有 classification / detection / segmentation，没有 RL / LLM。
- 领域实现放在 adapters；示例放在 examples；core 不 import adapters 或 examples。
- 训练引擎可以是现有手写循环、Lightning 或 Accelerate；`Runner` 只要求 step 是纯函数，不强制某个引擎。
- v1 用一个真实项目验证，但 core 的设计不依赖它。

## Consequences

**正面：**

- core 可跨 CV / NLP / RL / 多模态复用。
- 领域复杂度被隔离在 adapter 层。
- 本地优先、tracker 可选；Run Store 是事实来源。

**负面：**

- 需要纪律保持 core 干净。
- Adapter 增加一层薄薄的间接。
- 某些极端 workflow 需要 escape hatch。

## Alternatives considered

- **全自研训练框架**：拒绝。商品化工程成本高，且不产生研究价值。
- **平台中心（MLflow / Determined / W&B）**：拒绝。契约控制力弱，存在供应商锁定。
- **垂直领域模板**：拒绝。会把通用框架变成某个具体项目的迁移方案。
