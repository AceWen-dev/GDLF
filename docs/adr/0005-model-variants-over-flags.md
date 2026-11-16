# ADR 0005: Model structure varies by variant name, not by boolean flags

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** 项目所有者

## Context

模型结构在实验过程中持续变化。沿用的做法是在同一个模型文件里加开关：

```python
self.ordinal_head = bool(getattr(config, "ordinal_head", False))
self.metric_focal_v2 = bool(getattr(config, "metric_focal_v2", False))
...
if self.ordinal_head:
    ...
elif self.metric_anchor:
    ...
elif self.metric_focal_v2:
    ...
```

在真实项目上测量到的后果（56 个实验配置，一个 `model.name`）：

- 22 组不同的结构签名共用同一个名字；
- 13 个 bool/enum 字段在原理上描述 2¹³ 种组合，但 run 记录只能区分 1 个名字；
- `config_digest` 只描述**声明**的选择，看不见代码。模型文件被原地修改后，同一份 config 会在不同时间构建出不同的网络，而 digest 不变；
- 标定常数（`beta_ref`）、输入表示（米制深度编码）、冻结策略（`backbone`）和真正的拓扑开关混在同一个配置块里，其中 3 组"变体"的差别仅仅是 `beta_ref`。

另一条被认真考虑过的路线是组合式：`model.blocks: [{name, params}, ...]`，把结构写进 YAML。它被否掉，理由不是风格而是表达力：

- 互斥的输出路径在 `forward` 里是**提前返回 + 不同的 `aux` 键**（分档头返回 `ord_logits`，门控路径返回 `pred_beta`/`pred_direct`/`gate`），而 loss 依赖这些键。线性 block 列表无法表达"提前返回不同的输出契约"；
- 真正可组合的部分（零初始化旁路）**全部是加法**，顺序可交换，列表的顺序表达力为零；
- 异质 block 列表需要配置系统支持判别式联合，而当前 `dlef/config.py` 的 `_coerce` 对多候选 `Union` 直接原样返回、不做校验，等于放弃"未知键报错"和类型检查。

## Decision

1. **改拓扑 = 新变体名 = 新文件。** 增删模块、改连接、换输出头、换融合方式，一律新增 `core/models/<variant>.py` 并在 `models/build.py` 注册。共享部分抽成独立模块（如 `_trunk.py`），变体文件只写差异。
2. **改尺寸 = 同变体加字段。** width/depth/hidden 这类被 sweep 的旋钮可以留在同一变体，但新增字段的默认值必须复现旧行为；改变拓扑的默认值不允许。
3. **禁止用 bool 切拓扑。** builder 与 `forward` 内部不得按开关分支。N 个 bool 描述 2^N 个结构却只有 N 个名字，run 记录无法区分。
4. **配置分层。** 影响"是不是同一个模型"的字段写进 `ModelConfig.structural_signature()`；排除尺寸旋钮与标定常数。不属于模型的字段迁出：
   - 标定常数（如 `beta_ref`）→ 数据侧（dataset card / benchmark）；
   - 输入表示变换（如把 depth 换成 `log(mm/ref)`）→ `core/data/transforms/`；
   - 冻结/分层学习率 → optimizer / `TorchTrainingPolicy`，且不得两处同时控制同一件事。
5. **结构指纹是 run 记录的一部分。** `adapters/torch/structure.py` 在 `on_run_start` 把实际构建的模块哈希后写入 `<run_dir>/model_structure.json`。两个 digest 分工不同：
   - `digest`：层名 + 类型 + 形状 + 冻结状态，任何变化都移动；
   - `topology_digest`：只含层名 + 类型，对 width/depth sweep 稳定，标识架构族。
6. **变体数量必须可度量。** `core/tools/variants.py` 提供两个视图：
   - `configs`：按 `structural_signature()` 聚类，回答"配置树声明了几个变体"；
   - `runs`：按 run 目录里的指纹聚类，回答"实际跑了几个结构"，并检测**漂移**——同一 `config_digest` 对应多个 `structure_digest`，证明模型代码被原地修改。

## Consequences

**正面：**

- 打开 run 目录或 `run_info` 就能回答"这个 run 是什么结构"，不需要读模型源码或逐字段 diff 配置；
- 原地修改模型会以 DRIFT 形式暴露，避免把不可比的 run 放进同一张表；
- 重构可证伪：拆分前后 `structure_fingerprint` 相等即证明模型未被改变，且不需要训练；
- 变体爆炸变成一个数字，可以在 CI 里设阈值（`--check --max-variants`）；
- `topology_digest` 把"同一架构的不同尺寸"和"不同架构"区分开，避免把 sweep 误判成变体。

**负面：**

- 每个 run 多一个 `model_structure.json`（几十 KB 量级）；
- 项目必须手工维护 `structural_signature()`，漏写一个拓扑字段会让两个模型看起来像一个；
- 早于本次改动的旧 run 没有指纹，`variants runs` 会把它们计入 `runs_without_fingerprint`；
- 不提供自由组合能力；确实需要顺序敏感的组合时，必须另做设计（届时才考虑判别式联合）。

## Alternatives considered

- **自由 `blocks` 列表 + 判别式联合**：拒绝。对互斥输出路径表达力不足，对加法旁路顺序无意义，且需要框架层改动换取零收益。
- **只靠 `config_digest` 判断模型是否相同**：拒绝。它描述声明，不描述代码；原地修改完全不可见。
- **保留 flag，仅靠 DECISIONS.md 记录**：拒绝。文档纪律再好，run 记录里仍然只有一个名字，2^N 组合不可反查。
- **给每个 flag 组合自动生成名字**：拒绝。名字失去语义，且组合数爆炸。
