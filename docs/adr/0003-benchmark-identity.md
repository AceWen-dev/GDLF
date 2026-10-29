# ADR 0003: Benchmark identity is separate from run identity

- **Status:** Accepted
- **Date:** 2026-10-29
- **Deciders:** 项目所有者

## Context

测试集经常被重新划分、补数据或换指标。结果是每个实验的测试集不同，分数无法比较；同时，测试样本可能通过重复、同组样本、时间顺序或预处理统计泄漏进训练。

只记录 `split="test"` 和 `data_version` 不够：

- `data_version` 可能随 eval transform 或训练数据变化，导致 test access 计数被重置；
- 同一个测试集配不同指标实现，数字含义不同；
- 没有 benchmark id，报告无法判断两行是否可比。

## Decision

把 benchmark 定义为一个不可变对象：

```text
benchmark = test membership + label version
          + eval transform
          + metric implementation/version + aggregation + thresholds
          + evaluator version
```

拆成两个 id：

- `test_set_id`：dataset + split membership。用于 test access 计数和泄漏审计；改标签、transform 或指标不重置计数。
- `benchmark_id`：`test_set_id` + label version + eval transform + metric protocol。用于可比性；任一项变化都产生新 benchmark。

core（`dlef/benchmark.py`）只拥有 id 契约和 `BenchmarkSpec`。数据层负责：

- 物化 test membership，构建 benchmark manifest；
- 查重、group/time/label 泄漏检查；
- 将测试数据实体放在 `DLEF_DATA_ROOT` 之外。

`EvalRecord` / `EvalSpec` 增加 `benchmark_id` / `test_set_id`；`TestAccessLog` 按 `test_set_id` 计数；`Claim` 记录 `benchmark_id`；`compare` / `report` 对跨 benchmark 的结果标记为不可直接比较。测试集或指标变化时创建新 benchmark，不覆盖旧 benchmark；旧 run/claim 保留旧 id。

## Consequences

**正面：**

- 只改训练数据、模型、loss、超参时，benchmark 不变，结果可比；
- 改测试集/标签/eval transform/指标时，新 benchmark 明确出现，旧结果不被悄悄重解释；
- test access 不会因为换 transform 被重置；
- run manifest、eval record、claim 都能反查到 benchmark。

**负面：**

- 项目必须维护 benchmark manifest 和 membership；多一个需要提交的小文件；
- 跨 benchmark 的数字不能放在同一张排名表里；
- 重复/近似重复/group/time/label 检查仍要项目自己实现，core 不提供算法。

## Alternatives considered

- **只用 `split="test"` + `data_version`**：拒绝。无法区分测试集变化和指标变化，也无法防止 access 计数重置。
- **把 benchmark 构建放进 core**：拒绝。membership、查重和数据格式是领域/项目相关；core 只保留 id 和审计字段。
- **直接覆盖 test set**：拒绝。旧 run 无法复现，历史 claim 失效。
