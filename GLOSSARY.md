# GLOSSARY

> 本文件只定义项目语言，不记录实现细节、工具选型或待定问题。

## Adapter
某个契约（数据、模型、损失、指标、Tracker、Launcher）的具体实现。

## Artifact
由 Run 产出或消费的不可变对象：Checkpoint、预测、指标表、图。Artifact 以内容寻址，并携带产生它的 Run。

## Checkpoint
一种 Artifact：包含模型参数、优化器状态、调度器状态、随机数状态和恢复位置，足以让 Run 从中断处继续。

## Claim
报告或论文中的一个数字、表格或图。Claim 必须引用一个或多个 Evaluation；没有 Evaluation 的 Claim 不成立。

## Config
一次 Run 的完整输入声明。Config 是类型化的、经过校验的、可完全解析的；YAML、CLI、环境变量只是它的输入格式。

## Data Recipe
一次 Run 固定的数据组合：dataset version + split + transform。任一项变化都是新 recipe、新 Run；旧 Run 不被覆盖。

## Dataset Version
经过清洗、标签修正、去重后的不可变数据内容。实体数据存放在项目外，项目只保留代码和小型 manifest；版本之间只追加，不原地改写。

## Split
同一份 Dataset Version 上的 membership 视图（sample id 或相对路径），不是数据副本。Split 必须物化并被记录，训练时不得重新随机划分。

## Eval Spec
定义一次 Evaluation 的输入：checkpoint 选择规则、数据 split、预处理、指标及其聚合/置信区间方法。

## Evaluation (Eval)
一种 Run：把一个 Checkpoint、一份 Eval Spec 和一个数据 split 绑定，产出 Metrics Artifact。Evaluation 与训练分离，拥有自己的 Manifest。

## Experiment
围绕一个问题的一次有边界的探究。Experiment 有一份预先写下的 Protocol，以及一组 Runs。它是科学组织单位，不是代码目录。

## Launcher
把 Run 放到本地、集群或云上执行的接缝。Launcher 不改变 Run 的语义，只改变执行位置。

## Manifest
Run 的不可变身份记录：代码版本、解析后 Config 的哈希、数据版本、环境、Seed、主机和状态。没有 Manifest 的执行不算 Run。

## Project
一条研究线，拥有代码、数据源和一组 Experiments。

## Protocol
Experiment 开始前写下的问题、假设、自变量、控制变量、指标和停止规则。Protocol 先于 Run 存在。

## Run
Runner 的一次执行：一份解析后的 Config × 一个 Seed。Run 不可变；失败、中断、被杀的 Run 也要保留。

## Runner
拥有 train/eval/predict 循环的执行器。研究代码通过 hook 或 step 函数插入，不重写循环。

## Run Store
本地优先的 Run 与 Manifest 集合，是实验事实的来源。Tracker 只是它的可选索引。

## Seed
Run 的一个维度。多个 Seed 是多个 Run，不是 Run 内部的循环。

## Sweep
一组 Trial，探索一个搜索空间。

## Tracker
可选的索引与可视化镜像。Tracker 不是事实来源；Run Store 才是。

## Trial
属于某个 Sweep 的 Run。
