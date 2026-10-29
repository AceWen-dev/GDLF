# DLEF Contracts v0（domain-agnostic）

> **状态**：草案，供 review。不是实现。
>
> **硬规则**：core 里不出现任何领域词。image / text / audio / video、bbox / depth / mm、classification / detection / segmentation、RL / LLM、case / lesion —— 这些只能出现在 adapter 或 example 里。
>
> **d 的含义**：把原本藏在 Runner 里的横切关注点显式化。first-class 不等于 mandatory：有独立 contract、config schema、registry name、contract test，但一个实验可以不使用 scheduler / callback。

---

## 1. Contract map

| # | Contract | 类型 | 必需？ | core 拥有 | adapter 拥有 | 核心不变量 |
|---|---|---|---|---|---|---|
| 1 | Data | runtime | 是 | `DataSpec` / `Dataset` / `Batch` 校验 | 数据集实现、transform | 只读；split/transform 版本化 |
| 2 | Model | runtime | 是 | `Model` / `ModelOutput` | 网络结构 | 无 I/O、无 config、无 device、无 loss |
| 3 | Loss | runtime | 是 | `Loss` / `LossOutput` | loss 数学 | `total` + 命名 `terms` |
| 4 | Optimizer | runtime | 训练必需 | `Optimizer` / spec / param-group 校验 | 优化器实现 | 所有可训参数恰好覆盖一次 |
| 5 | Scheduler | runtime | 可选 | `Scheduler` / spec | 调度数学 | step 单位显式；状态可恢复 |
| 6 | Callback | runtime | 可选 | 生命周期、顺序、隔离 | callback 实现 | 只做副作用；顺序确定 |
| 7 | Metric | runtime | 评测必需 | `Metric` / `MetricResult` | 指标数学 | 按 split 命名；确定性 |
| 8 | Task / Runner | runtime | 是 | Runner 循环、`StepOutput`、resume | Task 的 step | step 纯函数；不写磁盘 |
| 9 | Config | service | 是 | schema、resolver、digest | 配置值 | 严格、已解析、有 hash |
| 10 | Run / Artifact | service | 是 | `RunManifest` / `RunStore` / `ArtifactStore` | 无 | manifest 先写；不可变；内容寻址 |
| 11 | Eval | service | 是 | `EvalSpec` / `EvalRecord` / benchmark id / 聚合 | 指标选择 | 独立 run；test access 计数 |
| 12 | CLI / Launcher | service | 是 | commands、launcher 协议 | 项目入口配置 | 薄入口；一条命令一个工作流 |
| 13 | Registry | service | 是 | register / create / resolve | 实现注册 | 不重名；显式 `class_path` 逃生口 |
| 14 | Sweep | service | 是 | `SweepSpec` / `TrialSpec` / queue / record | `trial_fn` | 一个 trial = 一个 run；失败隔离 |
| 15 | Retention | service | 是 | plan / apply / reference checks | 无 | 默认 dry-run；引用/血缘/sweep 保护 |
| 16 | TrainingPolicy | runtime | 可选（默认 NoOp） | hook 生命周期 | AMP / accum / clip / DDP | backward/step 必须走 policy |
| 17 | Claim | service | 是 | Claim / ClaimStore / report | 无 | claim → eval → run；test access 可见 |
| 18 | TestAccess | service | 是 | append-only log + count | 无 | 每个 test set 的 test 访问计数 |

**runtime contract**：每次实验都会实例化的对象。
**service**：框架提供的运行时设施，不是用户实现的算法组件。

---

## 2. 通用数据流

```text
Config
  │
  ▼
Registry ──► Task { Dataset, Model, Loss, Optimizer, Scheduler, Metrics, Callbacks }
  │
  ▼
Runner { train / eval / predict }
  │
  ├──► Run { Manifest, Metrics, Checkpoints }
  │        │
  │        └──► ArtifactStore
  │
  └──► EvalRecord ──► Compare / Report
```

`Task` 是唯一组装领域组件的地方。core 不知道 `Batch` 里是什么，也不知道 `predictions` 的语义；它只负责生命周期、契约和记录。

---

## 3. Runtime contracts

### 3.1 Data

```python
Batch = Mapping[str, Any]

@dataclass(frozen=True)
class DataSpec:
    name: str
    version: str
    schema: Mapping[str, Any]       # keys / dtype / shape / optional
    splits: Mapping[str, str]       # split name -> manifest path or digest
    transform: str                  # transform version
    stats: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class Dataset(Protocol):
    def __iter__(self) -> Iterator[Batch]: ...
```

约束：

- `Dataset` 是 **batch iterable**；batching / shuffling / padding / collation 都是 adapter 的责任。这样 streaming 数据、非张量数据也能接入。
- 数据只读；预处理产物写到新的 versioned artifact，不原地改。
- 数据集字节放在项目外（`DLEF_DATA_ROOT`）；项目只保存代码和小 manifest，不保存图片/表格副本。
- split 必须物化；不允许在 `__iter__` 里随机划分。
- split 是同一份 curated 数据的 membership 视图（sample id / 相对路径），不复制 train/val/test 数据目录。
- `DataSpec.version` 标识一次 data recipe（dataset + split + transform）；run manifest 记录 `role -> version`，旧 run 不随新数据变化。
- train / eval transform 必须是两个独立声明。
- 第一个 batch 必须通过 `DataSpec.schema` 校验（keys / dtype / shape）。
- core 不关心 batch 的语义；只关心它是 typed mapping。

### 3.1.1 Benchmark

`BenchmarkSpec`（`dlef/benchmark.py`）把测试集和评价协议绑成一个不可变 id：

```text
test_set_id   = dataset + split membership digest
benchmark_id  = test_set_id + label version + eval transform + metric versions
              + aggregation + thresholds + evaluator version
```

约束：

- `test_set_id` 用于防泄漏和 test access 计数；只改标签版本、eval transform 或指标不应重置访问计数。
- `benchmark_id` 用于比较；只有相同 `benchmark_id` 的 EvalRecord 才能直接放在一张表里。
- 改测试集、标签、eval transform、指标实现/聚合/阈值、评测代码版本 → 新 benchmark。
- benchmark manifest 是小型 JSON，应随项目提交；测试数据实体仍在 `DLEF_DATA_ROOT`。
- 构建 membership、查重、group/time/label 泄漏检查属于数据层；core 只拥有 id 和审计字段。

### 3.2 Model

```python
@dataclass(frozen=True)
class ModelOutput:
    predictions: Any
    aux: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class Model(Protocol):
    def __call__(self, batch: Batch) -> ModelOutput: ...
```

约束：

- 不做 I/O、不读 config、不移动 device、不计算 loss。
- 初始化必须可复现；初始化方案记录在 manifest。
- `predictions` 的语义由 adapter 定义；core 只要求它存在。
- `aux` 用于中间量、诊断、可视化；不进入 loss 的默认路径。

### 3.3 Loss

```python
@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    terms: Mapping[str, Tensor] = field(default_factory=dict)

@runtime_checkable
class Loss(Protocol):
    def __call__(self, output: ModelOutput, batch: Batch) -> LossOutput: ...
```

约束：

- `total` 必须是标量；`terms` 必须命名。
- reduction / normalization 必须显式（per-sample / per-token / sum / mean）。
- 权重来自 config，不硬编码。
- 每个 term 都记录；禁止只记录 total。
- loss 不放在 model 里；model 只产生输出。

### 3.4 Optimizer

```python
@runtime_checkable
class Optimizer(Protocol):
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

约束：

- 参数量、param groups、学习率、weight decay 必须能从 config 重建。
- 校验：所有 `requires_grad=True` 的参数恰好出现在一个 param group 里；无遗漏、无重复。
- optimizer state 必须进 checkpoint。
- core 只定义协议；具体优化器由 adapter 注册。

### 3.5 Scheduler

```python
@runtime_checkable
class Scheduler(Protocol):
    def step(self) -> None: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

约束：

- `step_unit` 必须显式：`epoch` 或 `step`。
- 缺省是 constant LR；没有 scheduler 是合法配置。
- `last_epoch` / state 必须进 checkpoint。
- 不在 core 里实现具体 schedule；adapter 注册。

### 3.6 Callback

```python
@runtime_checkable
class Callback(Protocol):
    def on_run_start(self, ctx: "RunContext") -> None: ...
    def on_epoch_start(self, ctx: "RunContext") -> None: ...
    def on_step_end(self, ctx: "RunContext", step: StepOutput) -> None: ...
    def on_epoch_end(self, ctx: "RunContext") -> None: ...
    def on_run_end(self, ctx: "RunContext") -> None: ...
    def on_exception(self, ctx: "RunContext", error: BaseException) -> None: ...
```

约束：

- callback 只做副作用；不修改 batch / model 参数。需要修改的，走 Runner 的显式 hook。
- 顺序确定；注册顺序就是执行顺序。
- 单个 callback 失败默认记录并继续；`fail_fast` 可配置。
- 典型用途：early stopping、EMA、profiling、gradient 监控、外部通知。
- 不是 loop 机制的替代品：grad clip、AMP、accumulation 仍在 Runner。

### 3.7 Metric

```python
@dataclass(frozen=True)
class MetricResult:
    values: Mapping[str, float]
    meta: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class Metric(Protocol):
    def update(self, output: ModelOutput, batch: Batch) -> None: ...
    def compute(self) -> MetricResult: ...
    def reset(self) -> None: ...
```

约束：

- 有状态累加器；`update` 逐 batch，`compute` 在 epoch/eval 结束。
- 指标名、版本、方向（max/min）、聚合方式必须可声明。
- 指标本身不知道 split；`EvalSpec` 绑定 split。
- 确定性：同样输入、同样顺序，结果一致。
- core 只定义协议；具体指标由 adapter 注册。

### 3.8 Task / Runner

```python
@dataclass(frozen=True)
class StepOutput:
    loss: Tensor | None
    terms: Mapping[str, Tensor] = field(default_factory=dict)
    predictions: Any | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

@runtime_checkable
class Task(Protocol):
    def get_model(self) -> Model: ...
    def train_step(self, batch: Batch) -> StepOutput: ...
    def val_step(self, batch: Batch) -> StepOutput: ...
    def configure_optimizer(self, model: Model) -> Optimizer: ...
    def configure_scheduler(self, optimizer: Optimizer) -> Scheduler | None: ...
    def configure_metrics(self) -> Mapping[str, Metric]: ...
    def configure_callbacks(self) -> Sequence[Callback]: ...
```

约束：

- **标准路径**：`train_step` 只返回 loss；Runner 负责 `backward` / `step` / AMP / accumulation / clipping。
- **逃生口**：需要自定义反向/多优化器/对抗训练时，Task 实现 `custom_train_step(batch)`，Runner 只负责生命周期和记录。逃生口必须显式声明，不能偷偷绕过。
- Task 不写磁盘；日志、artifact、checkpoint 由 Runner / RunStore 处理。
- Task 不读全局 config；所需值在构造时注入。
- Runner 拥有：device、AMP、grad accumulation、grad clip、checkpoint/resume、NaN/Inf fail-fast、吞吐记录、callback 调度。
- Runner 必须通过 kill-resume 测试：中断后续跑，结果与不中断一致（或差异可解释）。

### 3.9 Runner 的 public surface

```python
class Runner:
    def fit(self, task: Task, dataset: Dataset, config: Config) -> Run: ...
    def evaluate(self, task: Task, dataset: Dataset, spec: EvalSpec) -> EvalRecord: ...
    def predict(self, task: Task, dataset: Dataset, spec: PredictSpec) -> Artifact: ...
```

`Runner` 是 core 的具体实现，不是 Protocol。它不包含领域逻辑。

### 3.10 TrainingPolicy

```python
@runtime_checkable
class TrainingPolicy(Protocol):
    @property
    def is_main_process(self) -> bool: ...
    def setup(self) -> None: ...
    def on_fit_start(self, task, optimizer, config) -> None: ...
    def forward_context(self): ...
    def scale_loss(self, loss): ...
    def backward(self, loss): ...
    def should_step(self, step: int) -> bool: ...
    def should_flush(self, step: int) -> bool: ...
    def optimizer_step(self, task, optimizer) -> None: ...
    def zero_grad(self, optimizer) -> None: ...
    def on_fit_end(self, task, optimizer) -> None: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

约束：

- 默认 `NoOpTrainingPolicy`：每 batch backward/step/zero；`is_main_process=True`。
- `setup()` 在 run 创建前调用（DDP rank 决定谁写 run）。
- `forward_context()` 包住 train/val step（AMP autocast）。
- `scale_loss()` 只缩放送入 backward 的 loss；日志仍记录原始 loss。
- `should_step()` / `should_flush()` 控制 accumulation 边界；epoch 末尾 flush 残余梯度。
- `optimizer_step()` 负责 unscale / clip / step / scaler update；`zero_grad()` 负责清零。
- `state_dict()` 进 checkpoint（例如 scaler）；`load_state_dict()` 恢复。
- DDP 只让 rank 0 写 run / log / checkpoint；多 rank eval 与分布式 resume 是 v0 之外。

---

## 4. Services

### 4.1 Config

```python
@dataclass(frozen=True)
class Config:
    raw: Mapping[str, Any]
    schema: Any
    resolved: Mapping[str, Any]
    digest: str
    source: str = ""
```

约束：

- schema 是 dataclass tree；类型化。
- unknown key 报错；缺省值只在 schema 里定义。
- 运行前 dump `config.resolved.yaml`；digest 进入 RunManifest。
- secrets 只从 env 读；dump 时 redact。
- 不允许 config 里写任意 Python 表达式或 import；`class_path` 是唯一受控逃生口。

### 4.2 Run / Artifact

```python
@dataclass
class RunManifest:
    run_id: str
    project: str
    experiment: str
    config_digest: str
    code_version: str
    data_versions: Mapping[str, str]
    env: Mapping[str, Any]
    seed: int
    status: str                      # running / completed / failed / killed
    artifacts: Mapping[str, str]
    started_at: str
    ended_at: str | None = None
    parent_run_id: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
```

约束：

- 第一个 step 之前写 `status: running`；结束原子更新。
- 完成的 run 不可变；修正用新的 run 或 append-only annotation。
- artifact 以内容 hash 寻址；checkpoint 去重。
- RunStore 是事实来源；Tracker 只是可选索引。
- 本地优先；远程存储是 mirror，不是前提。

### 4.2.1 Checkpoint codec

`Runner` 接受一个 `CheckpointCodec`：

- `JsonCheckpointCodec`（默认，suffix `.json`）；
- adapter 可提供 `TorchCheckpointCodec`（suffix `.pt`，`torch.save`/`torch.load`）。

core 拥有 `Checkpoint` envelope；codec 拥有字节序列化。这样 core 不需要 import 任何张量库。

### 4.3 Eval

```python
@dataclass(frozen=True)
class EvalSpec:
    checkpoint: str
    split: str
    metrics: Sequence[str]
    aggregation: str
    ci_method: str | None = None
    test_access_policy: str = "once_per_claim"
    benchmark_id: str = ""
    test_set_id: str = ""

@dataclass(frozen=True)
class EvalRecord:
    eval_id: str
    split: str
    checkpoint_digest: str
    data_version: str
    eval_spec_digest: str
    metrics: Mapping[str, Any]
    predictions: str | None = None
    created_at: str = ""
    test_access: int = 0
    benchmark_id: str = ""
    test_set_id: str = ""
```

约束：

- eval 是独立 run；有自己的 manifest。
- metrics 按 split 命名；train/val/test 不混。
- test access 按 `test_set_id` 计数；同一 claim 重复访问要显式记录。
- 测试集 + 标签 + eval transform + metric 实现/聚合 = 一个 benchmark；
  `benchmark_id` 不同的 EvalRecord 不可直接比较。
- 多 seed 聚合、CI 是 eval 层的能力，不是 metric 实现的责任。
- 每个 claim 必须引用 eval id，并应记录 benchmark id。

### 4.4 CLI / Launcher

```python
@runtime_checkable
class Command(Protocol):
    name: str
    def run(self, argv: Sequence[str]) -> int: ...

@runtime_checkable
class Launcher(Protocol):
    def submit(self, run_spec: Mapping[str, Any]) -> str: ...
    def status(self, job_id: str) -> Mapping[str, Any]: ...
    def cancel(self, job_id: str) -> None: ...
```

命令集合（v0）：

```text
train     # 训练一个 run
eval      # 评测一个 checkpoint
sweep     # 生成/提交一组 run
compare   # 从 eval records 生成对比表
report    # 生成报告
doctor    # preflight：环境、数据、配置、磁盘
```

约束：

- 脚本是薄入口；业务逻辑在 Task/adapter。
- 一条命令一个工作流；同一份 config 在 local / cluster / cloud 上语义一致。
- `doctor` 在训练前检查：config、数据路径、GPU、磁盘、依赖。
- `smoke` 必须能在 CPU + tiny fixture 上跑完 10 step。

### 4.5 Registry

```python
class Registry:
    def register(self, kind: str, name: str, aliases: Sequence[str] = ()) -> Callable: ...
    def create(self, kind: str, config: Mapping[str, Any], expected_type: type | None = None) -> Any: ...
    def resolve(self, kind: str, name: str) -> type: ...
```

约束：

- 同一 `kind/name` 不允许重复注册。
- `class_path` 是显式逃生口；优先级高于 `name`。
- 支持 entry-point 插件发现。
- `create` 必须校验 `expected_type`；不兼容的组件不能装进插槽。
- Registry 是装配机制，不是运行对象。

### 4.6 Sweep

```python
@dataclass(frozen=True)
class SweepSpec:
    name: str
    params: Mapping[str, ParamSpace]
    method: str = "grid"          # grid | random
    max_trials: int | None = None
    seed: int = 0

@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    index: int
    params: Mapping[str, Any]

@dataclass
class TrialContext:
    sweep_id: str
    trial_id: str
    index: int
    gpu_ids: tuple[int, ...] = ()

@dataclass(frozen=True)
class TrialCommand:
    argv: Sequence[str]
    env: Mapping[str, str] = ...
    cwd: str | None = None
    timeout_s: float | None = None
```

约束：

- 一个 trial = 一个 run；trial 的 params 作为 config override。
- `grid` 只接受 `Choice`；`random` 支持 `Uniform` / `LogUniform` / `IntUniform`，必须有 `max_trials`。
- 失败隔离：一个 trial 失败记为 `failed`，不杀整个 sweep；sweep 状态为 `partial`。
- 两个 backend：
  - `LocalSweep` + `LocalQueue`：thread-based，in-process `trial_fn`。
  - `SubprocessSweep` + `SubprocessQueue`：每个 trial 一个 subprocess；`CUDA_VISIBLE_DEVICES` 由 GPU 池分配；stdout/stderr 落 `trials/<trial_id>.log`。
- 子进程 env 契约：`CUDA_VISIBLE_DEVICES` / `DLEF_SWEEP_ID` / `DLEF_TRIAL_ID` / `DLEF_TRIAL_INDEX` / `DLEF_GPU_IDS` / `DLEF_TRIAL_RESULT`。
- 子进程把 `TrialResult` JSON 写到 `DLEF_TRIAL_RESULT`；没有结果文件时，exit 0 = completed，非 0 = failed；非 0 永远覆盖 completed。
- `SweepStore` 落 `sweeps/<sweep_id>/sweep.json`，记录 spec / trials / results；logs/results 在 `trials/`。
- `LocalSweep` / `SubprocessSweep` 只负责调度和记录；trial 内部构建什么由 `trial_fn` / `command_fn` 决定。

### 4.7 Retention

```python
@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int = 5
    keep_best: int = 0
    keep_every: int = 0
    metric_key: str = ""
    metric_mode: str = "min"
    keep_tags: frozenset[str] = frozenset({"reported"})
    protect_status: frozenset[str] = frozenset({"running"})
    min_age_days: float = 0.0
    artifact_ttl_days: float = 30.0
    protect_sweep_runs: bool = True
    sweep_ttl_days: float | None = None
```

约束：

- `plan` 先于任何删除；默认 `apply(dry_run=True)`。
- 保护：per-experiment last/best/every-N；protected tags/status；young；parent lineage closure；sweep-referenced runs。
- Artifact 保护：被 kept run 引用的 digest；或 age < TTL。只有无引用且超过 TTL 的 artifact 才 delete。
- 删除 run 是 `shutil.rmtree`；删除 artifact 是内容寻址 store 的 `delete`。
- 所有 action 带 reason 和 size；plan 可审计。

### 4.8 Claim / TestAccess

```python
@dataclass
class Claim:
    claim_id: str
    text: str
    eval_ids: tuple[str, ...]
    metric: str = ""
    value: float | None = None
    unit: str = ""
    notes: str = ""
    benchmark_id: str = ""
```

约束：

- Claim 必须至少引用一个 eval id；report 生成时通过 `RunStore.find_eval` 解析。
- Claim 应记录 `benchmark_id`；如果 claim 和 eval 的 benchmark 不同，report 标 `MISMATCH`。
- `TestAccessLog` 是 append-only；`Runner.evaluate` 对 `split == "test"` 按 `test_set_id` 记录计数，写入 `EvalRecord.test_access`。
- 报告里 `test_access > 1` 标记 `WARN`；v0 只警告，不阻塞。
- Claim/report 是本地 JSON / Markdown；没有 Web UI。
- `ClaimStore` 一个 claim 一个 JSON 文件；`dlef report` 生成 Markdown。

---

## 5. 13 条不变量

1. **Run identity**：一个 run = 一份 resolved config × 一个 seed；seed 是维度，不是循环。
2. **Config**：第一个 step 前完全解析、dump、hash；unknown key 报错。
3. **Manifest**：第一个 step 前写 `running`；状态只允许 `running / completed / failed / killed`。
4. **Isolation**：训练代码不写 run 目录之外；artifact 走 store。
5. **Loss**：`total` + 命名 `terms`；每个 term 都记录。
6. **Metrics**：按 split 命名；eval 是独立 run。
7. **Lineage**：claim → eval → checkpoint → run，逐级可追。
8. **Resume**：checkpoint 包含 model / optimizer / scheduler / scaler / RNG / data position。
9. **Data**：只读；split 和 transform 版本化。
10. **Tracker**：只是索引；RunStore 是 canonical。
11. **Core purity**：core 无领域词；领域代码只在 adapter。
12. **First-class**：每个 contract 都有 schema、registry name、contract test。
13. **Benchmark**：test membership + labels + eval transform + metric protocol 冻结；只有相同 `benchmark_id` 才可直接比较。

---

## 6. core / adapter / example 边界

```text
core/
  contracts/      # Protocol: Dataset, Model, Loss, Optimizer, Scheduler, Callback, Metric, Task
  config/         # schema, resolver, digest
  runner/         # train / eval / predict loop
  policy/         # TrainingPolicy: AMP / accumulation / clipping / DDP seam
  sweep/          # SweepSpec, trials, thread/subprocess GPU-aware queues
  run/            # manifest, run store, artifact store
  claim/          # claims + test-access log
  retention/      # plan / apply / reference checks
  benchmark/      # test set + metric protocol -> benchmark id
  eval/           # eval spec, eval record, aggregation
  registry/       # register / create / resolve
  cli/            # train, eval, sweep, compare, report, doctor

adapters/
  torch/          # reference adapter: DataLoader/Model/Loss/Optimizer/Scheduler/Metric/Task + codec
  data/ model/ loss/ optimizer/ scheduler/ callback/ metric/ tracker/ launcher/

examples/
  toy/            # 线性回归；只证明 contract 能串起来
  torch_regression/  # synthetic tensor example via adapters/torch

templates/
  project/        # copyable project skeleton; imports dlef, does not vendor it
```

依赖方向：`core ← adapters ← examples`。core 不许 import adapters 或 examples。
项目模板 `templates/project/` 依赖 core 包，不复制 core（见 ADR 0002）。

**Reference adapter:** `adapters/torch/` 实现 `TorchDataLoader` / `TorchModel` / `TorchLoss` / `TorchOptimizer` / `TorchScheduler` / `TorchMetric` / `TorchTask` / `TorchTrainingPolicy` 和 `TorchCheckpointCodec`。它 import `dlef`；`dlef` 从不 import 它。

---

## 7. 明确不放进 core 的东西

- 任何模型结构、loss 公式、metric 实现、数据格式。
- 任何领域词（见文首）。
- 任何 tracker 厂商 SDK。
- 任何实验 protocol 的正文（那是文档，不是代码）。
- 任何“最佳实践”的默认值（默认值属于 template，不属于 core）。
- 任何可视化 / dashboard。

---

## 8. 已定 / 待定

**已定（2026-10-27）：**

1. **Runner 边界**：标准路径由 Runner 拥有 loop，`Task.train_step` 只返回 loss；`custom_train_step` 是显式逃生口。
2. **契约实现**：`typing.Protocol` + 装配时 runtime check；不强制继承。每个 contract 仍需 contract test（runtime check 只查方法存在，不查签名）。
3. **Config schema**：纯 dataclass + `__post_init__`/`validate()`；不引入 pydantic 作为硬依赖。
4. **Checkpoint envelope**：core 定义统一 `Checkpoint` dataclass（model / optimizer / scheduler / rng / epoch / step / manifest / data_position）；各组件 state 对 core 不透明。序列化由 `CheckpointCodec` 负责：默认 JSON，torch adapter 用 `.pt`。
5. **Sweep**：core 包含 `SweepSpec` + grid/random + `LocalQueue`/`GpuPool` + `SubprocessQueue`/`SubprocessSweep`；trial 内部由 `trial_fn` / `command_fn` 构建。
6. **Retention**：`RetentionPolicy` + `RetentionService.plan/apply`；默认 dry-run；引用检查、血缘闭包、sweep 保护。
7. **TrainingPolicy**：backward/step 走 policy seam；torch adapter 提供 AMP + accumulation + clipping + optional DDP；默认 NoOp 保持单进程行为。
8. **Delivery**：项目模板 `templates/project/` 依赖版本化 core，不 vendor core（ADR 0002）。
9. **Claim / TestAccess**：Claim 对象 + Markdown report；test-split access 按 `test_set_id` 计数，report 标记 WARN；跨 benchmark 显示不可比。

**待定：**

1. **Sweep backend**：v0 是本地 thread/subprocess；Slurm/cloud/remote backend 何时进来？
2. **HPO**：Optuna / Ray Tune 是 adapter 还是 core 的 `SweepSpec` 生成器？
3. **Launcher**：v0 只有同步本地 launcher；Slurm/cloud 的 job 状态机何时进入 core？
4. **Tracker**：v0 不接；接入时是 core 的 service，还是 adapter？
5. **Retention UX**：trash/restore、保留策略预设、`gc` 的 JSON 输出。
6. **DDP completeness**：多 rank eval、分布式 checkpoint resume、rank-aware sweep。
7. **Claim policy**：`test_access > 1` 是警告还是阻塞；claim/report UI。
