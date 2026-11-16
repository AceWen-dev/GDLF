"""Thin training entry point: config -> data/task -> Runner.fit -> eval."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import load_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from dlef.seed import seed_everything  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data import make_loaders  # noqa: E402
from core.runtime import (  # noqa: E402
    build_policy,
    build_runner,
    parse_overrides,
    resolve_output_root,
    write_trial_result,
)
from core.task import build_task  # noqa: E402


def _parse_tags(items: list[str]) -> dict[str, str]: #以下划线开头的表示内置函数只能被内部调用，函数作用是把命令行的参数转化为标准字典
    tags: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")#partition是字符串的内置方法加入item="learning_rate=0.01"使用就会以=为分割符返回（"learning_rate","=","0.01"),如果没有分割符就把改str分配给第一个字符
        name = name.strip()#strip是字符串方法，用于刮掉字符串前后的空格，制表符或者换行符等
        if not name: #空字符报错
            raise SystemExit(f"invalid --tag: {item!r}")
        tags[name] = value if separator else "true" #如果切出等号，和没有切除等号的三元表达
    return tags


def main() -> int:
    parser = argparse.ArgumentParser() #创建一个解析对象parder
    parser.add_argument("--config", required=True) #required=True表示必填参数
    parser.add_argument(
        "--set",
        dest="overrides",#改名如果没有这个调用参数是args.set,改完后变成args.overrides
        action="append", #允许的多次调用比如--set A=1.--set B=2
        default=[], #默认为空列表
        metavar="KEY=VALUE", #在终端输入-h显示键值对显示更漂亮
        help="typed override, e.g. --set model.name=linear --set loss.name=huber",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="attach a tag to this run, e.g. --tag baseline --tag debug; "
        "tags can protect runs from gc",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    config = load_config(
        args.config,
        ProjectConfig,
        overrides=[overrides] if overrides else [],
    )
    seed_everything(config.schema.seed)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    policy = build_policy(config.schema, ddp=ddp)#打包配置中的训练策略
    policy.setup()#启动训练策略
    device = policy.device

    runner = build_runner(resolve_output_root(config.schema, ROOT), policy) #打造训练引擎
    train_loader, val_loader = make_loaders(                               #打造数据引擎
        config.schema.data,
        config.schema.seed,
        distributed=ddp,
    )
    task = build_task(config.schema, device=device)  #打造模型，损失，优化器，调度器
    #开始训练
    run = runner.fit(
        task,
        train_loader,
        val_loader,
        config,
        tags=_parse_tags(args.tag),
    ) #训练的核心代码
    if run is None:
        print(f"rank {getattr(policy, '_rank', 0)}: training complete (non-main)")#getattr获得对象属性policy._rank
        return 0

    selection = getattr(config.schema, "selection", None)
    checkpoint_policy = getattr(selection, "checkpoint", "best_raw")
    checkpoint = run.checkpoint_path_for(checkpoint_policy)
    if not checkpoint.is_file(): #is_file()表示是否文件存在
        fallback = run.best_checkpoint_path
        checkpoint = fallback if fallback.is_file() else run.checkpoint_path
    if checkpoint.is_file():
        optimizer = task.configure_optimizer(task.get_model())
        runner.load_checkpoint(checkpoint, task, optimizer)

    record = runner.evaluate(
        task,
        val_loader,
        EvalSpec(checkpoint=str(checkpoint), split="val"),
        run,
    )
    write_trial_result(run, record.metrics)
    print(f"run: {run.run_id}")
    print(f"selection checkpoint: {checkpoint_policy} -> {checkpoint}")
    print(f"metrics: {record.metrics}")
    print(f"manifest: {run.manifest_path}")
    print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
