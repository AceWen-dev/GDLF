"""First-pass data audit for a DLEF project.

Checks what can be checked generically from the configured loaders:

- provenance cards: dataset card for the current recipe (and its manifest
  hash), weights card for ``model.init_checkpoint``, and cards for every
  version historical runs actually used;
- split sizes and DataSpec;
- numeric field statistics, NaNs/infs;
- duplicate sample ids within a split;
- sample-id overlap across splits;
- group overlap across splits (when PROJECT.yaml names a group key).

It is not a proof of no leakage. If your batches do not expose `sample_id` or
`group_by`, the audit reports a warning; add those keys to the dataset or audit
the membership manifest manually.

Read-only: it does not edit data, configs, runs, or manifests. Weights cards
are verified by hashing the referenced files, so large warm-start checkpoints
make this step IO-heavy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import ConfigError, load_config  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data import make_loader  # noqa: E402
from core.data.paths import data_root  # noqa: E402

ROLES = ("train", "val", "test")


@dataclass
class Stat:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    bad: int = 0

    def add(self, value: Any) -> None:
        if torch.is_tensor(value):
            if not (value.is_floating_point() or value.dtype in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }):
                return
            tensor = value.detach().to("cpu").float()
            finite = torch.isfinite(tensor)
            self.bad += int((~finite).sum().item())
            valid = tensor[finite]
            if valid.numel() == 0:
                return
            self.n += int(valid.numel())
            self.total += float(valid.sum().item())
            self.total_sq += float((valid * valid).sum().item())
            self.minimum = min(self.minimum, float(valid.min().item()))
            self.maximum = max(self.maximum, float(valid.max().item()))
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                self.bad += 1
                return
            self.n += 1
            self.total += number
            self.total_sq += number * number
            self.minimum = min(self.minimum, number)
            self.maximum = max(self.maximum, number)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self.add(item)

    def to_dict(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0, "bad": self.bad}
        mean = self.total / self.n
        variance = max(0.0, self.total_sq / self.n - mean * mean)
        return {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "bad": self.bad,
        }


def _batch_size(batch: Any) -> int:
    if not isinstance(batch, dict):
        return 0
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def _values(batch: Any, key: str) -> list[Any]:
    if not isinstance(batch, dict) or key not in batch:
        return []
    value = batch[key]
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value.item()]
        return value.detach().to("cpu").reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_placeholder(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip() or (
        value.strip().startswith("<") and value.strip().endswith(">")
    )


def _audit_role(
    config: ProjectConfig,
    role: str,
    *,
    sample_key: str,
    label_key: str,
    group_key: str,
    limit: int,
) -> dict[str, Any]:
    loader = make_loader(config.data, role, config.seed)
    spec = loader.spec
    fields: dict[str, Stat] = defaultdict(Stat)
    ids: list[str] = []
    groups: list[str] = []
    batches = 0
    samples = 0

    for batch_index, batch in enumerate(loader):
        if limit and batch_index >= limit:
            break
        batches += 1
        samples += _batch_size(batch)
        for key, value in batch.items():
            fields[str(key)].add(value)
        for value in _values(batch, sample_key):
            ids.append(str(value))
        for value in _values(batch, group_key):
            groups.append(str(value))

    field_stats = {key: value.to_dict() for key, value in fields.items()}
    label_stats = field_stats.get(label_key)
    return {
        "batches": batches,
        "samples": samples,
        "spec": spec.to_dict() if spec is not None else None,
        "fields": field_stats,
        "labels": label_stats,
        "ids": ids,
        "groups": groups,
    }


def _dataset_card(name: str, version: str, input_root: Path | None = None) -> Path:
    base = input_root if input_root is not None else ROOT
    return base / "input" / "datasets" / name / f"{version}.yaml"


def _weights_cards(input_root: Path | None = None) -> dict[str, dict[str, Any]]:
    base = input_root if input_root is not None else ROOT
    cards: dict[str, dict[str, Any]] = {}
    for path in sorted((base / "input" / "weights").glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(payload, dict) and payload.get("path"):
            cards[str(payload["path"])] = payload
    return cards


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_cards(
    config: ProjectConfig,
    errors: list[str],
    warnings: list[str],
    input_root: Path | None = None,
) -> None:
    """Provenance cards for the current recipe: dataset + warm-start weights."""

    name, version = config.data.name, config.data.version
    card_path = _dataset_card(name, version, input_root)
    if not card_path.is_file():
        errors.append(
            f"dataset card missing: input/datasets/{name}/{version}.yaml "
            "(record the manifest path+sha256 and split counts)"
        )
    else:
        try:
            card = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"dataset card unreadable: {exc}")
            card = {}
        if isinstance(card, dict):
            rel = str(card.get("manifest", "") or "")
            expected = str(card.get("manifest_sha256", "") or "")
            if rel and expected:
                try:
                    manifest_file = data_root() / rel
                except RuntimeError:
                    warnings.append("DLEF_DATA_ROOT not set; skipped dataset manifest hash check")
                else:
                    if not manifest_file.is_file():
                        errors.append(f"dataset card points to a missing manifest: {rel}")
                    elif _sha256(manifest_file) != expected:
                        errors.append(
                            f"dataset manifest hash mismatch: {rel} changed since the card was written"
                        )

    init = str(getattr(config.model, "init_checkpoint", "") or "")
    if not init:
        return
    try:
        target = data_root() / init
    except RuntimeError:
        warnings.append("DLEF_DATA_ROOT not set; skipped weights card check")
        return
    if not target.is_file():
        errors.append(f"model.init_checkpoint not found under $DLEF_DATA_ROOT: {init}")
        return
    card = _weights_cards(input_root).get(init)
    if card is None:
        errors.append(
            f"weights card missing for model.init_checkpoint {init}: "
            "add input/weights/<name>.yaml with its sha256"
        )
    elif str(card.get("sha256", "")) != _sha256(target):
        errors.append(f"weights card hash mismatch: {init} changed since the card was written")


def _audit_run_data_versions(
    runs_root: Path,
    warnings: list[str],
    input_root: Path | None = None,
) -> None:
    """Historical runs must stay explainable: every used version needs a card."""

    if not runs_root.is_dir():
        return
    missing: Counter[str] = Counter()
    for manifest_path in runs_root.rglob("manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for version in (payload.get("data_versions") or {}).values():
            text = str(version)
            if "@" not in text:
                continue
            name, _, ver = text.partition("@")
            if not _dataset_card(name, ver, input_root).is_file():
                missing[text] += 1
    for version, count in sorted(missing.items()):
        warnings.append(f"{count} run(s) use {version} without a dataset card")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "input" / "configs" / "base.yaml")
    )
    parser.add_argument("--project", default=str(ROOT / "PROJECT.yaml"))
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="max batches per split; 0 = all (default: 20)",
    )
    parser.add_argument(
        "--runs",
        default=str(ROOT / "output" / "runs"),
        help="run store root for the historical dataset-card check",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    errors: list[str] = []
    warnings: list[str] = []
    brief: dict[str, Any] = {}
    project_path = Path(args.project)
    if project_path.is_file():
        try:
            payload = yaml.safe_load(project_path.read_text(encoding="utf-8"))
            brief = payload if isinstance(payload, dict) else {}
        except (OSError, yaml.YAMLError) as exc:
            warnings.append(f"cannot read PROJECT.yaml: {exc}")

    data = brief.get("data", {}) if isinstance(brief, dict) else {}
    splits = brief.get("splits", {}) if isinstance(brief, dict) else {}
    sample_key = str(data.get("sample_id", "sample_id"))
    label_key = str(data.get("labels", "label"))
    group_key = str(splits.get("group_by", "group"))
    check_groups = group_key.strip().lower() not in {"none", "n/a", "na"}
    for key in (sample_key, label_key, group_key):
        if _is_placeholder(key):
            warnings.append(f"PROJECT.yaml key {key!r} is a placeholder")

    try:
        config = load_config(args.config, ProjectConfig).schema
    except (OSError, ConfigError, ValueError) as exc:
        print(f"ERROR: cannot load config {args.config}: {exc}", file=sys.stderr)
        return 1

    # Provenance cards first: cheap checks that fail fast before any data
    # iteration, so a missing card is never buried under batch statistics.
    _audit_cards(config, errors, warnings)
    _audit_run_data_versions(Path(args.runs), warnings)

    results: dict[str, Any] = {}
    try:
        for role in ROLES:
            results[role] = _audit_role(
                config,
                role,
                sample_key=sample_key,
                label_key=label_key,
                group_key=group_key if check_groups else "",
                limit=args.limit,
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: audit failed: {exc}", file=sys.stderr)
        return 1

    id_sets = {
        role: set(result["ids"]) for role, result in results.items() if result["ids"]
    }
    group_sets = (
        {
            role: set(result["groups"])
            for role, result in results.items()
            if result["groups"]
        }
        if check_groups
        else {}
    )
    overlaps: dict[str, int] = {}
    for left in ROLES:
        for right in ROLES:
            if left >= right:
                continue
            if left in id_sets and right in id_sets:
                overlaps[f"ids:{left}/{right}"] = len(
                    id_sets[left] & id_sets[right]
                )
            if left in group_sets and right in group_sets:
                overlaps[f"groups:{left}/{right}"] = len(
                    group_sets[left] & group_sets[right]
                )

    for role, result in results.items():
        if not result["ids"]:
            warnings.append(f"{role}: sample_id {sample_key!r} not found")
        else:
            duplicates = len(result["ids"]) - len(set(result["ids"]))
            if duplicates:
                errors.append(f"{role}: {duplicates} duplicate sample ids")
        if check_groups and not result["groups"]:
            warnings.append(f"{role}: group key {group_key!r} not found")
        for field, stats in result["fields"].items():
            if stats.get("bad", 0):
                errors.append(f"{role}.{field}: {stats['bad']} NaN/inf values")
    if _is_placeholder(label_key) or all(
        result["labels"] is None for result in results.values()
    ):
        warnings.append(f"labels key {label_key!r} not found")
    for name, count in overlaps.items():
        if count:
            errors.append(f"split overlap {name}: {count} shared values")

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not errors,
                    "config": args.config,
                    "keys": {
                        "sample_id": sample_key,
                        "labels": label_key,
                        "group_by": group_key,
                    },
                    "splits": {
                        role: {
                            key: value
                            for key, value in result.items()
                            if key not in {"ids", "groups"}
                        }
                        for role, result in results.items()
                    },
                    "overlaps": overlaps,
                    "errors": errors,
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not errors else 1

    print(f"DATA AUDIT {args.config}")
    print(f"data: {config.data.name}@{config.data.version}")
    for role, result in results.items():
        print(
            f"[{role}] batches={result['batches']} samples={result['samples']} "
            f"ids={len(set(result['ids']))} groups={len(set(result['groups']))}"
        )
        for field, stats in result["fields"].items():
            if stats.get("n", 0) == 0:
                print(f"  {field}: no numeric values")
                continue
            print(
                f"  {field}: n={stats['n']} mean={stats['mean']:.6g} "
                f"std={stats['std']:.6g} min={stats['min']:.6g} "
                f"max={stats['max']:.6g} bad={stats['bad']}"
            )
    if overlaps:
        print("overlaps:")
        for name, count in sorted(overlaps.items()):
            print(f"  {name}: {count}")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"RESULT: {'OK' if not errors else 'FAIL'} ({len(errors)} errors)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
