"""Count how many distinct structures hide behind each ``model.name``.

Two views answer two different questions:

- ``configs`` (declared): scan ``input/configs`` and cluster by
  ``ModelConfig.structural_signature()``. How many variants does the config
  tree imply, before training anything?
- ``runs`` (observed): scan run directories and cluster by the structure
  fingerprint each run recorded. How many variants actually ran, and did the
  model code drift? A config digest that maps to more than one structure
  digest proves the model was edited in place between runs, which makes those
  runs non-comparable even though their configs are identical.

Neither view imports torch or builds a model; both read files only, so they are
safe to run in CI.

The variant count is a smell, not a verdict. Sweeping a size knob stays inside
one signature; a topology switch should have been a new ``model.name``. When
the count is high, split the model into one file per variant instead of adding
another boolean. See ``docs/adr/0005-model-variants-over-flags.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import canonical_json, load_config  # noqa: E402
from dlef.run import RunStore  # noqa: E402
from core.config import ProjectConfig  # noqa: E402

# Mirrors adapters.torch.structure.DEFAULT_FILENAME. Duplicated on purpose: the
# tool must stay importable without torch so it can run in a lint/CI environment.
STRUCTURE_FILENAME = "model_structure.json"

DEFAULT_MAX_VARIANTS = 3


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _config_paths(configs_dir: Path) -> list[Path]:
    """The documented config layout; sweeps are spaces, not resolved configs."""

    paths: list[Path] = []
    base = configs_dir / "base.yaml"
    if base.is_file():
        paths.append(base)
    experiments = configs_dir / "experiments"
    if experiments.is_dir():
        paths.extend(sorted(experiments.rglob("*.yaml")))
    return paths


def _signature(model: Any) -> tuple[dict[str, Any], bool]:
    """Return ``(signature, used_fallback)`` for one model config."""

    declared = getattr(model, "structural_signature", None)
    if callable(declared):
        return dict(declared()), False
    return dict(model.to_dict()), True


def _diff(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(base) | set(other))
    return {key: other.get(key) for key in keys if base.get(key) != other.get(key)}


def collect_configs(configs_dir: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid: list[dict[str, Any]] = []
    fallback = False

    for path in _config_paths(configs_dir):
        try:
            config = load_config(str(path), ProjectConfig)
        except Exception as exc:  # noqa: BLE001 - report the bad config, keep scanning
            invalid.append({"config": _rel(path), "error": str(exc)})
            continue
        model = config.schema.model
        signature, used_fallback = _signature(model)
        fallback = fallback or used_fallback
        groups[model.name].append({"config": _rel(path), "signature": signature})

    rows: list[dict[str, Any]] = []
    for name in sorted(groups):
        entries = groups[name]
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            clusters[canonical_json(entry["signature"])].append(entry)

        baseline_key = max(clusters, key=lambda key: (len(clusters[key]), key))
        baseline = clusters[baseline_key][0]["signature"]
        variants = []
        for key in sorted(clusters, key=lambda k: (-len(clusters[k]), k)):
            members = clusters[key]
            signature = members[0]["signature"]
            variants.append(
                {
                    "configs": len(members),
                    "diff": _diff(baseline, signature),
                    "signature": signature,
                    "files": [member["config"] for member in members],
                    "is_baseline": key == baseline_key,
                }
            )
        rows.append(
            {
                "model": name,
                "configs": len(entries),
                "variants": len(variants),
                "baseline": baseline,
                "rows": variants,
            }
        )

    return {
        "view": "configs",
        "configs_dir": str(configs_dir),
        "signature_source": "ModelConfig.to_dict() (no structural_signature)"
        if fallback
        else "ModelConfig.structural_signature()",
        "total_configs": sum(len(value) for value in groups.values()),
        "total_variants": sum(row["variants"] for row in rows),
        "groups": rows,
        "invalid": invalid,
    }


def collect_runs(runs_root: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []

    if runs_root.is_dir():
        for run in RunStore(runs_root).iter_runs():
            path = run.directory / STRUCTURE_FILENAME
            if not path.is_file():
                missing.append(run.run_id)
                continue
            structure = json.loads(path.read_text(encoding="utf-8"))
            config = (
                json.loads(run.config_path.read_text(encoding="utf-8"))
                if run.config_path.is_file()
                else {}
            )
            model_name = str((config.get("model") or {}).get("name") or "<unknown>")
            groups[model_name].append(
                {
                    "run_id": run.run_id,
                    "status": run.manifest.status,
                    "config_digest": run.manifest.config_digest,
                    "structure_digest": structure.get("digest", ""),
                    "topology_digest": structure.get("topology_digest", ""),
                    "n_params": structure.get("n_params"),
                    "n_trainable": structure.get("n_trainable"),
                }
            )

    rows: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    for name in sorted(groups):
        entries = groups[name]
        by_config: dict[str, Counter] = defaultdict(Counter)
        for entry in entries:
            by_config[entry["config_digest"]][entry["structure_digest"]] += 1
        for config_digest, counter in sorted(by_config.items()):
            if len(counter) > 1:
                drift.append(
                    {
                        "model": name,
                        "config_digest": config_digest,
                        "structures": [
                            {"digest": digest, "runs": count}
                            for digest, count in sorted(counter.items())
                        ],
                    }
                )
        rows.append(
            {
                "model": name,
                "runs": len(entries),
                "topologies": len({entry["topology_digest"] for entry in entries}),
                "structures": len({entry["structure_digest"] for entry in entries}),
                "entries": entries,
            }
        )

    return {
        "view": "runs",
        "runs_root": str(runs_root),
        "total_runs": sum(len(value) for value in groups.values()),
        "groups": rows,
        "drift": drift,
        "runs_without_fingerprint": missing,
    }


def _print_configs(payload: dict[str, Any], limit: int) -> int:
    print(f"configs_dir: {payload['configs_dir']}")
    print(f"signature source: {payload['signature_source']}")
    print(
        f"total: {payload['total_configs']} configs -> "
        f"{payload['total_variants']} declared structures"
    )
    for group in payload["groups"]:
        print()
        print(f"model.name: {group['model']}")
        print(
            f"  configs: {group['configs']}   distinct structures: {group['variants']}"
        )
        for index, variant in enumerate(group["rows"], 1):
            label = "(baseline)" if variant["is_baseline"] else (
                "  ".join(f"{k}={v}" for k, v in variant["diff"].items()) or "(empty)"
            )
            print(f"  {index:2d}. [{variant['configs']:3d}] {label}")
            for path in variant["files"][:limit]:
                print(f"          {path}")
            if len(variant["files"]) > limit:
                print(f"          ... +{len(variant['files']) - limit}")
    for item in payload["invalid"]:
        print(f"\nINVALID {item['config']}: {item['error']}", file=sys.stderr)
    return 0


def _print_runs(payload: dict[str, Any], limit: int) -> int:
    print(f"runs_root: {payload['runs_root']}")
    print(f"total: {payload['total_runs']} runs with a structure fingerprint")
    for group in payload["groups"]:
        print()
        print(f"model.name: {group['model']}")
        print(
            f"  runs: {group['runs']}   topologies: {group['topologies']}   "
            f"structures: {group['structures']}"
        )
        for entry in group["entries"][:limit]:
            print(
                f"    {entry['run_id']}  struct#{entry['structure_digest'][:12]}  "
                f"topo#{entry['topology_digest'][:12]}  "
                f"params={entry['n_params']} trainable={entry['n_trainable']}  "
                f"config#{entry['config_digest'][:12]}"
            )
        if len(group["entries"]) > limit:
            print(f"    ... +{len(group['entries']) - limit}")

    if payload["drift"]:
        print()
        print("DRIFT: one config digest produced more than one structure.")
        print("       The model code changed in place; these runs are not comparable.")
        for item in payload["drift"]:
            print(f"  {item['model']} config#{item['config_digest'][:12]}:")
            for structure in item["structures"]:
                print(
                    f"    struct#{structure['digest'][:12]}  {structure['runs']} run(s)"
                )
    if payload["runs_without_fingerprint"]:
        count = len(payload["runs_without_fingerprint"])
        print(f"\nnote: {count} run(s) predate the structure fingerprint")
    return 0


def _check(payload: dict[str, Any], max_variants: int) -> int:
    if payload["view"] == "runs":
        return 1 if payload["drift"] else 0

    crowded = [
        f"{group['model']} ({group['variants']} structures)"
        for group in payload["groups"]
        if group["variants"] > max_variants
    ]
    if crowded:
        print(
            f"CHECK FAILED: more than {max_variants} structures under one "
            f"model.name: {', '.join(crowded)}",
            file=sys.stderr,
        )
    if payload["invalid"]:
        print(
            f"CHECK FAILED: {len(payload['invalid'])} invalid config(s)",
            file=sys.stderr,
        )
    return 1 if crowded or payload["invalid"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    configs_parser = sub.add_parser(
        "configs", help="cluster declared configs by structural signature"
    )
    configs_parser.add_argument(
        "--configs-dir",
        default=str(ROOT / "input" / "configs"),
        help="config tree to scan (default: input/configs)",
    )

    runs_parser = sub.add_parser(
        "runs", help="cluster recorded runs by structure fingerprint"
    )
    runs_parser.add_argument(
        "--runs",
        default=str(
            Path(os.environ.get("DLEF_OUTPUT_ROOT", str(ROOT / "output"))) / "runs"
        ),
    )

    for item in (configs_parser, runs_parser):
        item.add_argument("--json", action="store_true", help="machine-readable output")
        item.add_argument("--limit", type=int, default=3, help="files/runs listed")
        item.add_argument(
            "--check",
            action="store_true",
            help="exit 1 on too many variants (configs) or on drift (runs)",
        )
        item.add_argument(
            "--max-variants",
            type=int,
            default=DEFAULT_MAX_VARIANTS,
            help=f"configs view: fail above this many structures per model.name "
            f"(default {DEFAULT_MAX_VARIANTS})",
        )

    args = parser.parse_args()
    if args.command == "configs":
        payload = collect_configs(Path(args.configs_dir))
        printer = _print_configs
    else:
        payload = collect_runs(Path(args.runs))
        printer = _print_runs

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        printer(payload, args.limit)
    if args.check:
        return _check(payload, args.max_variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
