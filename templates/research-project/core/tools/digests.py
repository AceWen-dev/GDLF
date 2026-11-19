"""Resolve checkpoint digests: the bridge between eval records and weight files.

Every eval record references the checkpoint it evaluated by sha256. Without this
tool, answering "which weights produced this number" means hashing every
checkpoint in the store by hand. It answers in one command:

  - ``--find <sha256-prefix>``: which run/file the digest belongs to, plus the
    eval records that reference it;
  - ``--run <run_id>``: the digests of every checkpoint that run saved.

Fast path: runs record ``checkpoints/digests.json`` as they save. Legacy runs
without the index are hashed on the fly (read-only; nothing is written back).

Read-only: it never edits runs, manifests or eval records.

Examples:
  python core/tools/digests.py --find 5b1a1f99
  python core/tools/digests.py --run 20261117-101503_db70d95e_s20261020_ee81f8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.run import RunStore  # noqa: E402

_DEFAULT_OUTPUT = Path(os.environ.get("DLEF_OUTPUT_ROOT", str(ROOT / "output")))
DIGESTS_FILE = "digests.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digests(run_dir: Path) -> dict[str, str]:
    """Checkpoint name -> sha256, from the index when present, hashed otherwise."""

    index = run_dir / "checkpoints" / DIGESTS_FILE
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload:
                return {str(key): str(value) for key, value in payload.items()}
        except (OSError, ValueError):
            pass
    digests: dict[str, str] = {}
    for path in sorted((run_dir / "checkpoints").glob("*")):
        if path.is_file() and path.name != DIGESTS_FILE:
            digests[path.name] = _sha256(path)
    return digests


def _find_run(store: RunStore, selector: str) -> Path | None:
    candidate = Path(selector)
    if candidate.exists():
        return candidate if (candidate / "manifest.json").is_file() else None
    for run in store.iter_runs():
        if run.run_id == selector or run.dir.name == selector:
            return run.dir
    return None


def _eval_references(runs_root: Path, prefix: str) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for path in sorted(runs_root.rglob("evals/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        digest = str(payload.get("checkpoint_digest", ""))
        if digest.startswith(prefix):
            references.append(
                {
                    "eval_id": payload.get("eval_id", ""),
                    "split": payload.get("split", ""),
                    "run": path.parent.parent.name,
                }
            )
    return references


def _cmd_find(runs_root: Path, prefix: str) -> int:
    store = RunStore(runs_root)
    hits: list[tuple[str, str, str]] = []  # (run_id, file, digest)
    for run in store.iter_runs():
        for name, digest in _digests(run.dir).items():
            if digest.startswith(prefix):
                hits.append((run.run_id, name, digest))
    references = _eval_references(runs_root, prefix)

    if not hits and not references:
        print(f"no checkpoint matches digest prefix {prefix!r}")
        return 1

    print("checkpoints:")
    for run_id, name, digest in hits:
        print(f"  {digest[:16]}  {name}  run={run_id}")
    if references:
        print("referenced by evals:")
        for item in references:
            print(
                f"  {item['eval_id']}  split={item['split']}  eval_run={item['run']}"
            )
    return 0


def _cmd_run(runs_root: Path, selector: str) -> int:
    store = RunStore(runs_root)
    run_dir = _find_run(store, selector)
    if run_dir is None:
        print(f"run not found: {selector} (under {runs_root})", file=sys.stderr)
        return 2
    digests = _digests(run_dir)
    if not digests:
        print(f"(no checkpoints under {run_dir / 'checkpoints'})")
        return 0
    print("| file | digest |")
    print("|---|---|")
    for name, digest in sorted(digests.items()):
        print(f"| {name} | {digest[:16]}… |")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", default=str(_DEFAULT_OUTPUT / "runs"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--find",
        metavar="SHA256_PREFIX",
        help="reverse lookup: which run/file/evals match this digest prefix",
    )
    group.add_argument("--run", metavar="RUN_ID", help="list one run's checkpoint digests")
    args = parser.parse_args()

    runs_root = Path(args.runs)
    if args.find is not None:
        prefix = args.find.strip().lower()
        if len(prefix) < 6 or any(char not in "0123456789abcdef" for char in prefix):
            parser.error("--find needs at least 6 hex characters of the sha256")
        return _cmd_find(runs_root, prefix)
    return _cmd_run(runs_root, args.run)


if __name__ == "__main__":
    raise SystemExit(main())
