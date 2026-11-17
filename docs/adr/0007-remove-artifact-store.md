# ADR 0007 — Remove the artifact store

## Status

Accepted.

## Context

`ArtifactStore` copied every checkpoint, `best`/`best_ema`/`final_ema` snapshot
and eval record into a second content-addressed directory
(`output/artifacts/<sha[:2]>/<sha>/<name>`). In practice:

- checkpoints are unique per run, so content addressing provided no dedup;
- the store doubled disk usage for the largest files;
- the run directory already contains the checkpoint, eval record and config, so
  there was no second consumer;
- claims resolve `eval_id → checkpoint_digest → run`, not through artifacts;
- retention had to reason about artifact references and a separate TTL.

Real projects deleted `output/artifacts` and lost nothing, which confirms that
the store was a duplicate, not a record.

## Decision

- The **run directory is the artifact**: checkpoints, eval records, predictions
  and any other files live under `output/runs/.../`.
- `Runner` no longer writes checkpoints or eval records to an artifact store.
- `Run.save_artifact()` is removed. `RunManifest.artifacts` is accepted when
  reading old manifests but is never serialized again; new manifests omit it.
- The template no longer creates `output/artifacts` or wires an `ArtifactStore`.
- Retention no longer plans or deletes artifacts. The old `output/artifacts`
  directory is a legacy duplicate and can be deleted once by hand after
  verifying that run directories are intact.
- `ArtifactStore` remains importable as a deprecated shim with a failing
  `put()`, and `RunStore`/`Runner` still accept an `artifact_store=` argument
  and ignore it. This keeps older project wiring importable; it cannot create
  new artifacts.

## Consequences

- Disk usage drops by the size of the duplicated checkpoints.
- One storage layout instead of two; `run_info.py` points at files in the run
  directory.
- `gc` only manages runs and sweeps, plus candidate/claim protection.
- Projects that called `run.save_artifact()` directly must move those files
  into the run directory. The shim does not silently keep the old behavior.
- Old manifests still load; their `artifacts` field is ignored. Existing
  artifact directories are not deleted automatically because DLEF no longer
  manages them.
