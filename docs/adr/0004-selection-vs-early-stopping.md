# ADR 0004 — Selection is separate from early stopping

## Status

Accepted.

## Context

The Runner originally saved `best.pt` only inside the early-stopping branch. A
run with `early_stopping.enabled=false` therefore had no validation-selected
checkpoint and fell back to `last.pt`. The same `early_stopping.monitor` also
controlled which validation metric selected the best checkpoint, conflating two
different decisions.

## Decision

- `selection` is a typed top-level config block: `monitor`, `mode`,
  `smooth_window`, `checkpoint`.
- `early_stopping` remains a stopping policy: `monitor`, `patience`,
  `min_delta`, `min_epochs`, `smooth_window`.
- Selection runs every epoch when its monitor exists in validation metrics,
  regardless of `early_stopping.enabled`.
- `selection.monitor=""` falls back to `early_stopping.monitor` for backward
  compatibility.
- The training entry point loads the checkpoint named by
  `selection.checkpoint`:
  `best_raw`, `best_ema`, `final_ema`, or `last`.
- `best_raw` and `best_ema` are independent checkpoint trajectories;
  `final_ema` and `last` remain end-of-training snapshots.
- Test evaluation remains a separate run and is never used for selection.

## Consequences

- Runs can select a best validation checkpoint without enabling early stopping.
- Existing configs without `selection` keep working through the fallback.
- A config can now explicitly select EMA or raw weights instead of relying on
  an implicit checkpoint-priority rule.
- Selection and stopping may deliberately use different metrics, but the
  resolved config records both and the log prints both.
