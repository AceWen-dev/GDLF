# ADR 0012 — Close the provenance gaps: parent links, env facts, dirty fingerprints, predictions, cards, digest lookup

## Status

Accepted.

## Context

An audit of the on-disk record chain asked one question: after many
experiments, can a coding agent pick up the output tree alone and answer
"where did this experiment come from?" The main chain
(config → run → checkpoint → eval → claim) was already digest-linked, but the
audit found seven gaps where the answer required manual work or was simply
gone:

1. eval runs did not link back to their training run (`parent_run_id` unused);
2. the environment record had no dependency versions (no torch/CUDA);
3. `code_version` flagged `-dirty` but nothing identified *which* dirty tree;
4. external warm-start weights had no evidence chain (data had cards, weights
   had nothing);
5. `EvalRecord.predictions` existed but nothing ever wrote it;
6. dataset cards were a convention, never enforced — runs could cite versions
   with no card;
7. resolving `checkpoint_digest → run/file` meant re-hashing every checkpoint
   by hand.

## Decision

1. **Lineage**: `eval.py` records `parent_run_id=<training run id>` (taken
   from the manifest embedded in the checkpoint). gc's lineage closure now
   keeps the training run alive with its eval record.
2. **Environment**: `Runner(env_extra=...)` merges adapter-provided facts into
   every manifest's `env`; the torch policy contributes torch/cuda/cudnn/gpu.
   The core stays stack-agnostic — it only knows "extra key/values".
3. **Dirty trees**: every manifest records `env.git_dirty` — a sha256 over
   `git status --porcelain` + `git diff HEAD` (not the diff itself). Clean
   tree → `dirty: false`; no git → `dirty: null`. Two dirty runs can now be
   compared for "same working tree?".
4. **Weights cards**: `model.init_checkpoint` must be `$DLEF_DATA_ROOT`-relative
   and covered by a card in `input/weights/` (path + sha256), the same way
   datasets are pinned in `input/datasets/`.
5. **Predictions**: `EvalSpec.save_predictions` writes per-batch
   `predictions/<eval_id>.jsonl` into the eval run (tensors duck-type through
   `.tolist()`); `EvalRecord.predictions` keeps the relative path. The
   template's `eval.py` saves them by default (`--no-save-predictions` to
   opt out). The flag is part of `eval_spec_digest`.
6. **Card enforcement**: `make audit` fails on a missing/unmatched dataset
   card for the current recipe, a missing/unmatched weights card, and warns
   about historical run versions that have no card.
7. **Digest lookup**: every checkpoint save maintains
   `checkpoints/digests.json` (name → sha256), and `core/tools/digests.py`
   answers `--find <sha256-prefix>` (run + file + referencing evals) and
   `--run <id>` without re-hashing the store; legacy runs are hashed on the
   fly.

## Consequences

- "Which weights produced this number" is one command, not a script.
- Reproduction knows the stack (torch/cuda per run) and the exact dirty state.
- Warm starts are provable bytes, not assumed paths.
- Every write path (`fit`, `eval`) must remember to pass `env_extra`; the
  template's `build_runner` does this once so entry points cannot forget.
- Audits hash large weight files — IO-heavy by design, the price of proof.
- Runs predating this ADR load fine; their manifests simply lack the new
  fields, and `digests.py` falls back to hashing for them.
