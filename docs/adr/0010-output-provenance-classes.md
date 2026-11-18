# ADR 0008 — Output is organized by provenance

## Status

Accepted.

## Context

In real projects `output/` drifted into a junk drawer: a 212 MB `reports/`
mixing generated reports with hand-written diagnoses, manually exported eval
JSONs, copied stdout, pptx files and probe caches; a hand-maintained `INDEX.md`
that went stale and then lied about the current state; fine-tuned weights (1.6
GB) stored as if they were experiment output. The template's rule "never edit
generated output by hand" collided with the daily need to write analysis, and
reality routed around the rule.

The root cause: `output/` was defined by what it is *not* (not input, not
code), while files actually differ along three axes — provenance (who/what
wrote it), lifecycle (immutable / rebuildable / disposable) and size.

## Decision

A file's location under `output/` is decided by **where it comes from**, and
every file belongs to exactly one provenance class:

| Class | Dirs | Who writes | Editable? | Lifecycle |
|---|---|---|---|---|
| Recorded | `runs/`, `sweeps/`, `test_access/` | DLEF only | never | immutable; `dlef gc` |
| Judged | `candidates/`, `claims/` | the owner, via the tools | via tools only | status lifecycle |
| Rendered | `INDEX.md`, `reports/` | `make index` / `make report` | never by hand | delete freely, regenerate |
| Scratch | `scratch/` | anyone | freely | may vanish any time |

- The front door is `output/INDEX.md`, generated (`make index`): funnel,
  experiments, recent runs, shortlist, claims and approvers. Hand-maintained
  overviews are forbidden — they go stale and then mislead.
- Rendered views carry a generated stamp and are never hand-edited.
- Scratch is the sanctioned junk drawer: probes, exported JSONs, copied
  stdout, ad-hoc plots. Claims, candidates and INDEX must never reference it.
- Numbers do not get parked in `reports/` as JSON exports; evidence lives in
  eval records, and views are regenerated from them.
- Files a future run reads are data, not output: `$DLEF_DATA_ROOT` with a
  dataset card.

## Consequences

- Agents get a legal home for drafts, so `reports/` stays a real report.
- The index is always current because nobody maintains it by hand.
- Deleting `scratch/` never loses evidence; deleting `reports/` + `INDEX.md`
  loses nothing at all.
- Hand-written analysis moves to the repository (docs/analysis), where git
  versions it, instead of living in a disposable output tree.
