# ADR 0006 — Candidates sit between runs and claims

## Status

Accepted.

## Context

Runs accumulate much faster than reported results. After a few sweeps the run
store contains dozens or hundreds of entries, most of which are dead ends. Two
existing mechanisms do not solve the triage problem:

- **Run tags** (`baseline`, `reported`, `debug`) are written when the run starts.
  Triage happens after seeing results, and a completed manifest must not be
  edited. Tags also carry no reason, status or link to a claim.
- **Claims** are reported numbers with an evidence chain
  (`claim → eval → run`). A claim requires eval ids, is per-number rather than
  per-experiment, has no lifecycle, and is a final artifact — not a working set.

There was also no guarantee that curated evidence survived retention: `gc`
protected tags, sweeps and lineage, but not claims or any user-defined
shortlist.

## Decision

Add a first-class **Candidate** service between the run store and claims:

```text
runs → candidates → claims → report
```

- A candidate is a named pointer to one or more runs (normally one
  configuration across seeds), plus `reason`, `next`, an optional val `metric`,
  a priority and free tags.
- `run_ids` is a snapshot taken at creation time; later runs do not silently
  join a candidate.
- Status lifecycle: `candidate → promoted | rejected`; `promoted → archived`.
  `candidate` and `rejected` require a real reason. `promoted` requires at least
  one claim id.
- Candidates are compared on validation metrics. Test evaluation happens only
  after promotion, so the shortlist also limits test-set access.
- `dlef report` renders the shortlist above the claims; `make report` includes
  it.
- Retention protects runs referenced by `candidate`/`promoted` candidates and
  runs referenced by claims (`claim → eval → run`). `rejected`/`archived`
  candidates do not protect their runs.
- Candidates are never generated automatically from a metric ranking: they
  encode judgment (novelty, cost, robustness, an anomaly to inspect), which a
  leaderboard cannot express.

## Consequences

- The working set is small and readable without deleting raw runs.
- A rejected candidate remains as a recorded dead end even after its runs are
  garbage-collected.
- Curated evidence survives `gc`, so claims do not silently become `MISSING`.
- There is one more small JSON layer to maintain. An unmaintained candidate
  keeps protecting old runs until it is rejected or archived.
- Claims do not require a candidate; the layer is a recommended path, not a
  foreign-key constraint.
- Projects that do not need triage can ignore the layer entirely.
