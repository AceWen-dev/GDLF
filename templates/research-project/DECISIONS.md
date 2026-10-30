# DECISIONS.md — project decision log

<!--
Append-only. One entry per hard-to-reverse or surprising decision.
Do not rewrite or delete old entries; supersede them with a new entry.
Reversible choices (a normal experiment variant, a hyperparameter, a local
refactor) belong in the config or a run, not here.
-->

## When to write

- data source/version, split membership, grouping or leakage rule;
- frozen benchmark: test membership, labels, eval transform, metric protocol;
- baseline model/loss/optimizer/scheduler choice;
- output root, logging, retention or artifact policy future runs depend on;
- anything a future agent would otherwise re-litigate.

## When not to write

- code style, typos, local refactors;
- a normal experiment variant (it is a run);
- a reversible hyperparameter (sweep it).

IDs are sequential (`D-0001`, `D-0002`, ...) and are never reused.

## Entry template

```markdown
### D-000X — <title>
- Date: YYYY-MM-DD
- Status: proposed | accepted | superseded by D-000Y
- Context:
- Decision:
- Alternatives:
- Consequences:
- Evidence: config path, run id, eval id, benchmark id, commit
- Owner:
```

## Entries

<!-- Keep the template entries; the first project-specific decision is D-0003. -->

### D-0001 — Start from the DLEF research-project template

- Date: <YYYY-MM-DD>
- Status: accepted
- Context: new project; need config resolution, run manifests, eval records and
  a frozen benchmark from day one.
- Decision: start from `templates/research-project`; keep `core/task.py` as the
  composition root; install `dlef` as a package instead of vendoring it.
- Alternatives: an ad-hoc training repo; copying `dlef/` into the project.
- Consequences: every run has a manifest/config/checkpoint; test evaluation
  requires a frozen benchmark; project code stays in `core/`.
- Evidence: `AGENTS.md`, `PROJECT.yaml`, DLEF `docs/TUTORIAL.md`
- Owner: <name>

### D-0002 — Synthetic regression is smoke-only

- Date: <YYYY-MM-DD>
- Status: accepted
- Context: the template ships a synthetic 1-D regression placeholder to prove
  the wiring.
- Decision: use it only for smoke tests; replace data/model/loss/metric before
  running or reporting a real experiment.
- Alternatives: treat the placeholder as a baseline.
- Consequences: synthetic runs are tagged `smoke`/`debug` and never reported.
- Evidence: `input/configs/base.yaml`, `core/tests/test_smoke.py`
- Owner: <name>
