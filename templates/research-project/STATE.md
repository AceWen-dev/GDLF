# STATE.md — project state

<!--
Living snapshot, not a log. Read this before planning or editing.
Update it at the end of every work session / before handoff.
History belongs in DECISIONS.md and output/runs/; do not paste run logs here.
Keep it short enough to read in ~2 minutes. If STATE and the RunStore disagree,
the RunStore wins; fix STATE.
-->

---
project: <project-id>
phase: setup            # setup | data | baseline | ablation | benchmark | report
updated: YYYY-MM-DD
owner: <name>
primary_metric: <metric>
best_run: "-"           # run id
benchmark: "-"          # benchmark id
---

## 1. Question and scope

<!-- Mirror PROJECT.yaml. A semantic change here needs a DECISIONS.md entry. -->

- Research question:
- Unit of analysis:
- Input:
- Output:
- Primary metric:
- Success criterion:
- Non-goals:

## 2. Data and splits

- Data root: `$DLEF_DATA_ROOT/<...>` (no absolute paths in this file)
- Dataset / version:
- Sample id:
- Labels / units:
- Split policy:
  - train:
  - val:
  - test:
  - grouping / leakage rule:
- Frozen benchmark: `input/benchmarks/<name>.json` (`benchmark_id: ...`)
- Known data issues:

## 3. Current baseline

- Config: `input/configs/experiments/<experiment>/<variant>.yaml`
- Run id:
- Checkpoint:
- Tags:
- Val metrics:
- Test metrics (only with a frozen benchmark):
- What is frozen vs still moving:

## 4. Experiments

Only the experiments that matter for the current question. The full index is
`make runs` / `python core/tools/runs.py`.

| experiment | config | status | <primary metric> | run | note |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## 5. Open questions

- [ ] ...
- [ ] ...

## 6. Next actions

- [ ] ...
- [ ] ...

## 7. Known issues / gotchas

- ...

## 8. Handoff checklist

- [ ] `STATE.md` reflects the latest run and metrics.
- [ ] Every number above has a run/eval id; none is copied from `train.log`.
- [ ] `DECISIONS.md` has an entry for any hard-to-reverse choice.
- [ ] Test split was not used for tuning.
- [ ] Runs are tagged (`baseline`, `reported`, `debug`); `dlef gc` dry-run reviewed.
- [ ] No secrets or absolute data paths in this file.

## 9. Update protocol

1. Read `make runs` / `run_info.py`; pick the run, do not guess.
2. Update sections 3–7, replacing stale values (this is a snapshot, not a log).
3. Set `updated:` and `phase:` in the front matter.
4. Append to `DECISIONS.md` if the change is hard to reverse or changes data,
   split, benchmark, metric, baseline, logging, or retention policy.
5. If a number is unknown, write `TBD` or `UNKNOWN`; never invent it.
