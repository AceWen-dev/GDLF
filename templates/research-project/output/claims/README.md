# Claims — the registry of reported numbers

One JSON file per claim: `output/claims/<claim_id>.json`.

A **claim** is a number you intend to report (in a paper, a slide, or
`output/reports/report.md`), stored together with the evidence that produced it.
It is the glue between *what you measured* and *what you will write down*:

```text
claim -> eval_ids -> eval records -> checkpoint digest -> run (config / code / data)
```

Hand-copied numbers are the classic way results get corrupted: the wrong seed,
an old checkpoint, a val metric read as test, a number that survives a config
change. Claims fix that: you cite eval ids, `dlef report` resolves the chain,
and it prints `MISSING` / `MISMATCH` / `WARN` when the chain is broken.

## A note on this folder

Everything else under `output/` is generated: the Runner writes `runs/`,
`artifacts/`, `sweeps/`, `test_access/`, and you regenerate `reports/` from
claims. **This folder is the exception.** Claims are written only by you (or
your agent), through `ClaimStore`, and the files are small enough to commit.
Treat claims as your citation list: they should stay true even after old runs
are garbage-collected.

## File format

One claim, one file, file name = `claim_id`:

| Field | Required | Meaning |
| --- | --- | --- |
| `claim_id` | ✓ | stable unique name, e.g. `baseline-test-mae` |
| `text` | ✓ | the sentence a human will read, e.g. "Baseline test MAE, mean of 3 seeds" |
| `eval_ids` | ✓ | one or more EvalRecord ids this number came from (e.g. one test eval per seed) |
| `metric` | | which key inside `EvalRecord.metrics` the report resolves |
| `value` | | the reported value (e.g. the seed mean); if omitted, the report shows the per-eval value of `metric` |
| `unit` | | unit string, e.g. `mm` |
| `benchmark_id` | | pin the frozen benchmark; the report flags `MISMATCH` if an eval record was produced on a different one |
| `tags` | | free-form labels, e.g. `{"experiment": "baseline"}` |
| `notes` | | recipe, seeds, aggregation method |
| `created_at` | | filled automatically |

## Where eval ids come from

`core/commands/eval.py` prints them when it runs:

```text
eval run: 20261102-144845_3b34c71f_s7_72eba9
eval record: output/runs/my-project/baseline-eval/<eval_run_id>/evals/<eval_id>-test-<hex>.json
```

The same files sit in the eval run's `evals/` directory. `dlef report
--runs output/runs` resolves every id by scanning that tree, so an id from a
different output root will show as `MISSING`.

## Creating a claim

```python
import statistics

from dlef.claim import Claim, ClaimStore

eval_ids = (  # one test-split eval per seed
    "20261102-a1-test-0001",
    "20261102-a2-test-0002",
    "20261102-a3-test-0003",
)
values = (0.83, 0.85, 0.86)  # same metric read from those eval records

ClaimStore("output/claims").add(
    Claim(
        claim_id="baseline-test-mae",
        text="Baseline test MAE, mean of 3 seeds",
        eval_ids=eval_ids,
        metric="mae",
        value=statistics.mean(values),
        tags={"experiment": "baseline"},
        notes="best.pt selected on val; seeds 1/2/3",
    )
)
```

Rules already enforced by `ClaimStore`:

- `eval_ids` must be non-empty;
- duplicate `claim_id` → `ClaimError` (quotes, slashes, empty ids rejected too);
- the file is written under the store root as `<claim_id>.json`.

A claim may aggregate several eval records (seeds, cohorts); then `value`
should be the aggregate you actually report, and `notes` should say how it was
computed. One claim = one reported number/table/figure; do not stretch one
claim over several stories.

## Generating the report

```bash
python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md
# or
make report
```

The report resolves each `eval_id` → `(run, eval record)` and lists test-access
counts. Statuses you can see in a row:

- `MISSING` — the eval id does not exist under `--runs` (typo, gc-deleted run,
  different output root). Fix the id or re-run the eval.
- `MISMATCH` — the claim's `benchmark_id` differs from the eval record's. The
  number does not measure the benchmark you said; the claim must not be
  reported until this is fixed. Compare only claims with the same
  `benchmark_id`.
- `WARN n` — the test split of this test set has been evaluated `n` times.
  `n > 1` means the number is no longer a fresh test result; the report keeps
  the warning visible on purpose. Do not hide it, and do not tune on test.

## Conventions

- Name claims for what they say: `<experiment>-<metric>[-<variant>]`, or the
  final-story name (`main-result`). Stable, lowercase, no spaces.
- Claims about test numbers must cite test-split eval runs — never a metric
  copied from a training run or TensorBoard. Keep model selection on val.
- Do not hand-write the JSON or overwrite a claim's meaning in place. To fix a
  wrong claim, `ClaimStore.remove(claim_id)` and re-add a corrected one (or use
  a new id and let the old one die).
- Claims live in `output/`, but unlike runs they are safe to commit; do not
  hand-copy their values into papers — regenerate `output/reports/report.md`
  and cite that.
