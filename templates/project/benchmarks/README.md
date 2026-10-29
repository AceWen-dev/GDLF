# Benchmarks

One JSON file per frozen benchmark. Create it with:

```bash
python commands/benchmark.py create \
  --config configs/base.yaml \
  --out benchmarks/synthetic-test-v1.json
```

Commit the manifest. Treat it as immutable: if the test membership, labels,
eval transform or metric protocol changes, create a new benchmark file/version
(do not edit the old one). `eval.py --split test` verifies the manifest before
running.
