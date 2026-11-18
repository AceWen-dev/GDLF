# Input

Immutable experiment inputs:

```text
input/
  configs/       # experiment variants
  benchmarks/    # frozen benchmark manifests
  datasets/      # dataset cards: name, version, manifest path/hash, split counts
```

Dataset bytes are not stored here; they live outside the repo under
`$DLEF_DATA_ROOT` (see `core/data/paths.py`). `input/` contains the declarations
and small manifests that pin the data.
