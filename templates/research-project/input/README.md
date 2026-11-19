# Input

Immutable experiment inputs:

```text
input/
  configs/       # experiment variants
  benchmarks/    # frozen benchmark manifests
  datasets/      # dataset cards: name, version, manifest path/hash, split counts
  weights/       # provenance cards for warm-start/pretrained weight files
```

Dataset bytes are not stored here; they live outside the repo under
`$DLEF_DATA_ROOT` (see `core/data/paths.py`). `input/` contains the declarations
and small manifests that pin the data — and, for anything a model loads
(`model.init_checkpoint`), the card that pins those bytes too. `make audit`
verifies both.
