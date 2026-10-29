# Data scripts

Standalone data-pipeline entry points. Reusable readers/cleaners live in
`myproject/data/`, not here.

```text
acquire/   # video stream, download, import, source readers
prepare/   # clean, label mapping, dedup, convert
split/     # build membership, leakage checks, benchmark materialization
```

Example:

```text
scripts/data/acquire/video_stream.py
scripts/data/prepare/dedup.py
scripts/data/split/build_holdout.py
```

A video acquisition script should write raw bytes under
`$DLEF_DATA_ROOT/raw/<source>/<date>/`, then a prepare/split script builds the
curated dataset and membership manifests.
