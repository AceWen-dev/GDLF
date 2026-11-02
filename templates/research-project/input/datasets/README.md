# Datasets

Dataset **bytes** stay outside the repo under `$DLEF_DATA_ROOT`. This directory
holds the small, reviewable metadata that pins a training/validation dataset.

Suggested layout:

```text
input/datasets/
  <dataset>/
    <version>.yaml
```

Example card:

```yaml
name: <dataset-name>
version: <version>
manifest: curated/<dataset-name>/<version>/manifest.json
manifest_sha256: "<sha256 of manifest.json>"
rows: <N>
splits:
  train: <n>
  val: <n>
  test: <n>
sample_id: <sample-id-column>
group_by: <group-column>
label:
  field: <label-column>
  unit: <unit>
provenance:
  source: "<where the data came from>"
  created: "YYYY-MM-DD"
```

Rules:

- The card is metadata, not a copy of images or tensors.
- The full membership manifest stays on `$DLEF_DATA_ROOT`; record its path and
  SHA-256 here.
- `name` / `version` must match `data.name` / `data.version` in the run config.
- Split counts are a quick audit; they do not replace `tools/audit_data.py`.
- If the manifest changes, create a new dataset version/card; do not edit an old
  card in place.
