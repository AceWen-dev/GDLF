# Weights

Provenance cards for weight files a model **loads** (warm start, pretrained
backbones), the same way `input/datasets/` pins data. A config that says
`model.init_checkpoint: some/path.pt` is only half the story; the card proves
*which bytes* that path contained when the runs were made.

Dataset bytes stay outside the repo under `$DLEF_DATA_ROOT`; so do weight
files. This directory holds small YAML cards only.

## Card format

One file per weight input, named after the weights (any name works — the card
is matched to the config by its `path`):

```yaml
name: cc-warmstart-gdecay-ep20
path: cc_warmstart/gdecay_ep20.pt   # relative to $DLEF_DATA_ROOT
sha256: "<sha256 of the file>"
kind: warmstart                      # warmstart | pretrained | other
provenance:
  source: "which run/recipe produced these weights"
  created: "YYYY-MM-DD"
```

## Rules

- `make audit` verifies that every `model.init_checkpoint` in the current
  config exists under `$DLEF_DATA_ROOT`, has a card, and that the card's
  sha256 still matches the file. A mismatch means the weights changed after
  the card was written — fix the card only after you understand the change.
- Hashing large checkpoints makes `make audit` IO-heavy; that is the price of
  provable warm starts.
- Changing the weights file after runs were trained invalidates the provenance
  of those runs; move new weights to a new path and a new card instead.
