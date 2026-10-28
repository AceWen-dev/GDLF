PYTHON ?= python

.PHONY: train eval sweep report gc-dry test doctor

train:
	$(PYTHON) scripts/train.py --config configs/base.yaml

eval:
	@test -n "$(CHECKPOINT)" || (echo "usage: make eval CHECKPOINT=outputs/runs/.../checkpoints/last.pt"; exit 2)
	$(PYTHON) scripts/eval.py --checkpoint $(CHECKPOINT)

sweep:
	$(PYTHON) scripts/sweep.py --config configs/base.yaml

report:
	$(PYTHON) -m dlef.cli report --runs outputs/runs --claims claims --out report.md

# Change --metric-key to this project's primary metric before relying on it.
gc-dry:
	$(PYTHON) -m dlef.cli gc --runs outputs/runs --sweeps outputs/sweeps --keep-last 3 --keep-best 1 --metric-key mae --keep-tag reported --artifact-ttl-days 30

test:
	$(PYTHON) -m unittest discover -s tests -v

doctor:
	$(PYTHON) -m dlef.cli doctor
