PYTHON ?= python

.PHONY: train eval inspect runs configs sweep report gc-dry test doctor

train:
	$(PYTHON) commands/train.py --config configs/base.yaml

eval:
	@test -n "$(CHECKPOINT)" || (echo "usage: make eval CHECKPOINT=outputs/runs/.../checkpoints/last.pt"; exit 2)
	$(PYTHON) commands/eval.py --checkpoint $(CHECKPOINT)

inspect:
	@test -n "$(RUN)" || (echo "usage: make inspect RUN=<run_id>"; exit 2)
	$(PYTHON) commands/run_info.py --runs outputs/runs --run $(RUN)

runs:
	$(PYTHON) commands/runs.py --runs outputs/runs

configs:
	$(PYTHON) commands/configs.py list

sweep:
	$(PYTHON) commands/sweep.py --config configs/base.yaml

report:
	$(PYTHON) -m dlef.cli report --runs outputs/runs --claims claims --out report.md

# Change --metric-key to this project's primary metric before relying on it.
gc-dry:
	$(PYTHON) -m dlef.cli gc --runs outputs/runs --sweeps outputs/sweeps --keep-last 3 --keep-best 1 --metric-key mae --keep-tag reported --artifact-ttl-days 30

test:
	$(PYTHON) -m unittest discover -s tests -v

doctor:
	$(PYTHON) -m dlef.cli doctor
