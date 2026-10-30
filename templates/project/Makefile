PYTHON ?= python

.PHONY: train eval inspect runs runs-all configs sweep report gc-dry test doctor

train:
	$(PYTHON) core/commands/train.py --config input/configs/base.yaml

eval:
	@test -n "$(CHECKPOINT)" || (echo "usage: make eval CHECKPOINT=output/runs/.../checkpoints/last.pt"; exit 2)
	$(PYTHON) core/commands/eval.py --checkpoint $(CHECKPOINT)

inspect:
	@test -n "$(RUN)" || (echo "usage: make inspect RUN=<run_id>"; exit 2)
	$(PYTHON) core/tools/run_info.py --runs output/runs --run $(RUN)

runs:
	$(PYTHON) core/tools/runs.py --runs output/runs

runs-all:
	$(PYTHON) core/tools/runs.py --runs output/runs --all

configs:
	$(PYTHON) core/tools/configs.py list

sweep:
	$(PYTHON) core/commands/sweep.py --config input/configs/base.yaml

report:
	$(PYTHON) -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md

# Change --metric-key to this project's primary metric before relying on it.
gc-dry:
	$(PYTHON) -m dlef.cli gc --runs output/runs --sweeps output/sweeps --keep-last 3 --keep-best 1 --metric-key mae --keep-tag reported --keep-tag baseline --artifact-ttl-days 30 --sweep-ttl-days 30

test:
	$(PYTHON) -m unittest discover -s core/tests -v

doctor:
	$(PYTHON) -m dlef.cli doctor
