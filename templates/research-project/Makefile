PYTHON ?= python
OUTPUT ?= $(if $(DLEF_OUTPUT_ROOT),$(DLEF_OUTPUT_ROOT),output)

.PHONY: train eval inspect runs runs-all configs sweep report gc-dry test doctor

train:
	$(PYTHON) core/commands/train.py --config input/configs/base.yaml

eval:
	@test -n "$(CHECKPOINT)" || (echo "usage: make eval CHECKPOINT=$(OUTPUT)/runs/.../checkpoints/last.pt"; exit 2)
	$(PYTHON) core/commands/eval.py --checkpoint $(CHECKPOINT)

inspect:
	@test -n "$(RUN)" || (echo "usage: make inspect RUN=<run_id>"; exit 2)
	$(PYTHON) core/tools/run_info.py --runs $(OUTPUT)/runs --run $(RUN)

runs:
	$(PYTHON) core/tools/runs.py --runs $(OUTPUT)/runs

runs-all:
	$(PYTHON) core/tools/runs.py --runs $(OUTPUT)/runs --all

configs:
	$(PYTHON) core/tools/configs.py list

sweep:
	$(PYTHON) core/commands/sweep.py --config input/configs/base.yaml

report:
	$(PYTHON) -m dlef.cli report --runs $(OUTPUT)/runs --claims $(OUTPUT)/claims --out $(OUTPUT)/reports/report.md

# Change --metric-key to this project's primary metric before relying on it.
gc-dry:
	$(PYTHON) -m dlef.cli gc --runs $(OUTPUT)/runs --sweeps $(OUTPUT)/sweeps --keep-last 3 --keep-best 1 --metric-key mae --keep-tag reported --keep-tag baseline --artifact-ttl-days 30 --sweep-ttl-days 30

test:
	$(PYTHON) -m unittest discover -s core/tests -v

doctor:
	$(PYTHON) -m dlef.cli doctor
