PYTHON ?= python

.PHONY: test toy sweep gc-dry torch-demo torch-sweep torch-ddp template-smoke doctor clean

test:
	$(PYTHON) -m unittest discover -s tests -v

toy:
	$(PYTHON) examples/toy/run.py

sweep:
	$(PYTHON) examples/toy/sweep.py

gc-dry:
	$(PYTHON) -m dlef.cli gc --runs examples/toy/outputs/runs --sweeps examples/toy/outputs/sweeps --keep-last 1 --keep-best 0 --artifact-ttl-days 0 --no-protect-sweep-runs

torch-demo:
	$(PYTHON) examples/torch_regression/run.py

torch-sweep:
	$(PYTHON) examples/torch_regression/sweep.py

torch-ddp:
	CUDA_VISIBLE_DEVICES="" $(PYTHON) -m torch.distributed.run --standalone --nproc_per_node=2 examples/torch_regression/run.py

template-smoke:
	cd templates/research-project && PYTHONPATH=../.. $(PYTHON) -m unittest discover -s core/tests -v

doctor:
	$(PYTHON) -m dlef.cli doctor

clean:
	rm -rf examples/toy/outputs examples/torch_regression/outputs
