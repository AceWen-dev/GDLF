from __future__ import annotations

import contextlib
import os
from typing import Any, Mapping

import torch

from dlef.contracts import Optimizer, Task


class TorchTrainingPolicy:
    """AMP + gradient accumulation + clipping + optional DDP.

    ``setup()`` initializes the process group and rank before the Runner creates
    the run. ``on_fit_start`` wraps the model with DDP and creates the AMP
    scaler. ``forward_context()`` is the autocast context used by the Runner
    around train/val steps.

    DDP scope: training is per-rank; only rank 0 writes the run, logs and
    checkpoints. Multi-rank evaluation and distributed checkpoint resume are
    not covered in v0.
    """

    def __init__(
        self,
        *,
        amp: bool = False,
        grad_accum_steps: int = 1,
        grad_clip: float | None = None,
        ddp: bool = False,
    ) -> None:
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError("grad_clip must be > 0 when set")
        self.amp = bool(amp)
        self.grad_accum_steps = int(grad_accum_steps)
        self.grad_clip = grad_clip
        self.ddp = bool(ddp)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0
        self._scaler = None
        self._autocast = None

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    def setup(self) -> None:
        if not self.ddp:
            return
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available")
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_rank = int(os.environ.get("LOCAL_RANK", self._rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            self.device = torch.device("cuda", self._local_rank)
        else:
            self.device = torch.device("cpu")

    def on_fit_start(self, task: Task, optimizer: Optimizer, config: Any) -> None:
        if self.ddp and self._world_size > 1:
            model = task.get_model()
            module = getattr(model, "module", None)
            if module is None:
                raise TypeError("DDP requires a model adapter with a .module attribute")
            if not isinstance(module, torch.nn.parallel.DistributedDataParallel):
                model.module = torch.nn.parallel.DistributedDataParallel(
                    module,
                    device_ids=[self._local_rank] if torch.cuda.is_available() else None,
                )

        self.amp = self.amp and torch.cuda.is_available()
        if self.amp:
            try:
                self._scaler = torch.amp.GradScaler("cuda", enabled=True)
            except AttributeError:  # older torch
                self._scaler = torch.cuda.amp.GradScaler(enabled=True)
            self._autocast = torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            )
        else:
            self._scaler = None
            self._autocast = None

    def forward_context(self):
        return self._autocast or contextlib.nullcontext()

    def scale_loss(self, loss: Any) -> Any:
        return loss / self.grad_accum_steps if self.grad_accum_steps > 1 else loss

    def backward(self, loss: Any) -> None:
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def should_step(self, step: int) -> bool:
        return step % self.grad_accum_steps == 0

    def should_flush(self, step: int) -> bool:
        return step % self.grad_accum_steps != 0

    def optimizer_step(self, task: Task, optimizer: Optimizer) -> None:
        torch_optimizer = getattr(optimizer, "optimizer", None)
        if torch_optimizer is None:
            raise TypeError("TorchTrainingPolicy requires TorchOptimizer")
        if self._scaler is not None:
            self._scaler.unscale_(torch_optimizer)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(task.get_model().parameters(), self.grad_clip)
        if self._scaler is not None:
            self._scaler.step(torch_optimizer)
            self._scaler.update()
        else:
            torch_optimizer.step()

    def zero_grad(self, optimizer: Optimizer) -> None:
        optimizer.zero_grad()

    def on_fit_end(self, task: Task, optimizer: Optimizer) -> None:
        if self.ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "amp": self.amp,
            "grad_accum_steps": self.grad_accum_steps,
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._scaler is not None and state.get("scaler") is not None:
            self._scaler.load_state_dict(state["scaler"])
