"""Seed and capture Python / NumPy / Torch RNG state.

The core has no hard dependency on NumPy or Torch. Call ``seed_everything``
**before** building data and models; ``Runner.fit`` calls it again as a safety
net for training-time stochasticity.

The seed is part of the run identity: one run = one resolved config x one seed.
Use ``deterministic=True`` for deterministic cuDNN kernels (slower, and some ops
may still warn).

A checkpoint stores the state returned by ``capture_rng_state`` and restores it
with ``restore_rng_state``. The representation uses only lists, tuples and
scalars so the default JSON checkpoint codec works without NumPy or Torch.
"""

from __future__ import annotations

import random
from typing import Any


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG that is importable; never require an optional package."""

    seed = int(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, TypeError):
                pass
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    """Return a JSON-friendly snapshot of all available RNGs."""

    state: dict[str, Any] = {"python": random.getstate()}

    try:
        import numpy as np

        name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
        state["numpy"] = {
            "name": name,
            "keys": keys.tolist(),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }
    except ImportError:
        pass

    try:
        import torch

        state["torch"] = torch.get_rng_state().tolist()
        if torch.cuda.is_available():
            state["cuda"] = [
                device_state.tolist()
                for device_state in torch.cuda.get_rng_state_all()
            ]
    except ImportError:
        pass

    return state


def restore_rng_state(state: Any) -> None:
    """Restore a snapshot from ``capture_rng_state``.

    A legacy Python-only ``random.getstate()`` tuple/list is accepted for
    checkpoints written before multi-RNG capture existed.
    """

    if not state:
        return

    if isinstance(state, dict):
        python_state = state.get("python")
        if python_state is not None:
            random.setstate(_python_state(python_state))

        numpy_state = state.get("numpy")
        if numpy_state is not None:
            _restore_numpy(numpy_state)

        torch_state = state.get("torch")
        if torch_state is not None:
            _restore_torch(torch_state)

        cuda_states = state.get("cuda")
        if cuda_states:
            _restore_cuda(cuda_states)
        return

    # Legacy checkpoints stored the Python RNG state directly.
    random.setstate(_python_state(state))


def _python_state(value: Any) -> tuple[int, tuple[int, ...], Any]:
    version, internal, gaussian = value
    return int(version), tuple(int(item) for item in internal), gaussian


def _restore_numpy(value: Any) -> None:
    try:
        import numpy as np
    except ImportError:
        return

    if isinstance(value, dict):
        name = str(value["name"])
        keys = value["keys"]
        pos = int(value["pos"])
        has_gauss = int(value["has_gauss"])
        cached_gaussian = float(value["cached_gaussian"])
    else:
        name, keys, pos, has_gauss, cached_gaussian = value

    np.random.set_state(
        (
            name,
            np.asarray(keys, dtype=np.uint32),
            pos,
            has_gauss,
            cached_gaussian,
        )
    )


def _torch_byte_tensor(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().to(torch.uint8)
    return torch.tensor(value, dtype=torch.uint8)


def _restore_torch(value: Any) -> None:
    try:
        import torch
    except ImportError:
        return

    torch.set_rng_state(_torch_byte_tensor(value))


def _restore_cuda(states: Any) -> None:
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    for index, value in enumerate(states):
        if index >= torch.cuda.device_count():
            break
        torch.cuda.set_rng_state(_torch_byte_tensor(value), device=index)
