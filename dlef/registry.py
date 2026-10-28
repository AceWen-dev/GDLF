"""Typed component registry: name/class_path -> implementation.

The registry is assembly infrastructure, not a runtime contract. It does not
know what the components do; it only knows how to resolve and instantiate them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any, TypeVar

ComponentT = TypeVar("ComponentT")


class RegistryError(LookupError):
    """Duplicate registration, unknown name, or invalid class_path."""


class Registry:
    """Thread-safe registry keyed by (kind, name).

    ``kind`` is a slot name chosen by the adapter, e.g. "model", "loss",
    "metric", "optimizer", "scheduler", "tracker", "launcher".
    """

    def __init__(self) -> None:
        self._components: dict[str, dict[str, type[Any]]] = {}
        self._lock = RLock()

    def register(
        self,
        kind: str,
        name: str,
        *,
        aliases: tuple[str, ...] = (),
    ) -> Callable[[type[ComponentT]], type[ComponentT]]:
        keys = (name, *aliases)

        def decorator(component: type[ComponentT]) -> type[ComponentT]:
            with self._lock:
                implementations = self._components.setdefault(_normalize(kind), {})
                for key in keys:
                    normalized = _normalize(key)
                    existing = implementations.get(normalized)
                    if existing is not None and existing is not component:
                        raise RegistryError(f"duplicate registration: {kind}/{key}")
                    implementations[normalized] = component
            return component

        return decorator

    def resolve(self, kind: str, config: Mapping[str, Any] | str) -> type[Any]:
        if isinstance(config, str):
            name = config
            class_path = None
        else:
            name = str(config.get("name", ""))
            class_path = config.get("class_path")

        if class_path:
            return _load_class(str(class_path))
        if not name:
            raise RegistryError(f"{kind}: config requires 'name' or 'class_path'")

        with self._lock:
            implementations = self._components.get(_normalize(kind), {})
            component = implementations.get(_normalize(name))
            if component is None:
                available = ", ".join(sorted(implementations)) or "none"
                raise RegistryError(f"unknown {kind} implementation {name!r}; available: {available}")
            return component

    def create(
        self,
        kind: str,
        config: Mapping[str, Any],
        *,
        expected_type: type[ComponentT] | None = None,
    ) -> ComponentT:
        component_cls = self.resolve(kind, config)
        options = {
            key: value
            for key, value in config.items()
            if key not in {"name", "class_path", "enabled"}
        }
        from_config = getattr(component_cls, "from_config", None)
        component = from_config(options) if callable(from_config) else component_cls(**options)
        if expected_type is not None and not isinstance(component, expected_type):
            raise TypeError(
                f"{kind}: {component_cls.__name__} does not implement {expected_type.__name__}"
            )
        # Protocol-based contracts are checked structurally; this is a light
        # safety net for cases where expected_type is a runtime_checkable Protocol.
        return component

    def available(self, kind: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._components.get(_normalize(kind), {})))


def _load_class(path: str) -> type[Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, separator, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise RegistryError("class_path must be 'package.module:ClassName'")
    component = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(component, type):
        raise RegistryError(f"class_path does not resolve to a class: {path}")
    return component


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_")
