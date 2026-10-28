"""Typed configuration: dataclass schema, strict loading, resolved dump, digest.

Design choices (confirmed):
- dataclass-first, no pydantic/hydra dependency;
- unknown keys are errors;
- the resolved config is dumped and hashed before the first step;
- YAML is optional (JSON always works); adapters can add more loaders.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    """Raised when a config is missing, malformed, or contains unknown keys."""


class ConfigSchema:
    """Base class for typed config trees.

    Subclasses should be dataclasses. ``validate`` is called after construction.
    """

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfigSchema":
        return _build_dataclass(cls, data, path=cls.__name__)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Subclasses may override for range/cross-field checks."""


@dataclass
class TrainConfig(ConfigSchema):
    """Generic training knobs owned by the core, not by any domain."""

    seed: int = 0
    epochs: int = 1
    max_steps: int | None = None
    project: str = "default"
    experiment: str = "default"
    log_every: int = 1
    checkpoint_every: int = 0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ConfigError("epochs must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ConfigError("max_steps must be >= 1 when set")
        if self.log_every < 0:
            raise ConfigError("log_every must be >= 0")
        if self.checkpoint_every < 0:
            raise ConfigError("checkpoint_every must be >= 0")


@dataclass(frozen=True)
class Config:
    raw: Mapping[str, Any]
    schema: Any
    resolved: Mapping[str, Any]
    digest: str
    source: str = ""

    def dump(self) -> str:
        return json.dumps(self.resolved, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings; ``override`` wins for non-mappings."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, ABCMapping) and isinstance(merged.get(key), ABCMapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def resolve_config(
    schema_cls: type[ConfigSchema],
    raw: Mapping[str, Any],
    overrides: Sequence[Mapping[str, Any]] = (),
    *,
    source: str = "",
) -> Config:
    """Merge raw config + overrides, build the schema, validate, hash."""

    if not isinstance(raw, ABCMapping):
        raise ConfigError("config root must be a mapping")
    merged: dict[str, Any] = dict(raw)
    for override in overrides:
        if not isinstance(override, ABCMapping):
            raise ConfigError("config override must be a mapping")
        merged = deep_merge(merged, override)

    schema = schema_cls.from_dict(merged)
    schema.validate()
    resolved = schema.to_dict()
    return Config(raw=merged, schema=schema, resolved=resolved, digest=config_digest(resolved), source=source)


def load_config(
    path: str | Path,
    schema_cls: type[ConfigSchema],
    overrides: Sequence[Mapping[str, Any]] = (),
) -> Config:
    """Load JSON (always) or YAML (if PyYAML is installed)."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "YAML config requires PyYAML; install it or use a JSON config"
            ) from exc
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    if not isinstance(raw, ABCMapping):
        raise ConfigError(f"config root must be a mapping: {path}")
    return resolve_config(schema_cls, raw, overrides, source=str(path))


# ---------------------------------------------------------------------------
# Generic dataclass builder
# ---------------------------------------------------------------------------


def _build_dataclass(cls: type, data: Any, *, path: str) -> Any:
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{path}: {cls!r} is not a dataclass")
    if not isinstance(data, ABCMapping):
        raise ConfigError(f"{path}: expected mapping, got {type(data).__name__}")

    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {', '.join(unknown)}")

    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for name, item in fields.items():
        if name in data:
            values[name] = _coerce(data[name], hints.get(name, Any), path=f"{path}.{name}")
        elif item.default is not dataclasses.MISSING:
            values[name] = item.default
        elif item.default_factory is not dataclasses.MISSING:
            values[name] = item.default_factory()
        else:
            raise ConfigError(f"{path}: missing required key {name!r}")

    try:
        instance = cls(**values)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    validate = getattr(instance, "validate", None)
    if callable(validate):
        validate()
    return instance


def _coerce(value: Any, hint: Any, *, path: str) -> Any:
    if hint is Any or hint is object:
        return value

    origin = get_origin(hint)
    args = get_args(hint)

    # Optional / Union
    if origin in (types.UnionType, Union):
        non_none = tuple(item for item in args if item is not type(None))
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed")
        if len(non_none) == 1:
            return _coerce(value, non_none[0], path=path)
        return value

    # Mapping / dict
    if origin in (ABCMapping, dict):
        if not isinstance(value, ABCMapping):
            raise ConfigError(f"{path}: expected mapping, got {type(value).__name__}")
        return dict(value)

    # list / Sequence
    if origin in (list, tuple, ABCSequence):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected list/tuple, got {type(value).__name__}")
        if args:
            item_hint = args[0]
            return [_coerce(item, item_hint, path=f"{path}[]") for item in value]
        return list(value)

    # Nested dataclass
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return _build_dataclass(hint, value, path=path)

    # Scalars
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected float, got {type(value).__name__}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {type(value).__name__}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {type(value).__name__}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {type(value).__name__}")
        return value

    # Unknown/unsupported type: pass through; adapter validates more if needed.
    return value
