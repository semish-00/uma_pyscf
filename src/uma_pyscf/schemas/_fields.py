"""Field coercion and validation helpers shared by the schema dataclasses.

Every helper names the offending field with a dotted path (``results.s2``,
``structure.positions_angstrom[2]``) so a rejected record says where it is
wrong, and every failure raises :class:`~uma_pyscf.core.errors.ValidationError`.
Booleans are never accepted where an integer is expected, and no float field
accepts ``NaN`` or an infinity: a label that cannot be compared numerically is
not a label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Any

from ..core.elements import MAX_ATOMIC_NUMBER
from ..core.errors import ValidationError

__all__ = [
    "optional_finite_float",
    "optional_int",
    "optional_str",
    "reject_unknown_keys",
    "require_atomic_numbers",
    "require_bool",
    "require_finite_float",
    "require_int",
    "require_key",
    "require_mapping",
    "require_sequence",
    "require_str",
    "require_vector_rows",
    "validated_json_object",
]


def require_int(value: object, path: str) -> int:
    """Return ``value`` as an ``int``, rejecting booleans and every other type."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{path} must be an integer; got {value!r}.")
    return value


def optional_int(value: object, path: str) -> int | None:
    """Return ``value`` as an ``int``, or ``None`` when it is absent."""
    return None if value is None else require_int(value, path)


def require_bool(value: object, path: str) -> bool:
    """Return ``value`` as a ``bool``, rejecting ``0``/``1`` and truthy stand-ins."""
    if not isinstance(value, bool):
        raise ValidationError(f"{path} must be true or false; got {value!r}.")
    return value


def require_str(value: object, path: str) -> str:
    """Return ``value`` as a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a non-empty string; got {value!r}.")
    return value


def optional_str(value: object, path: str) -> str | None:
    """Return ``value`` as a non-empty string, or ``None`` when it is absent."""
    return None if value is None else require_str(value, path)


def require_finite_float(value: object, path: str) -> float:
    """Return ``value`` as a finite ``float``, rejecting ``NaN`` and infinities."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{path} must be a number; got {value!r}.")
    number = float(value)
    if not isfinite(number):
        raise ValidationError(f"{path} must be finite; got {value!r}.")
    return number


def optional_finite_float(value: object, path: str) -> float | None:
    """Return ``value`` as a finite ``float``, or ``None`` when it is absent."""
    return None if value is None else require_finite_float(value, path)


def require_sequence(value: object, path: str) -> tuple[Any, ...]:
    """Return ``value`` as a tuple, accepting lists and tuples but not strings."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValidationError(f"{path} must be a list; got {value!r}.")
    return tuple(value)


def require_mapping(value: object, path: str) -> dict[str, Any]:
    """Return ``value`` as a plain dict, requiring string keys."""
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object; got {value!r}.")
    for key in value:
        if not isinstance(key, str):
            raise ValidationError(f"{path} has a non-string key {key!r}.")
    return dict(value)


def require_key(data: Mapping[str, Any], key: str, path: str) -> Any:
    """Return ``data[key]``, naming the full path when the key is missing."""
    if key not in data:
        raise ValidationError(f"{path}.{key} is required and is missing.")
    return data[key]


def reject_unknown_keys(data: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    """Raise when ``data`` carries a key this schema version does not define.

    Records fail closed: an unrecognized key is either a typo or a newer schema
    version, and silently dropping it would lose scientific content.
    """
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValidationError(
            f"{path} has unknown key(s) {', '.join(repr(key) for key in unknown)}; "
            f"allowed keys are {', '.join(repr(key) for key in allowed)}."
        )


def require_atomic_numbers(value: object, path: str) -> tuple[int, ...]:
    """Return ``value`` as a non-empty tuple of atomic numbers within the table."""
    numbers = require_sequence(value, path)
    if not numbers:
        raise ValidationError(f"{path} must list at least one atom.")
    checked: list[int] = []
    for index, number in enumerate(numbers):
        atomic = require_int(number, f"{path}[{index}]")
        if not 1 <= atomic <= MAX_ATOMIC_NUMBER:
            raise ValidationError(
                f"{path}[{index}] must be an atomic number from 1 through "
                f"{MAX_ATOMIC_NUMBER}; got {atomic}."
            )
        checked.append(atomic)
    return tuple(checked)


def require_vector_rows(value: object, path: str) -> tuple[tuple[float, float, float], ...]:
    """Return ``value`` as rows of three finite Cartesian components."""
    rows = require_sequence(value, path)
    checked: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows):
        components = require_sequence(row, f"{path}[{index}]")
        if len(components) != 3:
            raise ValidationError(
                f"{path}[{index}] must have three components; got {len(components)}."
            )
        x, y, z = (
            require_finite_float(component, f"{path}[{index}][{axis}]")
            for axis, component in enumerate(components)
        )
        checked.append((x, y, z))
    return tuple(checked)


def validated_json_object(value: object, path: str) -> dict[str, Any]:
    """Return a deep copy of ``value`` after checking it is a JSON object.

    Provenance blocks (engine versions, QC history entries) accept free-form
    content, but it still has to survive a write-read round trip, so nested
    values are restricted to JSON types and floats must be finite. The copy
    keeps a record independent of the caller's dict after construction.
    """
    mapping = require_mapping(value, path)
    for key, item in mapping.items():
        _check_json_value(item, f"{path}.{key}")
    return deepcopy(mapping)


def _check_json_value(value: object, path: str) -> None:
    """Raise unless ``value`` is a JSON scalar, list, or object with finite floats."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        require_finite_float(value, path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path} has a non-string key {key!r}.")
            _check_json_value(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        for index, item in enumerate(value):
            _check_json_value(item, f"{path}[{index}]")
        return
    raise ValidationError(f"{path} must be a JSON value; got {type(value).__name__}.")
