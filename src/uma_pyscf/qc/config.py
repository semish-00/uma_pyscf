"""The QC config: where every threshold a verdict depends on is written down.

A QC threshold is a scientific condition, so it lives in a versioned file under
``configs/datasets/`` and never in this package (structure design section 4).
This module is the entrance for that file: it reads it, checks every key, and
hands back the mapping *verbatim* so the report can embed exactly what the run
was told and fingerprint it.

Nothing is defaulted. The sampling config can afford defaults for its geometry
filters because a filter it does not mention still produces candidates a human
reviews; a QC config that silently supplied, say, an ``s2_max_abs_deviation``
would publish a dataset whose spin-contamination tolerance nobody chose. So
every threshold is required, an unknown key stops the run with its path named,
and a threshold that is not a positive number is refused rather than coerced.

The three field helpers here -- :func:`positive_threshold`, :func:`flag`,
:func:`non_negative_int` -- are used twice on purpose: once by
:func:`validate_qc_config` when the file is read, and again by the individual
checks in :mod:`uma_pyscf.qc.electronic` and :mod:`uma_pyscf.qc.geometry` when
they take a threshold out of a section. That is what makes a check fail closed
even when it is called directly with a hand-built section instead of a loaded
config: a check that cannot find its threshold raises, and never quietly passes
the record it was asked to judge.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ..schemas._fields import (
    reject_unknown_keys,
    require_bool,
    require_finite_float,
    require_int,
    require_key,
    require_mapping,
    require_str,
    validated_json_object,
)

__all__ = [
    "ELECTRONIC_KEYS",
    "GEOMETRY_KEYS",
    "QC_CONFIG_SCHEMA_VERSION",
    "flag",
    "load_qc_config",
    "non_negative_int",
    "positive_threshold",
    "validate_qc_config",
]

QC_CONFIG_SCHEMA_VERSION = 1

#: The electronic-structure section's keys, all of them required.
ELECTRONIC_KEYS: tuple[str, ...] = (
    "require_converged",
    "require_s2_for_open_shell",
    "s2_max_abs_deviation",
    "gradient_max_abs_hartree_per_bohr",
    "gradient_norm_max_hartree_per_bohr",
)

#: The geometry section's keys, all of them required.
GEOMETRY_KEYS: tuple[str, ...] = (
    "covalent_factor",
    "bond_factor",
    "allow_fragments",
    "duplicate_decimals",
)

_TOP_LEVEL_KEYS = (
    "schema_version",
    "qc_id",
    "created",
    "derived_from",
    "description",
    "electronic",
    "geometry",
)
_REQUIRED_TOP_LEVEL_KEYS = ("schema_version", "qc_id", "electronic", "geometry")

_ELECTRONIC_FLAGS = ("require_converged", "require_s2_for_open_shell")
_ELECTRONIC_THRESHOLDS = (
    "s2_max_abs_deviation",
    "gradient_max_abs_hartree_per_bohr",
    "gradient_norm_max_hartree_per_bohr",
)
_GEOMETRY_THRESHOLDS = ("covalent_factor", "bond_factor")


def positive_threshold(section: Mapping[str, Any], key: str, path: str) -> float:
    """Return ``section[key]`` as a positive float, naming its full path on failure.

    A missing threshold is an error and not a skipped check: the run was asked
    to judge records against a condition that is not stated, and guessing one
    would put an unreviewed number into a released dataset.
    """
    value = require_finite_float(require_key(section, key, path), f"{path}.{key}")
    if value <= 0.0:
        raise ValidationError(f"{path}.{key} must be positive; got {value}.")
    return value


def flag(section: Mapping[str, Any], key: str, path: str) -> bool:
    """Return ``section[key]`` as a boolean, rejecting ``0``/``1`` stand-ins."""
    return require_bool(require_key(section, key, path), f"{path}.{key}")


def non_negative_int(section: Mapping[str, Any], key: str, path: str) -> int:
    """Return ``section[key]`` as an integer of zero or more."""
    value = require_int(require_key(section, key, path), f"{path}.{key}")
    if value < 0:
        raise ValidationError(f"{path}.{key} must not be negative; got {value}.")
    return value


def validate_qc_config(data: Any) -> dict[str, Any]:
    """Return ``data`` as a validated QC config, unchanged in content.

    The mapping is checked to be JSON-safe first, because it is embedded in the
    report and fingerprinted: a bare YAML date such as ``created: 2026-08-22``
    parses into a ``date`` object that would not survive the round trip, so it
    is refused here with its key named (quote it, and it is a string).

    Every section is then checked key by key. Nothing is filled in and nothing
    is normalized -- what comes back is what the report embeds and what
    ``config_sha256`` covers.
    """
    config = validated_json_object(data, "config")
    reject_unknown_keys(config, _TOP_LEVEL_KEYS, "config")
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        require_key(config, key, "config")

    version = require_int(config["schema_version"], "config.schema_version")
    if version != QC_CONFIG_SCHEMA_VERSION:
        raise ValidationError(
            f"config.schema_version must be {QC_CONFIG_SCHEMA_VERSION}; got {version}."
        )
    validate_record_id(require_str(config["qc_id"], "config.qc_id"))

    electronic = require_mapping(config["electronic"], "config.electronic")
    reject_unknown_keys(electronic, ELECTRONIC_KEYS, "config.electronic")
    for key in _ELECTRONIC_FLAGS:
        flag(electronic, key, "config.electronic")
    for key in _ELECTRONIC_THRESHOLDS:
        positive_threshold(electronic, key, "config.electronic")
    if not electronic["require_converged"]:
        raise ValidationError(
            "config.electronic.require_converged must be true in schema version 1. The "
            "flag is stated so the report records that convergence was required; there "
            "is no supported way to accept an unconverged label, and a config that asks "
            "for one is refused rather than honoured."
        )

    geometry = require_mapping(config["geometry"], "config.geometry")
    reject_unknown_keys(geometry, GEOMETRY_KEYS, "config.geometry")
    for key in _GEOMETRY_THRESHOLDS:
        positive_threshold(geometry, key, "config.geometry")
    flag(geometry, "allow_fragments", "config.geometry")
    non_negative_int(geometry, "duplicate_decimals", "config.geometry")
    return config


def load_qc_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a QC config file, returning it verbatim.

    The file is read with ``yaml.safe_load``, which also accepts JSON. An
    unreadable file, malformed YAML, and an invalid config are all reported as
    :class:`~uma_pyscf.core.errors.ValidationError` naming the file or the
    offending key, so a caller has one exception type to handle.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"QC config {source} cannot be read: {exc}.") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"QC config {source} is not valid YAML: {exc}.") from exc
    return validate_qc_config(loaded)
