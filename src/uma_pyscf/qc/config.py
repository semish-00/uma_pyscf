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

from ..core.elements import atomic_number
from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ..schemas._fields import (
    reject_unknown_keys,
    require_bool,
    require_finite_float,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    validated_json_object,
)
from ..schemas.label_record import Method

__all__ = [
    "ELECTRONIC_KEYS",
    "GEOMETRY_KEYS",
    "QC_CONFIG_SCHEMA_VERSION",
    "PROTOCOL_KEYS",
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
    "release_status",
    "protocol",
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

PROTOCOL_KEYS: tuple[str, ...] = (
    "protocol_id",
    "protocol_sha256",
    "engine_name",
    "required_versions",
    "required_runtime_keys",
    "allowed_elements",
    "max_atoms",
    "method",
    "allowed_density_fit",
    "initial_density",
    "initial_density_generated_on",
    "require_raw_checksum_sha256",
    "require_state_registry_for_non_default",
    "approved_state_provenance_prefix",
)
_PROTOCOL_METHOD_KEYS = (
    "functional",
    "basis",
    "ecp",
    "aux_basis",
    "grid_level",
    "nlc_grid_level",
    "grid_response",
    "scf_conv_tol",
    "scf_max_cycle",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _validate_protocol_section(value: object) -> None:
    """Validate the optional production-protocol QC section."""
    protocol = require_mapping(value, "config.protocol")
    reject_unknown_keys(protocol, PROTOCOL_KEYS, "config.protocol")
    for key in PROTOCOL_KEYS:
        require_key(protocol, key, "config.protocol")
    validate_record_id(require_str(protocol["protocol_id"], "config.protocol.protocol_id"))
    digest = require_str(protocol["protocol_sha256"], "config.protocol.protocol_sha256")
    if len(digest) != 64 or not set(digest.lower()) <= _HEX_DIGITS:
        raise ValidationError("config.protocol.protocol_sha256 must be 64 hexadecimal characters.")
    require_str(protocol["engine_name"], "config.protocol.engine_name")
    versions = require_mapping(protocol["required_versions"], "config.protocol.required_versions")
    if not versions:
        raise ValidationError("config.protocol.required_versions must not be empty.")
    for name, version in versions.items():
        require_str(name, "config.protocol.required_versions key")
        require_str(version, f"config.protocol.required_versions.{name}")
    runtime_keys = require_sequence(
        protocol["required_runtime_keys"], "config.protocol.required_runtime_keys"
    )
    if not runtime_keys:
        raise ValidationError("config.protocol.required_runtime_keys must not be empty.")
    checked_runtime_keys = [
        require_str(value, f"config.protocol.required_runtime_keys[{index}]")
        for index, value in enumerate(runtime_keys)
    ]
    if len(set(checked_runtime_keys)) != len(checked_runtime_keys):
        raise ValidationError("config.protocol.required_runtime_keys must not repeat values.")
    symbols = require_sequence(protocol["allowed_elements"], "config.protocol.allowed_elements")
    if not symbols:
        raise ValidationError("config.protocol.allowed_elements must not be empty.")
    seen_symbols: set[str] = set()
    for index, symbol in enumerate(symbols):
        name = require_str(symbol, f"config.protocol.allowed_elements[{index}]")
        atomic_number(name)
        if name in seen_symbols:
            raise ValidationError(f"config.protocol.allowed_elements repeats {name!r}.")
        seen_symbols.add(name)
    max_atoms = require_int(protocol["max_atoms"], "config.protocol.max_atoms")
    if max_atoms < 1:
        raise ValidationError("config.protocol.max_atoms must be at least 1.")
    method = require_mapping(protocol["method"], "config.protocol.method")
    reject_unknown_keys(method, _PROTOCOL_METHOD_KEYS, "config.protocol.method")
    for key in _PROTOCOL_METHOD_KEYS:
        require_key(method, key, "config.protocol.method")
    # Reuse the canonical Method validation by adding the separately stated
    # density-fitting value only for validation.
    Method.from_dict(dict(method) | {"density_fit": True})
    density_values = require_sequence(
        protocol["allowed_density_fit"], "config.protocol.allowed_density_fit"
    )
    if not density_values:
        raise ValidationError("config.protocol.allowed_density_fit must not be empty.")
    checked_density = [
        require_bool(value, f"config.protocol.allowed_density_fit[{index}]")
        for index, value in enumerate(density_values)
    ]
    if len(set(checked_density)) != len(checked_density):
        raise ValidationError("config.protocol.allowed_density_fit must not repeat values.")
    require_str(protocol["initial_density"], "config.protocol.initial_density")
    require_str(
        protocol["initial_density_generated_on"],
        "config.protocol.initial_density_generated_on",
    )
    for key in ("require_raw_checksum_sha256", "require_state_registry_for_non_default"):
        require_bool(protocol[key], f"config.protocol.{key}")
    require_str(
        protocol["approved_state_provenance_prefix"],
        "config.protocol.approved_state_provenance_prefix",
    )


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
    if "release_status" in config:
        require_str(config["release_status"], "config.release_status")
    if "protocol" in config:
        if "release_status" not in config:
            raise ValidationError(
                "config.release_status is required when config.protocol is present."
            )
        _validate_protocol_section(config["protocol"])

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
