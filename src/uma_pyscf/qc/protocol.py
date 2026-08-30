"""Gate 1 protocol, scope, and provenance checks for production labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.elements import atomic_number
from ..schemas._fields import (
    require_bool,
    require_int,
    require_mapping,
    require_sequence,
    require_str,
)
from ..schemas.label_record import LabelRecord

__all__ = [
    "PROTOCOL_CHECK_NAMES",
    "protocol_checks",
]

PROTOCOL_CHECK_NAMES: tuple[str, ...] = (
    "gate1_scope",
    "protocol_identity",
    "protocol_method",
    "runtime_versions",
    "initial_density_provenance",
    "raw_checksum_present",
    "state_registry",
)


def _check(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "observed": observed, "threshold": threshold}


def protocol_checks(
    record: LabelRecord, section: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Run every machine-checkable Conditional GO rule on one record."""
    allowed_symbols = tuple(
        require_str(value, "protocol.allowed_elements[]")
        for value in require_sequence(section.get("allowed_elements"), "protocol.allowed_elements")
    )
    allowed_numbers = {atomic_number(symbol) for symbol in allowed_symbols}
    max_atoms = require_int(section.get("max_atoms"), "protocol.max_atoms")
    outside = sorted(set(record.structure.atomic_numbers) - allowed_numbers)
    scope_observed = {"outside_atomic_numbers": outside, "atom_count": record.structure.atom_count}
    scope_threshold = {"allowed_elements": list(allowed_symbols), "max_atoms": max_atoms}

    expected_protocol_id = require_str(section.get("protocol_id"), "protocol.protocol_id")
    expected_protocol_sha = require_str(section.get("protocol_sha256"), "protocol.protocol_sha256")
    observed_identity = {
        "protocol_id": record.engine.versions.get("protocol_id"),
        "protocol_sha256": record.engine.versions.get("protocol_sha256"),
    }
    identity_threshold = {
        "protocol_id": expected_protocol_id,
        "protocol_sha256": expected_protocol_sha,
    }

    expected_method = require_mapping(section.get("method"), "protocol.method")
    observed_method = record.method.to_dict()
    observed_density = observed_method.pop("density_fit")
    allowed_density = tuple(
        require_bool(value, "protocol.allowed_density_fit[]")
        for value in require_sequence(
            section.get("allowed_density_fit"), "protocol.allowed_density_fit"
        )
    )
    method_passed = observed_method == expected_method and observed_density in allowed_density

    expected_engine = require_str(section.get("engine_name"), "protocol.engine_name")
    required_versions = require_mapping(
        section.get("required_versions"), "protocol.required_versions"
    )
    required_runtime_keys = tuple(
        require_str(value, "protocol.required_runtime_keys[]")
        for value in require_sequence(
            section.get("required_runtime_keys"), "protocol.required_runtime_keys"
        )
    )
    version_observed = {
        key: record.engine.versions.get(key) for key in sorted(required_versions)
    }
    missing_runtime_keys = sorted(
        key for key in required_runtime_keys if key not in record.engine.versions
    )
    versions_passed = record.engine.name == expected_engine and all(
        version_observed[key] == value for key, value in required_versions.items()
    ) and not missing_runtime_keys

    expected_guess = require_str(section.get("initial_density"), "protocol.initial_density")
    expected_location = require_str(
        section.get("initial_density_generated_on"), "protocol.initial_density_generated_on"
    )
    observed_guess = {
        "initial_density": record.state.initial_guess,
        "generated_on": record.engine.versions.get("initial_density_generated_on"),
    }
    initial_passed = (
        record.state.initial_guess == expected_guess
        and record.engine.versions.get("initial_density") == expected_guess
        and record.engine.versions.get("initial_density_generated_on") == expected_location
    )

    raw_required = require_bool(
        section.get("require_raw_checksum_sha256"), "protocol.require_raw_checksum_sha256"
    )
    raw_present = record.raw.checksum_sha256 is not None

    registry_required = require_bool(
        section.get("require_state_registry_for_non_default"),
        "protocol.require_state_registry_for_non_default",
    )
    prefix = require_str(
        section.get("approved_state_provenance_prefix"),
        "protocol.approved_state_provenance_prefix",
    )
    non_default = record.state.charge != 0 or record.state.multiplicity != 1
    state_provenance = record.state.state_provenance
    registry_passed = (
        not registry_required
        or not non_default
        or (state_provenance is not None and state_provenance.startswith(prefix))
    )

    return (
        _check("gate1_scope", not outside and record.structure.atom_count <= max_atoms,
               scope_observed, scope_threshold),
        _check("protocol_identity", observed_identity == identity_threshold,
               observed_identity, identity_threshold),
        _check(
            "protocol_method",
            method_passed,
            record.method.to_dict(),
            dict(expected_method) | {"allowed_density_fit": list(allowed_density)},
        ),
        _check(
            "runtime_versions",
            versions_passed,
            {
                "engine_name": record.engine.name,
                "versions": version_observed,
                "missing_runtime_keys": missing_runtime_keys,
            },
            {
                "engine_name": expected_engine,
                "versions": dict(required_versions),
                "required_runtime_keys": list(required_runtime_keys),
            },
        ),
        _check(
            "initial_density_provenance",
            initial_passed,
            observed_guess,
            {"initial_density": expected_guess, "generated_on": expected_location},
        ),
        _check("raw_checksum_present", raw_present or not raw_required, raw_present, raw_required),
        _check(
            "state_registry",
            registry_passed,
            {"non_default": non_default, "state_provenance": state_provenance},
            {"required_for_non_default": registry_required, "approved_prefix": prefix},
        ),
    )
