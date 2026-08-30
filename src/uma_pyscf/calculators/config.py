"""Versioned DFT protocol loading and fail-closed Conditional GO rules.

The production config is the executable form of Gate 1 decision 0003.  This
module validates every field needed to reproduce a calculation, rejects scope
expansion before PySCF is imported, and turns the method block into the
canonical :class:`~uma_pyscf.schemas.label_record.Method` record.

No scientific value is defaulted here.  A missing key stops the run, because a
plausible default would be an unreviewed change to the frozen protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import yaml

from ..core.elements import atomic_number
from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ..schemas._fields import (
    reject_unknown_keys,
    require_bool,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    validated_json_object,
)
from ..schemas.candidate import CandidateRecord
from ..schemas.label_record import Method

__all__ = [
    "DFT_CONFIG_SCHEMA_VERSION",
    "load_dft_config",
    "method_from_config",
    "resource_for_candidate",
    "scope_violations",
    "validate_dft_config",
]

DFT_CONFIG_SCHEMA_VERSION = 1

_TOP_LEVEL_KEYS = (
    "schema_version",
    "protocol_id",
    "created",
    "derived_from",
    "decision",
    "description",
    "engine",
    "method",
    "initial_density",
    "scope",
    "resources",
    "retry",
    "provenance",
    "release_controls",
)
_REQUIRED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS
_ENGINE_KEYS = (
    "name",
    "required_versions",
    "required_gpu_name",
    "container_image",
    "container_sha256_file",
    "python_overlay",
    "python_lock_file",
)
_METHOD_KEYS = (
    "functional",
    "basis",
    "ecp",
    "aux_basis",
    "grid_level",
    "nlc_grid_level",
    "grid_response",
    "density_fit",
    "scf_conv_tol",
    "scf_max_cycle",
)
_INITIAL_DENSITY_KEYS = ("guess", "generated_on", "pass_explicit_dm0")
_SCOPE_KEYS = (
    "allowed_elements",
    "max_atoms",
    "periodic",
    "calculation",
    "blocked_state_provenance",
    "require_state_registry_for_non_default",
    "approved_state_provenance_prefix",
)
_RESOURCE_KEYS = ("gpu_count", "candidates_per_process", "tiers")
_RESOURCE_TIER_KEYS = ("max_atoms", "ncpus", "max_memory_mb", "walltime")
_RETRY_KEYS = ("stop_on_first_failure", "attempts")
_ATTEMPT_KEYS = ("id", "density_fit", "retry_on")
_PROVENANCE_KEYS = (
    "required_runtime_keys",
    "require_protocol_sha256",
    "require_input_fingerprint_sha256",
    "require_raw_checksum_sha256",
)
_RELEASE_KEYS = (
    "thresholds_status",
    "composition_baseline_required",
    "state_registry_required_for_non_default",
    "release_allowed",
)
_SUPPORTED_FAILURE_CATEGORIES = frozenset(
    {"scf_not_converged", "scf_root_mismatch", "spin_contamination", "gradient_outlier"}
)
_WALLTIME_PATTERN = re.compile(r"\d{2,3}:[0-5]\d:[0-5]\d")


def _positive_int(value: object, path: str) -> int:
    number = require_int(value, path)
    if number < 1:
        raise ValidationError(f"{path} must be at least 1; got {number}.")
    return number


def _string_list(value: object, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    items = require_sequence(value, path)
    if not items and not allow_empty:
        raise ValidationError(f"{path} must list at least one value.")
    strings = tuple(require_str(item, f"{path}[{index}]") for index, item in enumerate(items))
    if len(set(strings)) != len(strings):
        raise ValidationError(f"{path} must not contain duplicate values; got {strings!r}.")
    return strings


def _validate_engine(value: object) -> None:
    engine = require_mapping(value, "config.engine")
    reject_unknown_keys(engine, _ENGINE_KEYS, "config.engine")
    for key in _ENGINE_KEYS:
        require_key(engine, key, "config.engine")
    if require_str(engine["name"], "config.engine.name") != "gpu4pyscf":
        raise ValidationError("config.engine.name must be 'gpu4pyscf' in schema version 1.")
    versions = require_mapping(engine["required_versions"], "config.engine.required_versions")
    required_names = ("python", "pyscf", "gpu4pyscf", "cupy", "cutensor")
    reject_unknown_keys(versions, required_names, "config.engine.required_versions")
    for name in required_names:
        require_str(require_key(versions, name, "config.engine.required_versions"),
                    f"config.engine.required_versions.{name}")
    for key in _ENGINE_KEYS[2:]:
        require_str(engine[key], f"config.engine.{key}")


def _validate_method(value: object) -> None:
    method = require_mapping(value, "config.method")
    reject_unknown_keys(method, _METHOD_KEYS, "config.method")
    for key in _METHOD_KEYS:
        require_key(method, key, "config.method")
    Method.from_dict(method)


def _validate_initial_density(value: object) -> None:
    density = require_mapping(value, "config.initial_density")
    reject_unknown_keys(density, _INITIAL_DENSITY_KEYS, "config.initial_density")
    for key in _INITIAL_DENSITY_KEYS:
        require_key(density, key, "config.initial_density")
    if require_str(density["guess"], "config.initial_density.guess") != "minao":
        raise ValidationError("config.initial_density.guess must be 'minao' for protocol v1.")
    if (
        require_str(density["generated_on"], "config.initial_density.generated_on")
        != "cpu_before_device_conversion"
    ):
        raise ValidationError(
            "config.initial_density.generated_on must be 'cpu_before_device_conversion'."
        )
    if not require_bool(density["pass_explicit_dm0"], "config.initial_density.pass_explicit_dm0"):
        raise ValidationError("config.initial_density.pass_explicit_dm0 must be true.")


def _validate_scope(value: object) -> None:
    scope = require_mapping(value, "config.scope")
    reject_unknown_keys(scope, _SCOPE_KEYS, "config.scope")
    for key in _SCOPE_KEYS:
        require_key(scope, key, "config.scope")
    symbols = _string_list(scope["allowed_elements"], "config.scope.allowed_elements")
    for index, symbol in enumerate(symbols):
        atomic_number(symbol)
        if symbol not in {"H", "Si", "Ge", "Cl"}:
            raise ValidationError(
                f"config.scope.allowed_elements[{index}] expands Gate 1 to {symbol!r}."
            )
    _positive_int(scope["max_atoms"], "config.scope.max_atoms")
    if scope["max_atoms"] > 8:
        raise ValidationError("config.scope.max_atoms cannot exceed the Gate 1 limit of 8.")
    if require_bool(scope["periodic"], "config.scope.periodic"):
        raise ValidationError(
            "config.scope.periodic must be false; periodic systems are out of scope."
        )
    if require_str(scope["calculation"], "config.scope.calculation") != "energy_gradient":
        raise ValidationError("config.scope.calculation must be 'energy_gradient'.")
    _string_list(
        scope["blocked_state_provenance"],
        "config.scope.blocked_state_provenance",
        allow_empty=True,
    )
    require_bool(
        scope["require_state_registry_for_non_default"],
        "config.scope.require_state_registry_for_non_default",
    )
    require_str(
        scope["approved_state_provenance_prefix"],
        "config.scope.approved_state_provenance_prefix",
    )


def _validate_resources(value: object, max_atoms: int) -> None:
    resources = require_mapping(value, "config.resources")
    reject_unknown_keys(resources, _RESOURCE_KEYS, "config.resources")
    for key in _RESOURCE_KEYS:
        require_key(resources, key, "config.resources")
    if _positive_int(resources["gpu_count"], "config.resources.gpu_count") != 1:
        raise ValidationError("config.resources.gpu_count must be 1 for protocol v1.")
    if (
        _positive_int(
            resources["candidates_per_process"], "config.resources.candidates_per_process"
        )
        != 1
    ):
        raise ValidationError(
            "config.resources.candidates_per_process must be 1 to preserve process isolation."
        )
    tiers = require_sequence(resources["tiers"], "config.resources.tiers")
    if not tiers:
        raise ValidationError("config.resources.tiers must list at least one tier.")
    previous = 0
    for index, value in enumerate(tiers):
        path = f"config.resources.tiers[{index}]"
        tier = require_mapping(value, path)
        reject_unknown_keys(tier, _RESOURCE_TIER_KEYS, path)
        for key in _RESOURCE_TIER_KEYS:
            require_key(tier, key, path)
        tier_atoms = _positive_int(tier["max_atoms"], f"{path}.max_atoms")
        if tier_atoms <= previous:
            raise ValidationError(f"{path}.max_atoms must be strictly increasing.")
        previous = tier_atoms
        _positive_int(tier["ncpus"], f"{path}.ncpus")
        _positive_int(tier["max_memory_mb"], f"{path}.max_memory_mb")
        walltime = require_str(tier["walltime"], f"{path}.walltime")
        if not _WALLTIME_PATTERN.fullmatch(walltime):
            raise ValidationError(f"{path}.walltime must have HH:MM:SS form; got {walltime!r}.")
    if previous < max_atoms:
        raise ValidationError(
            "config.resources.tiers do not cover config.scope.max_atoms "
            f"({previous} < {max_atoms})."
        )


def _validate_retry(value: object, primary_density_fit: bool) -> None:
    retry = require_mapping(value, "config.retry")
    reject_unknown_keys(retry, _RETRY_KEYS, "config.retry")
    for key in _RETRY_KEYS:
        require_key(retry, key, "config.retry")
    require_bool(retry["stop_on_first_failure"], "config.retry.stop_on_first_failure")
    attempts = require_sequence(retry["attempts"], "config.retry.attempts")
    if len(attempts) != 2:
        raise ValidationError("config.retry.attempts must contain primary and direct fallback.")
    ids: list[str] = []
    densities: list[bool] = []
    for index, value in enumerate(attempts):
        path = f"config.retry.attempts[{index}]"
        attempt = require_mapping(value, path)
        reject_unknown_keys(attempt, _ATTEMPT_KEYS, path)
        for key in _ATTEMPT_KEYS:
            require_key(attempt, key, path)
        identifier = validate_record_id(require_str(attempt["id"], f"{path}.id"))
        ids.append(identifier)
        densities.append(require_bool(attempt["density_fit"], f"{path}.density_fit"))
        categories = _string_list(attempt["retry_on"], f"{path}.retry_on", allow_empty=True)
        unsupported = sorted(set(categories) - _SUPPORTED_FAILURE_CATEGORIES)
        if unsupported:
            raise ValidationError(f"{path}.retry_on has unsupported categories {unsupported!r}.")
    if len(set(ids)) != len(ids):
        raise ValidationError("config.retry.attempts ids must be unique.")
    if densities != [primary_density_fit, False]:
        raise ValidationError(
            "config.retry.attempts must run the frozen density-fitting primary first and "
            "the direct fallback second."
        )


def _validate_provenance(value: object) -> None:
    provenance = require_mapping(value, "config.provenance")
    reject_unknown_keys(provenance, _PROVENANCE_KEYS, "config.provenance")
    for key in _PROVENANCE_KEYS:
        require_key(provenance, key, "config.provenance")
    runtime_keys = _string_list(
        provenance["required_runtime_keys"], "config.provenance.required_runtime_keys"
    )
    for required in ("pyscf", "gpu4pyscf", "cuda_device_name"):
        if required not in runtime_keys:
            raise ValidationError(
                f"config.provenance.required_runtime_keys must include {required!r}."
            )
    for key in _PROVENANCE_KEYS[1:]:
        if not require_bool(provenance[key], f"config.provenance.{key}"):
            raise ValidationError(f"config.provenance.{key} must be true.")


def _validate_release_controls(value: object) -> None:
    controls = require_mapping(value, "config.release_controls")
    reject_unknown_keys(controls, _RELEASE_KEYS, "config.release_controls")
    for key in _RELEASE_KEYS:
        require_key(controls, key, "config.release_controls")
    require_str(controls["thresholds_status"], "config.release_controls.thresholds_status")
    for key in ("composition_baseline_required", "state_registry_required_for_non_default"):
        if not require_bool(controls[key], f"config.release_controls.{key}"):
            raise ValidationError(f"config.release_controls.{key} must be true.")
    if require_bool(controls["release_allowed"], "config.release_controls.release_allowed"):
        raise ValidationError(
            "config.release_controls.release_allowed must remain false until scientific "
            "thresholds and the state registry are approved."
        )


def validate_dft_config(data: Any) -> dict[str, Any]:
    """Return a deep-copied, strictly validated DFT protocol config."""
    config = validated_json_object(data, "config")
    reject_unknown_keys(config, _TOP_LEVEL_KEYS, "config")
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        require_key(config, key, "config")
    version = require_int(config["schema_version"], "config.schema_version")
    if version != DFT_CONFIG_SCHEMA_VERSION:
        raise ValidationError(
            f"config.schema_version must be {DFT_CONFIG_SCHEMA_VERSION}; got {version}."
        )
    validate_record_id(require_str(config["protocol_id"], "config.protocol_id"))
    require_str(config["created"], "config.created")
    require_str(config["derived_from"], "config.derived_from")
    if require_str(config["decision"], "config.decision") != "conditional_go":
        raise ValidationError("config.decision must be 'conditional_go'.")
    require_str(config["description"], "config.description")
    _validate_engine(config["engine"])
    _validate_method(config["method"])
    _validate_initial_density(config["initial_density"])
    _validate_scope(config["scope"])
    scope = require_mapping(config["scope"], "config.scope")
    _validate_resources(
        config["resources"], require_int(scope["max_atoms"], "config.scope.max_atoms")
    )
    method = require_mapping(config["method"], "config.method")
    _validate_retry(
        config["retry"], require_bool(method["density_fit"], "config.method.density_fit")
    )
    _validate_provenance(config["provenance"])
    _validate_release_controls(config["release_controls"])
    return config


def load_dft_config(path: str | Path) -> dict[str, Any]:
    """Read YAML/JSON at ``path`` and validate it as the production protocol."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"DFT config {source} cannot be read: {exc}.") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"DFT config {source} is not valid YAML: {exc}.") from exc
    return validate_dft_config(loaded)


def method_from_config(config: Mapping[str, Any], *, density_fit: bool | None = None) -> Method:
    """Return the canonical method block, optionally selecting the direct fallback."""
    method = require_mapping(require_key(config, "method", "config"), "config.method")
    values = dict(method)
    if density_fit is not None:
        values["density_fit"] = density_fit
    return Method.from_dict(values)


def scope_violations(candidate: CandidateRecord, config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every Conditional GO scope violation for ``candidate``."""
    if not isinstance(candidate, CandidateRecord):
        raise ValidationError(
            f"candidate must be a CandidateRecord; got {type(candidate).__name__}."
        )
    scope = require_mapping(require_key(config, "scope", "config"), "config.scope")
    allowed = {atomic_number(symbol) for symbol in _string_list(scope["allowed_elements"],
                                                               "config.scope.allowed_elements")}
    violations: list[str] = []
    outside = sorted(set(candidate.structure.atomic_numbers) - allowed)
    if outside:
        violations.append(f"atomic_numbers_outside_gate1:{','.join(map(str, outside))}")
    max_atoms = require_int(scope["max_atoms"], "config.scope.max_atoms")
    if candidate.structure.atom_count > max_atoms:
        violations.append(
            f"atom_count_exceeds_gate1:{candidate.structure.atom_count}>{max_atoms}"
        )
    provenance = candidate.state.state_provenance
    blocked = set(
        _string_list(
            scope["blocked_state_provenance"],
            "config.scope.blocked_state_provenance",
            allow_empty=True,
        )
    )
    if provenance in blocked:
        violations.append(f"state_provenance_blocked:{provenance}")
    require_registry = require_bool(
        scope["require_state_registry_for_non_default"],
        "config.scope.require_state_registry_for_non_default",
    )
    is_non_default = candidate.state.charge != 0 or candidate.state.multiplicity != 1
    if require_registry and is_non_default:
        prefix = require_str(
            scope["approved_state_provenance_prefix"],
            "config.scope.approved_state_provenance_prefix",
        )
        if provenance is None or not provenance.startswith(prefix):
            violations.append("non_default_state_missing_approved_registry_provenance")
    return tuple(violations)


def resource_for_candidate(
    candidate: CandidateRecord, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the first resource tier that covers ``candidate``."""
    resources = require_mapping(require_key(config, "resources", "config"), "config.resources")
    for raw_tier in require_sequence(resources["tiers"], "config.resources.tiers"):
        tier = require_mapping(raw_tier, "config.resources.tiers[]")
        if candidate.structure.atom_count <= require_int(tier["max_atoms"], "tier.max_atoms"):
            return dict(tier) | {"gpu_count": resources["gpu_count"]}
    raise ValidationError(
        f"No resource tier covers {candidate.structure.atom_count} atoms for "
        f"{candidate.record_id}."
    )
