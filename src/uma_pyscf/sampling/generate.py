"""Turn a sampling config into a candidate manifest and a geometry QC report.

One config file in, two records out, and nothing else consulted: the same file
regenerates the same two files byte for byte, on any host, on any day. That is
why nothing here reads the clock, the environment, or a global random state --
every random draw is seeded from the config, and the identity of a run is the
config's own content fingerprint rather than the time it happened to be made.

The generator distinguishes two kinds of bad news, and the distinction runs
through the whole module:

* A **rejection** is a scientific verdict about one structure -- atoms too
  close, the molecule fell apart, the geometry is one already generated. The
  candidate is left out of the manifest and written to the QC report with the
  reason and the checks that were run. Generation continues.
* An **error** is a statement that the run cannot be trusted: an unknown config
  key or operation kind, a missing structure file, an element without a
  covalent radius, a charge and multiplicity that cannot describe a state, a
  coordinate that came out non-finite. These raise, because a config that says
  something the generator does not understand may be asking for something
  entirely different from what would be produced.

Candidates are never dropped quietly: the QC report accounts for every one the
config asked for, accepted and rejected alike, and the manifest is the accepted
subset.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from ..core.elements import atomic_number
from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, validate_record_id
from ..core.io import write_json_atomic
from ..core.spin import electron_count, multiplicity_to_spin_2s, validate_electron_spin_parity
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
from ..schemas.candidate import CandidateManifest, CandidateRecord, GeometryQcReport
from ..schemas.label_record import ElectronicState, Structure
from .filters import (
    Fingerprint,
    fragment_count,
    minimum_distance_violation,
    pair_distance_fingerprint,
)
from .geometry import gaussian_displacement, scale_bond
from .siblings import expand_states

__all__ = [
    "DEFAULT_FILTERS",
    "OPERATION_KINDS",
    "SAMPLING_CONFIG_SCHEMA_VERSION",
    "FilterSettings",
    "generate_candidates",
    "load_sampling_config",
    "read_xyz_structure",
    "write_outputs",
]

SAMPLING_CONFIG_SCHEMA_VERSION = 1

#: Filter settings a config need not state. The defaults are the Part I ladder's:
#: 0.65 times the covalent radii sum is the collision cutoff that experiment used
#: for its displaced geometries, and 1.3 times the sum is the usual generous
#: bonding criterion, generous on purpose so that a stretched bond still counts
#: as a bond and a scan is not mistaken for a dissociation.
DEFAULT_FILTERS: dict[str, Any] = {
    "covalent_factor": 0.65,
    "bond_factor": 1.3,
    "allow_fragments": False,
    "duplicate_decimals": 3,
}

_TOP_LEVEL_KEYS = (
    "schema_version",
    "sampling_id",
    "created",
    "derived_from",
    "description",
    "structures",
    "operations",
    "filters",
)
_REQUIRED_TOP_LEVEL_KEYS = ("schema_version", "sampling_id", "structures", "operations")
_SOURCE_STRUCTURE_KEYS = ("id", "xyz_path")
_FILTER_KEYS = tuple(DEFAULT_FILTERS)

#: The operation kinds a v1 config may ask for, with the keys each one takes.
#: A kind that is not listed, or a key that is not listed for its kind, stops
#: the run: a misspelled ``sigma_angstrom`` would otherwise silently fall back
#: to some other amplitude.
OPERATION_KINDS: dict[str, tuple[str, ...]] = {
    "bond_scan": (
        "kind",
        "structure",
        "charge",
        "multiplicity",
        "anchor_index",
        "moved_index",
        "factors",
    ),
    "cartesian_displacement": (
        "kind",
        "structure",
        "charge",
        "multiplicity",
        "sigma_angstrom",
        "count",
        "seed",
    ),
    "state_expansion": ("kind", "structure", "states"),
}

_STATE_KEYS = ("charge", "multiplicity")


@dataclass(frozen=True)
class FilterSettings:
    """The geometry filter thresholds one run was configured with."""

    covalent_factor: float
    bond_factor: float
    allow_fragments: bool
    duplicate_decimals: int


@dataclass(frozen=True)
class _Proposal:
    """One candidate before the filters have had their say."""

    record_id: str
    structure: Structure
    state: ElectronicState
    generation_parameters: dict[str, Any]
    source_structure_id: str
    modifies_geometry: bool


def _format_number(value: int | float) -> str:
    """Return a record-id-safe spelling of a number.

    ``.`` becomes ``p`` and ``-`` becomes ``m`` so the result matches the record
    id pattern, and the digits themselves come from ``repr``, which is injective
    for floats: two different amplitudes can never be spelled the same way and
    collide into one candidate id.
    """
    text = str(value) if isinstance(value, int) else repr(float(value))
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def read_xyz_structure(path: str | Path) -> Structure:
    """Read an XYZ file into a :class:`Structure`, validating it in full.

    The count line has to match the number of atom lines, every symbol has to be
    a real element, and every coordinate has to be a finite number. Anything
    else raises: a seed geometry that is not what the file appears to say would
    poison every candidate derived from it. The comment line is free text and is
    not interpreted.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Structure file {source} cannot be read: {exc}.") from exc
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValidationError(
            f"{source} is not an XYZ file: it needs a count line, a comment line, and "
            f"one line per atom, but it has {len(lines)} line(s)."
        )
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValidationError(
            f"{source} line 1 must be the atom count; got {lines[0]!r}."
        ) from exc
    if count < 1:
        raise ValidationError(f"{source} line 1 declares {count} atoms; at least one is needed.")
    body = lines[2:]
    if len(body) < count:
        raise ValidationError(
            f"{source} declares {count} atoms but carries only {len(body)} atom line(s)."
        )
    for offset, extra in enumerate(body[count:]):
        if extra.strip():
            raise ValidationError(
                f"{source} line {count + 3 + offset} follows the {count} declared atoms: "
                f"{extra.strip()!r}."
            )
    numbers: list[int] = []
    positions: list[tuple[float, float, float]] = []
    for index, line in enumerate(body[:count]):
        fields = line.split()
        if len(fields) != 4:
            raise ValidationError(
                f"{source} line {index + 3} must be a symbol and three coordinates; "
                f"got {line.strip()!r}."
            )
        numbers.append(atomic_number(fields[0]))
        try:
            x, y, z = (float(field) for field in fields[1:])
        except ValueError as exc:
            raise ValidationError(
                f"{source} line {index + 3} has a non-numeric coordinate: {line.strip()!r}."
            ) from exc
        for axis, component in enumerate((x, y, z)):
            if not isfinite(component):
                raise ValidationError(
                    f"{source} line {index + 3} coordinate {axis} is not finite: {line.strip()!r}."
                )
        positions.append((x, y, z))
    return Structure(atomic_numbers=tuple(numbers), positions_angstrom=tuple(positions))


def _validated_state_pair(data: Any, path: str) -> tuple[int, int]:
    """Return the ``(charge, multiplicity)`` pair a config state block states."""
    mapping = require_mapping(data, path)
    reject_unknown_keys(mapping, _STATE_KEYS, path)
    charge = require_int(require_key(mapping, "charge", path), f"{path}.charge")
    multiplicity = require_int(require_key(mapping, "multiplicity", path), f"{path}.multiplicity")
    if multiplicity < 1:
        raise ValidationError(f"{path}.multiplicity must be at least 1; got {multiplicity}.")
    return charge, multiplicity


def _validate_operation(operation: Any, path: str, structure_ids: Sequence[str]) -> None:
    """Check one operation block, naming the offending key when it fails."""
    mapping = require_mapping(operation, path)
    kind = require_str(require_key(mapping, "kind", path), f"{path}.kind")
    if kind not in OPERATION_KINDS:
        raise ValidationError(
            f"{path}.kind {kind!r} is not a known operation; this schema version "
            f"implements {', '.join(repr(name) for name in sorted(OPERATION_KINDS))}."
        )
    reject_unknown_keys(mapping, OPERATION_KINDS[kind], path)
    for key in OPERATION_KINDS[kind]:
        require_key(mapping, key, path)
    structure_id = require_str(mapping["structure"], f"{path}.structure")
    if structure_id not in structure_ids:
        raise ValidationError(
            f"{path}.structure {structure_id!r} is not one of the configured structures "
            f"({', '.join(repr(name) for name in structure_ids)})."
        )
    if kind == "state_expansion":
        states = require_sequence(mapping["states"], f"{path}.states")
        if not states:
            raise ValidationError(f"{path}.states must list at least one state.")
        for index, state in enumerate(states):
            _validated_state_pair(state, f"{path}.states[{index}]")
        return

    multiplicity = require_int(mapping["multiplicity"], f"{path}.multiplicity")
    if multiplicity < 1:
        raise ValidationError(f"{path}.multiplicity must be at least 1; got {multiplicity}.")
    require_int(mapping["charge"], f"{path}.charge")

    if kind == "bond_scan":
        for key in ("anchor_index", "moved_index"):
            index_value = require_int(mapping[key], f"{path}.{key}")
            if index_value < 0:
                raise ValidationError(f"{path}.{key} must not be negative; got {index_value}.")
        factors = require_sequence(mapping["factors"], f"{path}.factors")
        if not factors:
            raise ValidationError(f"{path}.factors must list at least one factor.")
        seen: set[float] = set()
        for index, factor in enumerate(factors):
            value = require_finite_float(factor, f"{path}.factors[{index}]")
            if value <= 0.0:
                raise ValidationError(f"{path}.factors[{index}] must be positive; got {value}.")
            if value in seen:
                raise ValidationError(
                    f"{path}.factors[{index}] repeats the factor {value}; every factor of a "
                    "scan produces one candidate and they would share an id."
                )
            seen.add(value)
        return

    sigma = require_finite_float(mapping["sigma_angstrom"], f"{path}.sigma_angstrom")
    if sigma <= 0.0:
        raise ValidationError(f"{path}.sigma_angstrom must be positive; got {sigma}.")
    count = require_int(mapping["count"], f"{path}.count")
    if count < 1:
        raise ValidationError(f"{path}.count must be at least 1; got {count}.")
    require_int(mapping["seed"], f"{path}.seed")


def load_sampling_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a sampling config, returning it verbatim.

    The file is read with ``yaml.safe_load``, which also accepts JSON, and the
    result is checked to be JSON-safe: a bare YAML date such as
    ``created: 2026-08-22`` is parsed by YAML into a ``date`` object that would
    not survive being embedded in the manifest, so it is refused here with its
    key named (quote it, and it is a string).

    Nothing is filled in and nothing is normalized -- the returned dict is what
    the manifest embeds and what ``config_sha256`` fingerprints -- but every key
    is checked, and an unknown one stops the run rather than being ignored.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Sampling config {source} cannot be read: {exc}.") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"Sampling config {source} is not valid YAML: {exc}.") from exc
    config = validated_json_object(loaded, "config")
    reject_unknown_keys(config, _TOP_LEVEL_KEYS, "config")
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        require_key(config, key, "config")

    version = require_int(config["schema_version"], "config.schema_version")
    if version != SAMPLING_CONFIG_SCHEMA_VERSION:
        raise ValidationError(
            f"config.schema_version must be {SAMPLING_CONFIG_SCHEMA_VERSION}; got {version}."
        )
    validate_record_id(require_str(config["sampling_id"], "config.sampling_id"))

    structures = require_sequence(config["structures"], "config.structures")
    if not structures:
        raise ValidationError("config.structures must list at least one structure.")
    structure_ids: list[str] = []
    for index, entry in enumerate(structures):
        entry_path = f"config.structures[{index}]"
        mapping = require_mapping(entry, entry_path)
        reject_unknown_keys(mapping, _SOURCE_STRUCTURE_KEYS, entry_path)
        structure_id = validate_record_id(
            require_str(require_key(mapping, "id", entry_path), f"{entry_path}.id")
        )
        if structure_id in structure_ids:
            raise ValidationError(
                f"{entry_path}.id {structure_id!r} is already used by an earlier structure."
            )
        require_str(require_key(mapping, "xyz_path", entry_path), f"{entry_path}.xyz_path")
        structure_ids.append(structure_id)

    operations = require_sequence(config["operations"], "config.operations")
    if not operations:
        raise ValidationError("config.operations must list at least one operation.")
    for index, operation in enumerate(operations):
        _validate_operation(operation, f"config.operations[{index}]", structure_ids)

    _filter_settings(config)
    return config


def _filter_settings(config: dict[str, Any]) -> FilterSettings:
    """Return the effective filter thresholds, applying the documented defaults."""
    raw = config.get("filters")
    mapping = require_mapping(raw, "config.filters") if raw is not None else {}
    reject_unknown_keys(mapping, _FILTER_KEYS, "config.filters")
    settings = {**DEFAULT_FILTERS, **mapping}
    covalent_factor = require_finite_float(
        settings["covalent_factor"], "config.filters.covalent_factor"
    )
    bond_factor = require_finite_float(settings["bond_factor"], "config.filters.bond_factor")
    for name, value in (("covalent_factor", covalent_factor), ("bond_factor", bond_factor)):
        if value <= 0.0:
            raise ValidationError(f"config.filters.{name} must be positive; got {value}.")
    decimals = require_int(settings["duplicate_decimals"], "config.filters.duplicate_decimals")
    if decimals < 0:
        raise ValidationError(
            f"config.filters.duplicate_decimals must not be negative; got {decimals}."
        )
    return FilterSettings(
        covalent_factor=covalent_factor,
        bond_factor=bond_factor,
        allow_fragments=require_bool(
            settings["allow_fragments"], "config.filters.allow_fragments"
        ),
        duplicate_decimals=decimals,
    )


def _state_for(
    structure: Structure, charge: int, multiplicity: int, kind: str, path: str
) -> ElectronicState:
    """Return the electronic state an operation asks for, parity checked."""
    try:
        validate_electron_spin_parity(
            electron_count(structure.atomic_numbers, charge), multiplicity
        )
    except ValidationError as exc:
        raise ValidationError(f"{path}: {exc}") from exc
    return ElectronicState(
        charge=charge,
        multiplicity=multiplicity,
        spin_2s=multiplicity_to_spin_2s(multiplicity),
        state_provenance=f"sampling_operation:{kind}",
    )


def _with_provenance(
    structure: Structure, source_id: str, kind: str, seed: int | None
) -> Structure:
    """Return ``structure`` carrying the provenance of the operation that made it."""
    return Structure(
        atomic_numbers=structure.atomic_numbers,
        positions_angstrom=structure.positions_angstrom,
        parent_structure_id=source_id,
        sampling_method=kind,
        random_seed=seed,
    )


def _propose(
    operation: dict[str, Any], seeds: dict[str, Structure], sampling_id: str, path: str
) -> list[_Proposal]:
    """Expand one operation into the candidates it asks for, in config order."""
    kind = str(operation["kind"])
    source_id = str(operation["structure"])
    seed_structure = seeds[source_id]
    verbatim = deepcopy(operation)
    proposals: list[_Proposal] = []

    if kind == "bond_scan":
        anchor = int(operation["anchor_index"])
        moved = int(operation["moved_index"])
        state = _state_for(
            seed_structure, int(operation["charge"]), int(operation["multiplicity"]), kind, path
        )
        for factor in operation["factors"]:
            value = float(factor)
            geometry = scale_bond(seed_structure, anchor, moved, value)
            proposals.append(
                _Proposal(
                    record_id=(
                        f"{sampling_id}_{source_id}_bond{anchor}{moved}_x{_format_number(value)}"
                    ),
                    structure=_with_provenance(geometry, source_id, kind, None),
                    state=state,
                    generation_parameters={"operation": deepcopy(verbatim), "factor": value},
                    source_structure_id=source_id,
                    modifies_geometry=True,
                )
            )
        return proposals

    if kind == "cartesian_displacement":
        sigma = float(operation["sigma_angstrom"])
        base_seed = int(operation["seed"])
        state = _state_for(
            seed_structure, int(operation["charge"]), int(operation["multiplicity"]), kind, path
        )
        for index in range(int(operation["count"])):
            record_seed = base_seed + index
            geometry = gaussian_displacement(seed_structure, sigma, record_seed)
            proposals.append(
                _Proposal(
                    record_id=(
                        f"{sampling_id}_{source_id}_disp{_format_number(sigma)}"
                        f"_s{_format_number(record_seed)}"
                    ),
                    structure=_with_provenance(geometry, source_id, kind, record_seed),
                    state=state,
                    generation_parameters={
                        "operation": deepcopy(verbatim),
                        "seed": record_seed,
                        "index": index,
                    },
                    source_structure_id=source_id,
                    modifies_geometry=True,
                )
            )
        return proposals

    pairs = [
        _validated_state_pair(requested, f"{path}.states[{index}]")
        for index, requested in enumerate(operation["states"])
    ]
    expanded = expand_states(seed_structure, pairs)
    for (charge, multiplicity), state in zip(pairs, expanded, strict=True):
        proposals.append(
            _Proposal(
                record_id=(f"{sampling_id}_{source_id}_q{_format_number(charge)}m{multiplicity}"),
                structure=_with_provenance(seed_structure, source_id, kind, None),
                state=state,
                generation_parameters={
                    "operation": deepcopy(verbatim),
                    "state": {"charge": charge, "multiplicity": multiplicity},
                },
                source_structure_id=source_id,
                modifies_geometry=False,
            )
        )
    return proposals


def _require_finite_geometry(proposal: _Proposal) -> None:
    """Raise when an operation produced a coordinate that is not a finite number.

    This is a check on this package's own arithmetic rather than on user input,
    which is why it is an error and not a rejection: a non-finite coordinate
    means the generator computed something meaningless, and the run stops.
    """
    for index, position in enumerate(proposal.structure.positions_angstrom):
        for axis, component in enumerate(position):
            if not isfinite(component):
                raise ValidationError(
                    f"Candidate {proposal.record_id} has a non-finite coordinate at atom "
                    f"{index} axis {axis}: {component!r}."
                )


def _rejection_reason_for_distance(violation: dict[str, Any], factor: float) -> str:
    """Return the QC reason for a pair of atoms that sit inside the collision cutoff."""
    left, right = violation["atom_indices"]
    left_symbol, right_symbol = violation["symbols"]
    return (
        f"minimum distance {violation['distance_angstrom']:.4f} A between atoms "
        f"{left} ({left_symbol}) and {right} ({right_symbol}) is below the cutoff "
        f"{violation['cutoff_angstrom']:.4f} A (covalent_factor {factor})."
    )


def generate_candidates(config_path: str | Path) -> tuple[CandidateManifest, GeometryQcReport]:
    """Generate every candidate a config asks for and judge each one.

    Returns the manifest of accepted candidates and the QC report covering all
    of them. Neither carries a timestamp, and both name the config by the
    fingerprint of its content, so running this twice on one config produces two
    identical pairs of files.
    """
    source = Path(config_path)
    config = load_sampling_config(source)
    settings = _filter_settings(config)
    sampling_id = str(config["sampling_id"])
    digest = canonical_json_fingerprint(config)

    seeds: dict[str, Structure] = {}
    for entry in config["structures"]:
        xyz_path = (source.parent / str(entry["xyz_path"])).resolve()
        seeds[str(entry["id"])] = read_xyz_structure(xyz_path)

    proposals: list[_Proposal] = []
    for index, operation in enumerate(config["operations"]):
        proposals.extend(
            _propose(dict(operation), seeds, sampling_id, f"config.operations[{index}]")
        )

    seen_ids: set[str] = set()
    for proposal in proposals:
        if proposal.record_id in seen_ids:
            raise ValidationError(
                f"Two operations produce the candidate id {proposal.record_id!r}; a "
                "candidate id names one calculation and the config has to distinguish them."
            )
        seen_ids.add(proposal.record_id)

    seed_fingerprints = {
        structure_id: (
            tuple(sorted(structure.atomic_numbers)),
            pair_distance_fingerprint(structure, settings.duplicate_decimals),
        )
        for structure_id, structure in seeds.items()
    }
    accepted: list[CandidateRecord] = []
    entries: list[dict[str, Any]] = []
    registry: dict[tuple[tuple[int, ...], Fingerprint, int, int], str] = {}

    for proposal in proposals:
        _require_finite_geometry(proposal)
        checks: dict[str, Any] = {"finite_coordinates": True}

        violation = minimum_distance_violation(proposal.structure, settings.covalent_factor)
        checks["minimum_distance"] = {
            "covalent_factor": settings.covalent_factor,
            "violation": deepcopy(violation),
        }
        if violation is not None:
            entries.append(
                {
                    "record_id": proposal.record_id,
                    "status": "rejected",
                    "checks": checks,
                    "reason": _rejection_reason_for_distance(violation, settings.covalent_factor),
                }
            )
            continue

        fragments = fragment_count(proposal.structure, settings.bond_factor)
        checks["fragments"] = {
            "bond_factor": settings.bond_factor,
            "count": fragments,
            "allow_fragments": settings.allow_fragments,
        }
        if fragments > 1 and not settings.allow_fragments:
            entries.append(
                {
                    "record_id": proposal.record_id,
                    "status": "rejected",
                    "checks": checks,
                    "reason": (
                        f"the geometry separates into {fragments} fragments at bond_factor "
                        f"{settings.bond_factor} and allow_fragments is false."
                    ),
                }
            )
            continue

        composition = tuple(sorted(proposal.structure.atomic_numbers))
        fingerprint = pair_distance_fingerprint(proposal.structure, settings.duplicate_decimals)
        duplicate_of: str | None = None
        duplicate_kind: str | None = None
        for structure_id, (seed_composition, seed_fingerprint) in seed_fingerprints.items():
            # A state expansion is meant to keep its seed's geometry, so it is not
            # compared against that seed; every other operation claims to change the
            # geometry, and one that reproduces its source is a calculation already
            # accounted for.
            if not proposal.modifies_geometry and structure_id == proposal.source_structure_id:
                continue
            if (seed_composition, seed_fingerprint) == (composition, fingerprint):
                duplicate_of = structure_id
                duplicate_kind = "source_structure"
                break
        key = (composition, fingerprint, proposal.state.charge, proposal.state.multiplicity)
        if duplicate_of is None and key in registry:
            duplicate_of = registry[key]
            duplicate_kind = "candidate"
        checks["duplicate"] = {
            "decimals": settings.duplicate_decimals,
            "duplicate_of": duplicate_of,
            "duplicate_of_kind": duplicate_kind,
        }
        if duplicate_of is not None:
            noun = "source structure" if duplicate_kind == "source_structure" else "candidate"
            entries.append(
                {
                    "record_id": proposal.record_id,
                    "status": "rejected",
                    "checks": checks,
                    "reason": (
                        f"the geometry duplicates {noun} {duplicate_of!r} to "
                        f"{settings.duplicate_decimals} decimals."
                    ),
                }
            )
            continue

        registry[key] = proposal.record_id
        accepted.append(
            CandidateRecord(
                record_id=proposal.record_id,
                structure=proposal.structure,
                state=proposal.state,
                generation_parameters=proposal.generation_parameters,
            )
        )
        entries.append(
            {
                "record_id": proposal.record_id,
                "status": "accepted",
                "checks": checks,
                "reason": None,
            }
        )

    manifest = CandidateManifest(
        sampling_id=sampling_id,
        config_sha256=digest,
        config=config,
        records=tuple(accepted),
    )
    report = GeometryQcReport(
        sampling_id=sampling_id, config_sha256=digest, entries=tuple(entries)
    )
    return manifest, report


def write_outputs(
    manifest: CandidateManifest, report: GeometryQcReport, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write both records atomically and return their paths.

    The two files are named after the sampling id, so one directory can hold
    several runs, and both are published through
    :func:`~uma_pyscf.core.io.write_json_atomic`: a failed write leaves the
    previous pair in place rather than half of a new one.
    """
    if manifest.sampling_id != report.sampling_id:
        raise ValidationError(
            f"The manifest is for sampling run {manifest.sampling_id!r} but the QC report is "
            f"for {report.sampling_id!r}; they are written as one pair."
        )
    if manifest.config_sha256 != report.config_sha256:
        raise ValidationError(
            f"The manifest and the QC report disagree about the config fingerprint "
            f"({manifest.config_sha256} against {report.config_sha256})."
        )
    directory = Path(output_dir)
    manifest_path = directory / f"{manifest.sampling_id}_candidates.json"
    report_path = directory / f"{report.sampling_id}_geometry_qc.json"
    write_json_atomic(manifest_path, manifest.to_dict())
    write_json_atomic(report_path, report.to_dict())
    return manifest_path, report_path
