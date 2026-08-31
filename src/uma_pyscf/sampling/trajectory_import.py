"""Import deterministically thinned ASE trajectory frames as candidates."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, sha256_of_file, validate_record_id
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
from .filters import fragment_count, minimum_distance_violation, pair_distance_fingerprint
from .generate import FilterSettings, write_outputs

__all__ = [
    "import_trajectory_candidates",
    "load_trajectory_import_config",
    "mass_weighted_arc_length_indices",
    "uniform_frame_indices",
    "write_trajectory_outputs",
]

_TOP_LEVEL_KEYS = (
    "schema_version",
    "sampling_id",
    "created",
    "description",
    "state",
    "trajectories",
    "filters",
)
_STATE_KEYS = ("charge", "multiplicity")
_TRAJECTORY_KEYS = ("trajectory_id", "parent_id", "path", "count", "frame_selection")
_FILTER_KEYS = ("covalent_factor", "bond_factor", "allow_fragments", "duplicate_decimals")
_FRAME_SELECTIONS = ("uniform_index", "mass_weighted_arc_length")
# Standard atomic weights (Da) for the explicitly supported H/Si/Ge/Cl pilot domain.
# Keeping this small table local makes thinning reproducible without importing an
# ASE-version-dependent data table; elements outside the declared domain fail closed.
_ATOMIC_MASSES_DALTON = {1: 1.008, 14: 28.085, 17: 35.45, 32: 72.630}


def uniform_frame_indices(frame_count: int, requested_count: int) -> tuple[int, ...]:
    """Choose deterministic, endpoint-inclusive, uniformly spaced frame indices."""
    if frame_count < 1:
        raise ValidationError("A trajectory must contain at least one frame.")
    if requested_count < 1 or requested_count > frame_count:
        raise ValidationError(
            f"Requested {requested_count} frames from a trajectory containing {frame_count}."
        )
    if requested_count == 1:
        return (frame_count // 2,)
    indices = tuple(
        round(index * (frame_count - 1) / (requested_count - 1))
        for index in range(requested_count)
    )
    if len(set(indices)) != requested_count:
        raise ValidationError("Uniform frame selection unexpectedly produced duplicate indices.")
    return indices


def _validate_requested_count(frame_count: int, requested_count: int) -> None:
    if frame_count < 1:
        raise ValidationError("A trajectory must contain at least one frame.")
    if requested_count < 1 or requested_count > frame_count:
        raise ValidationError(
            f"Requested {requested_count} frames from a trajectory containing {frame_count}."
        )


def _mass_weighted_arc_coordinates(frames: list[Any]) -> tuple[float, ...]:
    _validate_requested_count(len(frames), 1)
    reference_numbers = tuple(int(value) for value in frames[0].numbers)
    try:
        masses = tuple(_ATOMIC_MASSES_DALTON[number] for number in reference_numbers)
    except KeyError as exc:
        raise ValidationError(
            "Mass-weighted thinning supports only H/Si/Ge/Cl; "
            f"trajectory contains atomic number {exc.args[0]}."
        ) from exc
    if not masses or any(not math.isfinite(mass) or mass <= 0 for mass in masses):
        raise ValidationError("Trajectory atomic masses must be finite and positive.")
    total_mass = sum(masses)
    cumulative = [0.0]
    previous = frames[0]
    for frame_index, current in enumerate(frames[1:], start=1):
        numbers = tuple(int(value) for value in current.numbers)
        if numbers != reference_numbers:
            raise ValidationError(
                f"Trajectory frame {frame_index} changes atom identity or ordering."
            )
        if len(previous.positions) != len(masses) or len(current.positions) != len(masses):
            raise ValidationError(f"Trajectory frame {frame_index} has the wrong atom count.")
        squared = 0.0
        for mass, before, after in zip(
            masses, previous.positions, current.positions, strict=True
        ):
            displacement_squared = sum(
                (float(after[axis]) - float(before[axis])) ** 2 for axis in range(3)
            )
            squared += mass * displacement_squared
        step = math.sqrt(squared / total_mass)
        if not math.isfinite(step):
            raise ValidationError(
                f"Trajectory frame {frame_index} has non-finite Cartesian displacement."
            )
        cumulative.append(cumulative[-1] + step)
        previous = current
    return tuple(cumulative)


def _coordinate_indices(coordinates: tuple[float, ...], requested_count: int) -> tuple[int, ...]:
    frame_count = len(coordinates)
    _validate_requested_count(frame_count, requested_count)
    if requested_count == 1:
        target = coordinates[-1] / 2.0
        chosen = min(
            range(frame_count),
            key=lambda index: (abs(coordinates[index] - target), index),
        )
        return (chosen,)
    if coordinates[-1] == 0.0:
        return uniform_frame_indices(frame_count, requested_count)
    indices = [0]
    for target_index in range(1, requested_count - 1):
        target = coordinates[-1] * target_index / (requested_count - 1)
        lower = indices[-1] + 1
        upper = frame_count - (requested_count - target_index)
        chosen = min(
            range(lower, upper + 1),
            key=lambda index: (abs(coordinates[index] - target), index),
        )
        indices.append(chosen)
    indices.append(frame_count - 1)
    return tuple(indices)


def mass_weighted_arc_length_indices(
    frames: list[Any], requested_count: int
) -> tuple[int, ...]:
    """Choose endpoint-inclusive frames uniformly in mass-weighted Cartesian arc length."""
    return _coordinate_indices(_mass_weighted_arc_coordinates(frames), requested_count)


def load_trajectory_import_config(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate a trajectory import YAML file."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"Trajectory import config {source} cannot be read: {exc}.") from exc
    config = validated_json_object(raw, "trajectory_import_config")
    reject_unknown_keys(config, _TOP_LEVEL_KEYS, "trajectory_import_config")
    version = require_int(
        require_key(config, "schema_version", "trajectory_import_config"),
        "trajectory_import_config.schema_version",
    )
    if version != 1:
        raise ValidationError("trajectory_import_config.schema_version must be 1.")
    validate_record_id(
        require_str(
            require_key(config, "sampling_id", "trajectory_import_config"),
            "trajectory_import_config.sampling_id",
        )
    )
    state = require_mapping(
        require_key(config, "state", "trajectory_import_config"),
        "trajectory_import_config.state",
    )
    reject_unknown_keys(state, _STATE_KEYS, "trajectory_import_config.state")
    require_int(
        require_key(state, "charge", "trajectory_import_config.state"),
        "trajectory_import_config.state.charge",
    )
    multiplicity = require_int(
        require_key(state, "multiplicity", "trajectory_import_config.state"),
        "trajectory_import_config.state.multiplicity",
    )
    if multiplicity < 1:
        raise ValidationError("trajectory_import_config.state.multiplicity must be positive.")
    trajectories = require_sequence(
        require_key(config, "trajectories", "trajectory_import_config"),
        "trajectory_import_config.trajectories",
    )
    if not trajectories:
        raise ValidationError("trajectory_import_config.trajectories must not be empty.")
    seen: set[str] = set()
    for index, item in enumerate(trajectories):
        item_path = f"trajectory_import_config.trajectories[{index}]"
        entry = require_mapping(item, item_path)
        reject_unknown_keys(entry, _TRAJECTORY_KEYS, item_path)
        trajectory_id = validate_record_id(
            require_str(
                require_key(entry, "trajectory_id", item_path), f"{item_path}.trajectory_id"
            )
        )
        if trajectory_id in seen:
            raise ValidationError(f"{item_path}.trajectory_id repeats {trajectory_id!r}.")
        seen.add(trajectory_id)
        validate_record_id(
            require_str(require_key(entry, "parent_id", item_path), f"{item_path}.parent_id")
        )
        require_str(require_key(entry, "path", item_path), f"{item_path}.path")
        count = require_int(require_key(entry, "count", item_path), f"{item_path}.count")
        if count < 1:
            raise ValidationError(f"{item_path}.count must be positive.")
        selection = require_str(
            entry.get("frame_selection", "uniform_index"), f"{item_path}.frame_selection"
        )
        if selection not in _FRAME_SELECTIONS:
            raise ValidationError(
                f"{item_path}.frame_selection must be one of {_FRAME_SELECTIONS!r}; "
                f"got {selection!r}."
            )
    filters = require_mapping(
        require_key(config, "filters", "trajectory_import_config"),
        "trajectory_import_config.filters",
    )
    reject_unknown_keys(filters, _FILTER_KEYS, "trajectory_import_config.filters")
    for key in ("covalent_factor", "bond_factor"):
        value = require_finite_float(
            require_key(filters, key, "trajectory_import_config"),
            f"trajectory_import_config.filters.{key}",
        )
        if value <= 0:
            raise ValidationError(f"trajectory_import_config.filters.{key} must be positive.")
    require_bool(
        require_key(filters, "allow_fragments", "trajectory_import_config"),
        "trajectory_import_config.filters.allow_fragments",
    )
    decimals = require_int(
        require_key(filters, "duplicate_decimals", "trajectory_import_config"),
        "trajectory_import_config.filters.duplicate_decimals",
    )
    if decimals < 0:
        raise ValidationError(
            "trajectory_import_config.filters.duplicate_decimals must not be negative."
        )
    return config


def _read_trajectory(path: Path) -> list[Any]:
    try:
        from ase.io import read  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValidationError(
            "ASE is required for trajectory import; install the dataset optional dependency."
        ) from exc
    try:
        frames = read(path, index=":")
    except Exception as exc:
        raise ValidationError(f"ASE could not read trajectory {path}: {exc}.") from exc
    return list(frames)


def _settings(config: dict[str, Any]) -> FilterSettings:
    raw = config["filters"]
    return FilterSettings(
        covalent_factor=float(raw["covalent_factor"]),
        bond_factor=float(raw["bond_factor"]),
        allow_fragments=bool(raw["allow_fragments"]),
        duplicate_decimals=int(raw["duplicate_decimals"]),
    )


def import_trajectory_candidates(
    config_path: str | Path, source_root: str | Path
) -> tuple[CandidateManifest, GeometryQcReport]:
    """Import selected frames, apply geometry QC, and retain trajectory provenance."""
    config = load_trajectory_import_config(config_path)
    sampling_id = str(config["sampling_id"])
    charge = int(config["state"]["charge"])
    multiplicity = int(config["state"]["multiplicity"])
    settings = _settings(config)
    resolved = deepcopy(config)
    resolved_sources: list[dict[str, Any]] = []
    proposals: list[CandidateRecord] = []

    for item in config["trajectories"]:
        trajectory_id = str(item["trajectory_id"])
        parent_id = str(item["parent_id"])
        relative_path = Path(str(item["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValidationError(
                f"Trajectory path {relative_path} must be relative and may not contain '..'."
            )
        source = Path(source_root) / relative_path
        frames = _read_trajectory(source)
        frame_selection = str(item.get("frame_selection", "uniform_index"))
        if frame_selection == "mass_weighted_arc_length":
            arc_coordinates = _mass_weighted_arc_coordinates(frames)
            indices = _coordinate_indices(arc_coordinates, int(item["count"]))
            total_mass_weighted_arc_length = arc_coordinates[-1]
        else:
            indices = uniform_frame_indices(len(frames), int(item["count"]))
            total_mass_weighted_arc_length = None
        source_hash = sha256_of_file(source)
        resolved_sources.append(
            {
                "trajectory_id": trajectory_id,
                "path": relative_path.as_posix(),
                "sha256": source_hash,
                "frame_count": len(frames),
                "frame_selection": frame_selection,
                "selected_frame_indices": list(indices),
                "total_mass_weighted_arc_length_angstrom": total_mass_weighted_arc_length,
            }
        )
        for frame_index in indices:
            atoms = frames[frame_index]
            if any(bool(value) for value in atoms.pbc):
                raise ValidationError(
                    f"{trajectory_id} frame {frame_index} is periodic; "
                    "this pilot is molecular only."
                )
            structure = Structure(
                atomic_numbers=tuple(int(value) for value in atoms.numbers),
                positions_angstrom=tuple(
                    (float(row[0]), float(row[1]), float(row[2])) for row in atoms.positions
                ),
                parent_structure_id=parent_id,
                sampling_method=f"ase_trajectory_{frame_selection}",
            )
            validate_electron_spin_parity(
                electron_count(structure.atomic_numbers, charge), multiplicity
            )
            state = ElectronicState(
                charge=charge,
                multiplicity=multiplicity,
                spin_2s=multiplicity_to_spin_2s(multiplicity),
                state_provenance="trajectory_import_config",
            )
            proposals.append(
                CandidateRecord(
                    record_id=f"{sampling_id}_{trajectory_id}_f{frame_index:05d}",
                    structure=structure,
                    state=state,
                    generation_parameters={
                        "trajectory_id": trajectory_id,
                        "frame_index": frame_index,
                        "frame_selection": frame_selection,
                        "source_path": relative_path.as_posix(),
                        "source_sha256": source_hash,
                    },
                )
            )

    resolved["resolved_sources"] = resolved_sources
    digest = canonical_json_fingerprint(resolved)
    accepted: list[CandidateRecord] = []
    entries: list[dict[str, Any]] = []
    seen_geometries: dict[tuple[Any, ...], str] = {}
    for record in proposals:
        structure = record.structure
        checks: dict[str, Any] = {"finite_coordinates": True}
        violation = minimum_distance_violation(structure, settings.covalent_factor)
        checks["minimum_distance"] = {
            "covalent_factor": settings.covalent_factor,
            "violation": violation,
        }
        fragments = fragment_count(structure, settings.bond_factor)
        checks["fragments"] = {
            "bond_factor": settings.bond_factor,
            "count": fragments,
            "allow_fragments": settings.allow_fragments,
        }
        key = (
            tuple(sorted(structure.atomic_numbers)),
            pair_distance_fingerprint(structure, settings.duplicate_decimals),
            record.state.charge,
            record.state.multiplicity,
        )
        duplicate_of = seen_geometries.get(key)
        checks["duplicate"] = {
            "decimals": settings.duplicate_decimals,
            "duplicate_of": duplicate_of,
        }
        reason: str | None = None
        if violation is not None:
            reason = "minimum-distance collision"
        elif fragments > 1 and not settings.allow_fragments:
            reason = f"geometry has {fragments} fragments and allow_fragments is false"
        elif duplicate_of is not None:
            reason = f"geometry duplicates candidate {duplicate_of!r}"
        if reason is None:
            seen_geometries[key] = record.record_id
            accepted.append(record)
            status = "accepted"
        else:
            status = "rejected"
        entries.append(
            {
                "record_id": record.record_id,
                "status": status,
                "checks": checks,
                "reason": reason,
            }
        )

    return (
        CandidateManifest(
            sampling_id=sampling_id,
            config_sha256=digest,
            config=resolved,
            records=tuple(accepted),
        ),
        GeometryQcReport(sampling_id=sampling_id, config_sha256=digest, entries=tuple(entries)),
    )


def write_trajectory_outputs(
    manifest: CandidateManifest, report: GeometryQcReport, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write imported candidates with the same atomic writer as normal sampling."""
    return write_outputs(manifest, report, output_dir)
