"""The canonical label record: one converged DFT label, in native units.

Schema ``uma-pyscf-label-record-v1``. A record keeps the units the calculation
produced -- energy in hartree, gradient in hartree/bohr, positions in angstrom
-- and names the unit in every key, so no reader has to guess. Two consequences
of that rule are enforced here rather than trusted:

* A record stores the **gradient** and nothing else. ``forces_*`` fields are
  refused outright, because the gradient-to-force sign inversion happens once,
  in the datasets export layer, on the way to fairchem's eV and eV/angstrom.
* The spin multiplicity ``2S+1`` is the source of truth. ``spin_2s`` is stored
  as a convenience for the PySCF side and must equal ``multiplicity - 1``;
  :meth:`LabelRecord.from_dict` re-derives it and rejects a mismatch instead of
  believing the stored value.

Every block is a frozen dataclass with an explicit ``to_dict``/``from_dict``
pair, so the on-disk shape is visible in the source and no third-party schema
library is needed. The constructors normalize sequences into tuples and
validate unconditionally: an object of one of these types is always a valid
block. ``from_dict`` additionally fails closed on unknown keys, because an
unrecognized key is either a typo or a newer schema version, and silently
dropping it would lose scientific content.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ..core.spin import electron_count, multiplicity_to_spin_2s, validate_electron_spin_parity
from ._fields import (
    optional_finite_float,
    optional_int,
    optional_str,
    reject_unknown_keys,
    require_atomic_numbers,
    require_bool,
    require_finite_float,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    require_vector_rows,
    validated_json_object,
)

__all__ = [
    "CANONICAL_UNITS",
    "LABEL_RECORD_SCHEMA",
    "QC_STATUSES",
    "ElectronicState",
    "Engine",
    "LabelRecord",
    "Method",
    "QcState",
    "RawArtifact",
    "Results",
    "Structure",
]

LABEL_RECORD_SCHEMA = "uma-pyscf-label-record-v1"

#: The only unit mapping a v1 record may carry. Deviations are rejected rather
#: than converted, so a record that means something else cannot be read as if
#: it meant this.
CANONICAL_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "energy": "hartree",
        "gradient": "hartree/bohr",
        "positions": "angstrom",
    }
)

QC_STATUSES: tuple[str, ...] = ("pending", "accepted", "rejected")

_STRUCTURE_KEYS = (
    "atomic_numbers",
    "positions_angstrom",
    "parent_structure_id",
    "sampling_method",
    "random_seed",
)
_STATE_KEYS = ("charge", "multiplicity", "spin_2s", "initial_guess", "state_provenance")
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
_ENGINE_KEYS = ("name", "versions")
_RESULTS_KEYS = (
    "energy_hartree",
    "gradient_hartree_per_bohr",
    "converged",
    "n_iterations",
    "s2",
    "s2_target",
    "s2_deviation",
    "wall_time_seconds",
    "scf_wall_time_seconds",
    "gradient_wall_time_seconds",
)
_RAW_KEYS = ("logical_location", "checksum_sha256")
_QC_KEYS = ("status", "history")
_RECORD_KEYS = (
    "schema",
    "record_id",
    "structure",
    "state",
    "method",
    "engine",
    "results",
    "raw",
    "qc",
    "units",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _reject_force_keys(data: Mapping[str, Any], path: str) -> None:
    """Raise on any key that names a force instead of a gradient."""
    for key in data:
        if "force" in str(key).lower():
            raise ValidationError(
                f"{path}.{key} is not allowed: a canonical label record stores the "
                "gradient in hartree/bohr only. The gradient-to-force sign inversion "
                "happens exactly once, in the datasets export layer that writes "
                "fairchem units, and never in a record."
            )


@dataclass(frozen=True, kw_only=True)
class Structure:
    """Atoms and Cartesian geometry, plus where the geometry came from.

    Atom order is the index: ``atomic_numbers[i]`` and ``positions_angstrom[i]``
    describe the same atom, and gradient rows follow the same order.
    """

    atomic_numbers: tuple[int, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    parent_structure_id: str | None = None
    sampling_method: str | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        numbers = require_atomic_numbers(self.atomic_numbers, "structure.atomic_numbers")
        positions = require_vector_rows(self.positions_angstrom, "structure.positions_angstrom")
        if len(positions) != len(numbers):
            raise ValidationError(
                f"structure.positions_angstrom has {len(positions)} rows but "
                f"structure.atomic_numbers lists {len(numbers)} atoms."
            )
        parent = optional_str(self.parent_structure_id, "structure.parent_structure_id")
        if parent is not None:
            validate_record_id(parent)
        seed = optional_int(self.random_seed, "structure.random_seed")
        if seed is not None and seed < 0:
            raise ValidationError(f"structure.random_seed must not be negative; got {seed}.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "parent_structure_id", parent)
        object.__setattr__(
            self,
            "sampling_method",
            optional_str(self.sampling_method, "structure.sampling_method"),
        )
        object.__setattr__(self, "random_seed", seed)

    @property
    def atom_count(self) -> int:
        """Number of atoms in the structure."""
        return len(self.atomic_numbers)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this block."""
        return {
            "atomic_numbers": list(self.atomic_numbers),
            "positions_angstrom": [list(row) for row in self.positions_angstrom],
            "parent_structure_id": self.parent_structure_id,
            "sampling_method": self.sampling_method,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Structure:
        """Build a :class:`Structure` from ``data``, rejecting unknown keys."""
        mapping = require_mapping(data, "structure")
        reject_unknown_keys(mapping, _STRUCTURE_KEYS, "structure")
        return cls(
            atomic_numbers=require_key(mapping, "atomic_numbers", "structure"),
            positions_angstrom=require_key(mapping, "positions_angstrom", "structure"),
            parent_structure_id=mapping.get("parent_structure_id"),
            sampling_method=mapping.get("sampling_method"),
            random_seed=mapping.get("random_seed"),
        )


@dataclass(frozen=True, kw_only=True)
class ElectronicState:
    """Total charge and spin state. Neither field has a default anywhere."""

    charge: int
    multiplicity: int
    spin_2s: int
    initial_guess: str | None = None
    state_provenance: str | None = None

    def __post_init__(self) -> None:
        charge = require_int(self.charge, "state.charge")
        multiplicity = require_int(self.multiplicity, "state.multiplicity")
        derived = multiplicity_to_spin_2s(multiplicity)
        stored = require_int(self.spin_2s, "state.spin_2s")
        if stored != derived:
            raise ValidationError(
                f"state.spin_2s is {stored} but multiplicity {multiplicity} derives "
                f"{derived}. The multiplicity is the source of truth; spin_2s is only "
                "carried along for the PySCF side."
            )
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "multiplicity", multiplicity)
        object.__setattr__(self, "spin_2s", stored)
        object.__setattr__(
            self, "initial_guess", optional_str(self.initial_guess, "state.initial_guess")
        )
        object.__setattr__(
            self,
            "state_provenance",
            optional_str(self.state_provenance, "state.state_provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this block."""
        return {
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "spin_2s": self.spin_2s,
            "initial_guess": self.initial_guess,
            "state_provenance": self.state_provenance,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ElectronicState:
        """Build an :class:`ElectronicState`, re-deriving ``spin_2s`` on the way."""
        mapping = require_mapping(data, "state")
        reject_unknown_keys(mapping, _STATE_KEYS, "state")
        return cls(
            charge=require_key(mapping, "charge", "state"),
            multiplicity=require_key(mapping, "multiplicity", "state"),
            spin_2s=require_key(mapping, "spin_2s", "state"),
            initial_guess=mapping.get("initial_guess"),
            state_provenance=mapping.get("state_provenance"),
        )


@dataclass(frozen=True, kw_only=True)
class Method:
    """The DFT protocol the label was produced with, in full."""

    functional: str
    basis: str
    ecp: str | None
    aux_basis: str | None
    grid_level: int
    nlc_grid_level: int
    grid_response: bool
    density_fit: bool
    scf_conv_tol: float
    scf_max_cycle: int

    def __post_init__(self) -> None:
        grid_level = require_int(self.grid_level, "method.grid_level")
        nlc_grid_level = require_int(self.nlc_grid_level, "method.nlc_grid_level")
        for name, level in (("grid_level", grid_level), ("nlc_grid_level", nlc_grid_level)):
            if level < 0:
                raise ValidationError(f"method.{name} must not be negative; got {level}.")
        conv_tol = require_finite_float(self.scf_conv_tol, "method.scf_conv_tol")
        if conv_tol <= 0:
            raise ValidationError(f"method.scf_conv_tol must be positive; got {conv_tol}.")
        max_cycle = require_int(self.scf_max_cycle, "method.scf_max_cycle")
        if max_cycle < 1:
            raise ValidationError(f"method.scf_max_cycle must be at least 1; got {max_cycle}.")
        object.__setattr__(self, "functional", require_str(self.functional, "method.functional"))
        object.__setattr__(self, "basis", require_str(self.basis, "method.basis"))
        object.__setattr__(self, "ecp", optional_str(self.ecp, "method.ecp"))
        object.__setattr__(self, "aux_basis", optional_str(self.aux_basis, "method.aux_basis"))
        object.__setattr__(self, "grid_level", grid_level)
        object.__setattr__(self, "nlc_grid_level", nlc_grid_level)
        object.__setattr__(
            self, "grid_response", require_bool(self.grid_response, "method.grid_response")
        )
        object.__setattr__(
            self, "density_fit", require_bool(self.density_fit, "method.density_fit")
        )
        object.__setattr__(self, "scf_conv_tol", conv_tol)
        object.__setattr__(self, "scf_max_cycle", max_cycle)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this block."""
        return {
            "functional": self.functional,
            "basis": self.basis,
            "ecp": self.ecp,
            "aux_basis": self.aux_basis,
            "grid_level": self.grid_level,
            "nlc_grid_level": self.nlc_grid_level,
            "grid_response": self.grid_response,
            "density_fit": self.density_fit,
            "scf_conv_tol": self.scf_conv_tol,
            "scf_max_cycle": self.scf_max_cycle,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Method:
        """Build a :class:`Method` from ``data``, rejecting unknown keys."""
        mapping = require_mapping(data, "method")
        reject_unknown_keys(mapping, _METHOD_KEYS, "method")
        return cls(
            functional=require_key(mapping, "functional", "method"),
            basis=require_key(mapping, "basis", "method"),
            ecp=mapping.get("ecp"),
            aux_basis=mapping.get("aux_basis"),
            grid_level=require_key(mapping, "grid_level", "method"),
            nlc_grid_level=require_key(mapping, "nlc_grid_level", "method"),
            grid_response=require_key(mapping, "grid_response", "method"),
            density_fit=require_key(mapping, "density_fit", "method"),
            scf_conv_tol=require_key(mapping, "scf_conv_tol", "method"),
            scf_max_cycle=require_key(mapping, "scf_max_cycle", "method"),
        )


@dataclass(frozen=True, kw_only=True)
class Engine:
    """Which program produced the label, and the versions of everything in it.

    ``versions`` is provenance, so its key set is open: whatever the calculation
    host could report about pyscf, gpu4pyscf, cupy, libxc, CUDA, or the
    interpreter is kept verbatim. Values are strings or ``None`` ("was not
    installed"), never structured objects.
    """

    name: str
    versions: dict[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        versions = require_mapping(self.versions, "engine.versions")
        checked: dict[str, str | None] = {}
        for key, value in versions.items():
            checked[key] = optional_str(value, f"engine.versions.{key}")
        object.__setattr__(self, "name", require_str(self.name, "engine.name"))
        object.__setattr__(self, "versions", checked)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this block."""
        return {"name": self.name, "versions": dict(self.versions)}

    @classmethod
    def from_dict(cls, data: Any) -> Engine:
        """Build an :class:`Engine`; unknown keys inside ``versions`` are allowed."""
        mapping = require_mapping(data, "engine")
        reject_unknown_keys(mapping, _ENGINE_KEYS, "engine")
        return cls(
            name=require_key(mapping, "name", "engine"),
            versions=mapping.get("versions") or {},
        )


@dataclass(frozen=True, kw_only=True)
class Results:
    """What the calculation returned, in calculation-native units.

    The gradient is dE/dR in hartree/bohr with one row per atom, in the same
    atom order as the structure. There is no force field here by design.
    """

    energy_hartree: float
    gradient_hartree_per_bohr: tuple[tuple[float, float, float], ...]
    converged: bool
    n_iterations: int | None = None
    s2: float | None = None
    s2_target: float | None = None
    s2_deviation: float | None = None
    wall_time_seconds: float | None = None
    scf_wall_time_seconds: float | None = None
    gradient_wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        iterations = optional_int(self.n_iterations, "results.n_iterations")
        if iterations is not None and iterations < 0:
            raise ValidationError(f"results.n_iterations must not be negative; got {iterations}.")
        object.__setattr__(
            self,
            "energy_hartree",
            require_finite_float(self.energy_hartree, "results.energy_hartree"),
        )
        object.__setattr__(
            self,
            "gradient_hartree_per_bohr",
            require_vector_rows(
                self.gradient_hartree_per_bohr, "results.gradient_hartree_per_bohr"
            ),
        )
        object.__setattr__(self, "converged", require_bool(self.converged, "results.converged"))
        object.__setattr__(self, "n_iterations", iterations)
        for name in ("s2", "s2_target", "s2_deviation"):
            object.__setattr__(
                self, name, optional_finite_float(getattr(self, name), f"results.{name}")
            )
        for name in (
            "wall_time_seconds",
            "scf_wall_time_seconds",
            "gradient_wall_time_seconds",
        ):
            seconds = optional_finite_float(getattr(self, name), f"results.{name}")
            if seconds is not None and seconds < 0:
                raise ValidationError(f"results.{name} must not be negative; got {seconds}.")
            object.__setattr__(self, name, seconds)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this block."""
        return {
            "energy_hartree": self.energy_hartree,
            "gradient_hartree_per_bohr": [list(row) for row in self.gradient_hartree_per_bohr],
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "s2": self.s2,
            "s2_target": self.s2_target,
            "s2_deviation": self.s2_deviation,
            "wall_time_seconds": self.wall_time_seconds,
            "scf_wall_time_seconds": self.scf_wall_time_seconds,
            "gradient_wall_time_seconds": self.gradient_wall_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Results:
        """Build :class:`Results`, refusing force fields and unknown keys."""
        mapping = require_mapping(data, "results")
        _reject_force_keys(mapping, "results")
        reject_unknown_keys(mapping, _RESULTS_KEYS, "results")
        return cls(
            energy_hartree=require_key(mapping, "energy_hartree", "results"),
            gradient_hartree_per_bohr=require_key(mapping, "gradient_hartree_per_bohr", "results"),
            converged=require_key(mapping, "converged", "results"),
            n_iterations=mapping.get("n_iterations"),
            s2=mapping.get("s2"),
            s2_target=mapping.get("s2_target"),
            s2_deviation=mapping.get("s2_deviation"),
            wall_time_seconds=mapping.get("wall_time_seconds"),
            scf_wall_time_seconds=mapping.get("scf_wall_time_seconds"),
            gradient_wall_time_seconds=mapping.get("gradient_wall_time_seconds"),
        )


@dataclass(frozen=True, kw_only=True)
class RawArtifact:
    """Where the untracked raw output lives, and what it hashed to.

    The location is logical (a path relative to a run root, an archive member),
    never an absolute path on one host, so a record stays meaningful elsewhere.
    """

    logical_location: str | None = None
    checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        checksum = optional_str(self.checksum_sha256, "raw.checksum_sha256")
        if checksum is not None:
            lowered = checksum.lower()
            if len(lowered) != 64 or not set(lowered) <= _HEX_DIGITS:
                raise ValidationError(
                    f"raw.checksum_sha256 must be 64 hexadecimal characters; got {checksum!r}."
                )
            checksum = lowered
        object.__setattr__(
            self, "logical_location", optional_str(self.logical_location, "raw.logical_location")
        )
        object.__setattr__(self, "checksum_sha256", checksum)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this block."""
        return {
            "logical_location": self.logical_location,
            "checksum_sha256": self.checksum_sha256,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RawArtifact:
        """Build a :class:`RawArtifact` from ``data``, rejecting unknown keys."""
        mapping = require_mapping(data, "raw")
        reject_unknown_keys(mapping, _RAW_KEYS, "raw")
        return cls(
            logical_location=mapping.get("logical_location"),
            checksum_sha256=mapping.get("checksum_sha256"),
        )


@dataclass(frozen=True, kw_only=True)
class QcState:
    """QC verdict and the append-only history that led to it.

    ``status`` is restricted to the three verdicts the QC milestone defines, so
    a record can never claim a state the release gate does not know. Each
    history entry timestamps itself (``utc``) and names what happened
    (``event``); anything else it carries is free-form provenance.
    """

    status: str
    history: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        status = require_str(self.status, "qc.status")
        if status not in QC_STATUSES:
            raise ValidationError(
                f"qc.status must be one of {', '.join(repr(value) for value in QC_STATUSES)}; "
                f"got {status!r}."
            )
        entries: list[dict[str, Any]] = []
        for index, entry in enumerate(require_sequence(self.history, "qc.history")):
            path = f"qc.history[{index}]"
            mapping = require_mapping(entry, path)
            for key in ("utc", "event"):
                require_str(mapping.get(key), f"{path}.{key}")
            entries.append(validated_json_object(mapping, path))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "history", tuple(entries))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this block."""
        return {"status": self.status, "history": [deepcopy(entry) for entry in self.history]}

    @classmethod
    def from_dict(cls, data: Any) -> QcState:
        """Build a :class:`QcState` from ``data``, rejecting unknown keys."""
        mapping = require_mapping(data, "qc")
        reject_unknown_keys(mapping, _QC_KEYS, "qc")
        return cls(
            status=require_key(mapping, "status", "qc"),
            history=mapping.get("history") or (),
        )


@dataclass(frozen=True, kw_only=True)
class LabelRecord:
    """One label: a structure, its electronic state, how it was computed, and what came out.

    Cross-block invariants are checked here, where all the blocks are visible:
    the gradient has one row per atom, and the electron count implied by the
    atoms and the charge is compatible with the multiplicity.
    """

    schema: str = LABEL_RECORD_SCHEMA
    record_id: str
    structure: Structure
    state: ElectronicState
    method: Method
    engine: Engine
    results: Results
    raw: RawArtifact
    qc: QcState
    units: dict[str, str] = field(default_factory=lambda: dict(CANONICAL_UNITS))

    def __post_init__(self) -> None:
        if self.schema != LABEL_RECORD_SCHEMA:
            raise ValidationError(
                f"record.schema must be {LABEL_RECORD_SCHEMA!r}; got {self.schema!r}."
            )
        units = require_mapping(self.units, "record.units")
        if units != dict(CANONICAL_UNITS):
            raise ValidationError(
                f"record.units must be exactly {dict(CANONICAL_UNITS)!r}; got {units!r}. "
                "A v1 record keeps calculation-native units and is never rewritten in "
                "place to say otherwise."
            )
        for name, block, expected in (
            ("structure", self.structure, Structure),
            ("state", self.state, ElectronicState),
            ("method", self.method, Method),
            ("engine", self.engine, Engine),
            ("results", self.results, Results),
            ("raw", self.raw, RawArtifact),
            ("qc", self.qc, QcState),
        ):
            if not isinstance(block, expected):
                raise ValidationError(
                    f"record.{name} must be a {expected.__name__}; got {type(block).__name__}."
                )
        gradient_rows = len(self.results.gradient_hartree_per_bohr)
        if gradient_rows != self.structure.atom_count:
            raise ValidationError(
                f"results.gradient_hartree_per_bohr has {gradient_rows} rows but the "
                f"structure has {self.structure.atom_count} atoms."
            )
        validate_electron_spin_parity(
            electron_count(self.structure.atomic_numbers, self.state.charge),
            self.state.multiplicity,
        )
        object.__setattr__(self, "record_id", validate_record_id(self.record_id))
        object.__setattr__(self, "units", units)

    @property
    def electron_count(self) -> int:
        """Electron count implied by the atoms and the total charge."""
        return electron_count(self.structure.atomic_numbers, self.state.charge)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of the whole record."""
        return {
            "schema": self.schema,
            "record_id": self.record_id,
            "structure": self.structure.to_dict(),
            "state": self.state.to_dict(),
            "method": self.method.to_dict(),
            "engine": self.engine.to_dict(),
            "results": self.results.to_dict(),
            "raw": self.raw.to_dict(),
            "qc": self.qc.to_dict(),
            "units": dict(self.units),
        }

    @classmethod
    def from_dict(cls, data: Any) -> LabelRecord:
        """Build a :class:`LabelRecord` from a decoded JSON object.

        The schema string is checked before anything else, so an older or
        foreign record is refused by name rather than by a confusing field
        error further in.
        """
        mapping = require_mapping(data, "record")
        _reject_force_keys(mapping, "record")
        schema = mapping.get("schema")
        if schema != LABEL_RECORD_SCHEMA:
            raise ValidationError(
                f"record.schema must be {LABEL_RECORD_SCHEMA!r}; got {schema!r}."
            )
        reject_unknown_keys(mapping, _RECORD_KEYS, "record")
        return cls(
            schema=schema,
            record_id=require_key(mapping, "record_id", "record"),
            structure=Structure.from_dict(require_key(mapping, "structure", "record")),
            state=ElectronicState.from_dict(require_key(mapping, "state", "record")),
            method=Method.from_dict(require_key(mapping, "method", "record")),
            engine=Engine.from_dict(require_key(mapping, "engine", "record")),
            results=Results.from_dict(require_key(mapping, "results", "record")),
            raw=RawArtifact.from_dict(require_key(mapping, "raw", "record")),
            qc=QcState.from_dict(require_key(mapping, "qc", "record")),
            units=require_key(mapping, "units", "record"),
        )
