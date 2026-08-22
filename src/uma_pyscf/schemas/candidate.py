"""Structure candidates and the geometry QC verdict that gates them.

Two versioned records live here, and they are written as a pair by the sampling
milestone:

* ``uma-pyscf-candidate-manifest-v1`` -- the candidates that passed every
  geometry filter, each with the electronic state it is to be computed in and
  the generation parameters that produced it. This is the file the label
  pipeline reads; a candidate that is not in it is not computed.
* ``uma-pyscf-geometry-qc-v1`` -- the verdict on *every* candidate the
  generator built, accepted and rejected alike, with the checks that were run
  and, for a rejection, the reason. Nothing is dropped silently: a structure
  that never reaches the manifest is still accounted for here.

Neither record carries a wall-clock timestamp. Regenerating from the same
config file has to reproduce both files byte for byte, so the only thing that
identifies a run is its content: the config is embedded verbatim and
``config_sha256`` is the fingerprint of that embedded content, which means a
reader can recompute it from the record alone. :meth:`CandidateManifest.from_dict`
does exactly that and rejects a mismatch, the same way the label record
re-derives ``spin_2s`` instead of believing it.

The house rules of :mod:`uma_pyscf.schemas.label_record` apply unchanged: frozen
kw-only dataclasses, explicit ``to_dict``/``from_dict``, validation in the
constructor, and unknown keys refused -- except inside ``generation_parameters``
and ``config``, which are free-form JSON by design and are validated as JSON
values instead.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, validate_record_id
from ..core.spin import electron_count, validate_electron_spin_parity
from ._fields import (
    optional_str,
    reject_unknown_keys,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    validated_json_object,
)
from .label_record import ElectronicState, Structure

__all__ = [
    "CANDIDATE_MANIFEST_SCHEMA",
    "CANDIDATE_STATUSES",
    "GEOMETRY_QC_SCHEMA",
    "CandidateManifest",
    "CandidateRecord",
    "GeometryQcReport",
]

CANDIDATE_MANIFEST_SCHEMA = "uma-pyscf-candidate-manifest-v1"
GEOMETRY_QC_SCHEMA = "uma-pyscf-geometry-qc-v1"

#: The verdicts a geometry QC entry may carry. A candidate is either in the
#: manifest or explained, and there is no third state.
CANDIDATE_STATUSES: tuple[str, ...] = ("accepted", "rejected")

_CANDIDATE_KEYS = ("record_id", "structure", "state", "generation_parameters")
_MANIFEST_KEYS = ("schema", "sampling_id", "config_sha256", "config", "records")
_QC_KEYS = ("schema", "sampling_id", "config_sha256", "counts", "entries")
_ENTRY_KEYS = ("record_id", "status", "checks", "reason")
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, path: str) -> str:
    """Return ``value`` as a lowercase 64-character hex digest."""
    digest = require_str(value, path).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValidationError(f"{path} must be 64 hexadecimal characters; got {value!r}.")
    return digest


def _validated_entry(value: object, path: str) -> dict[str, Any]:
    """Return one geometry QC entry, normalized and fully validated."""
    mapping = require_mapping(value, path)
    reject_unknown_keys(mapping, _ENTRY_KEYS, path)
    record_id = require_str(require_key(mapping, "record_id", path), f"{path}.record_id")
    validate_record_id(record_id)
    status = require_str(require_key(mapping, "status", path), f"{path}.status")
    if status not in CANDIDATE_STATUSES:
        raise ValidationError(
            f"{path}.status must be one of "
            f"{', '.join(repr(value) for value in CANDIDATE_STATUSES)}; got {status!r}."
        )
    checks = validated_json_object(require_key(mapping, "checks", path), f"{path}.checks")
    reason = optional_str(mapping.get("reason"), f"{path}.reason")
    if status == "rejected" and reason is None:
        raise ValidationError(
            f"{path}.reason is required when status is 'rejected': a candidate that "
            "does not reach the manifest has to say why."
        )
    if status == "accepted" and reason is not None:
        raise ValidationError(
            f"{path}.reason must be null when status is 'accepted'; got {reason!r}."
        )
    return {"record_id": record_id, "status": status, "checks": checks, "reason": reason}


@dataclass(frozen=True, kw_only=True)
class CandidateRecord:
    """One proposed structure plus the electronic state it is to be computed in.

    The generation provenance is not a separate block: ``structure`` already
    carries ``parent_structure_id`` (which seed it came from),
    ``sampling_method`` (which operation made it), and ``random_seed`` where an
    operation used one, so a candidate and the label eventually computed from it
    describe their origin the same way. ``generation_parameters`` adds the
    operation's own settings -- the scan factor, the displacement sigma, the
    state that was requested -- as free-form JSON.
    """

    record_id: str
    structure: Structure
    state: ElectronicState
    generation_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, block, expected in (
            ("structure", self.structure, Structure),
            ("state", self.state, ElectronicState),
        ):
            if not isinstance(block, expected):
                raise ValidationError(
                    f"candidate.{name} must be a {expected.__name__}; got {type(block).__name__}."
                )
        validate_electron_spin_parity(
            electron_count(self.structure.atomic_numbers, self.state.charge),
            self.state.multiplicity,
        )
        object.__setattr__(self, "record_id", validate_record_id(self.record_id))
        object.__setattr__(
            self,
            "generation_parameters",
            validated_json_object(self.generation_parameters, "candidate.generation_parameters"),
        )

    @property
    def electron_count(self) -> int:
        """Electron count implied by the atoms and the total charge."""
        return electron_count(self.structure.atomic_numbers, self.state.charge)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of this candidate."""
        return {
            "record_id": self.record_id,
            "structure": self.structure.to_dict(),
            "state": self.state.to_dict(),
            "generation_parameters": deepcopy(self.generation_parameters),
        }

    @classmethod
    def from_dict(cls, data: Any) -> CandidateRecord:
        """Build a :class:`CandidateRecord` from ``data``, rejecting unknown keys."""
        mapping = require_mapping(data, "candidate")
        reject_unknown_keys(mapping, _CANDIDATE_KEYS, "candidate")
        return cls(
            record_id=require_key(mapping, "record_id", "candidate"),
            structure=Structure.from_dict(require_key(mapping, "structure", "candidate")),
            state=ElectronicState.from_dict(require_key(mapping, "state", "candidate")),
            generation_parameters=mapping.get("generation_parameters") or {},
        )


@dataclass(frozen=True, kw_only=True)
class CandidateManifest:
    """The accepted candidates of one sampling run, with the config that made them.

    ``config`` is the loaded sampling config verbatim and ``config_sha256`` is
    :func:`~uma_pyscf.core.ids.canonical_json_fingerprint` of it. Hashing the
    content rather than the file bytes is deliberate: the digest then depends on
    what the run was told to do and not on the YAML's comments or key order, and
    any reader holding the manifest can recompute it.
    """

    schema: str = CANDIDATE_MANIFEST_SCHEMA
    sampling_id: str
    config_sha256: str
    config: dict[str, Any]
    records: tuple[CandidateRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_MANIFEST_SCHEMA:
            raise ValidationError(
                f"manifest.schema must be {CANDIDATE_MANIFEST_SCHEMA!r}; got {self.schema!r}."
            )
        config = validated_json_object(self.config, "manifest.config")
        digest = _require_sha256(self.config_sha256, "manifest.config_sha256")
        expected = canonical_json_fingerprint(config)
        if digest != expected:
            raise ValidationError(
                f"manifest.config_sha256 is {digest} but the embedded config fingerprints "
                f"to {expected}. The digest is derived from the config, never stated "
                "independently of it."
            )
        records: list[CandidateRecord] = []
        seen: set[str] = set()
        for index, record in enumerate(require_sequence(self.records, "manifest.records")):
            path = f"manifest.records[{index}]"
            if not isinstance(record, CandidateRecord):
                raise ValidationError(
                    f"{path} must be a CandidateRecord; got {type(record).__name__}."
                )
            if record.record_id in seen:
                raise ValidationError(
                    f"{path}.record_id {record.record_id!r} is already used by an earlier "
                    "record; candidate ids identify a calculation and must be unique."
                )
            seen.add(record.record_id)
            records.append(record)
        object.__setattr__(self, "sampling_id", validate_record_id(self.sampling_id))
        object.__setattr__(self, "config_sha256", digest)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "records", tuple(records))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of the whole manifest."""
        return {
            "schema": self.schema,
            "sampling_id": self.sampling_id,
            "config_sha256": self.config_sha256,
            "config": deepcopy(self.config),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Any) -> CandidateManifest:
        """Build a :class:`CandidateManifest` from a decoded JSON object.

        The schema string is checked first, so a foreign or older manifest is
        refused by name rather than by a confusing field error further in.
        """
        mapping = require_mapping(data, "manifest")
        schema = mapping.get("schema")
        if schema != CANDIDATE_MANIFEST_SCHEMA:
            raise ValidationError(
                f"manifest.schema must be {CANDIDATE_MANIFEST_SCHEMA!r}; got {schema!r}."
            )
        reject_unknown_keys(mapping, _MANIFEST_KEYS, "manifest")
        raw_records = require_sequence(
            require_key(mapping, "records", "manifest"), "manifest.records"
        )
        return cls(
            schema=schema,
            sampling_id=require_key(mapping, "sampling_id", "manifest"),
            config_sha256=require_key(mapping, "config_sha256", "manifest"),
            config=require_key(mapping, "config", "manifest"),
            records=tuple(CandidateRecord.from_dict(record) for record in raw_records),
        )


@dataclass(frozen=True, kw_only=True)
class GeometryQcReport:
    """Every candidate the generator built, with its geometry verdict.

    One entry per candidate, in generation order: ``record_id``, ``status``,
    the ``checks`` that were run with their results, and ``reason`` -- required
    for a rejection, forbidden for an acceptance. Entries stop at the first
    failing check, so ``checks`` shows what was actually evaluated rather than
    implying a check ran after the verdict was already decided.

    The counts in :meth:`to_dict` are derived, never stored authoritatively:
    :meth:`from_dict` recomputes them and refuses a report whose totals disagree
    with its own entries.
    """

    schema: str = GEOMETRY_QC_SCHEMA
    sampling_id: str
    config_sha256: str
    entries: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.schema != GEOMETRY_QC_SCHEMA:
            raise ValidationError(
                f"report.schema must be {GEOMETRY_QC_SCHEMA!r}; got {self.schema!r}."
            )
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, entry in enumerate(require_sequence(self.entries, "report.entries")):
            checked = _validated_entry(entry, f"report.entries[{index}]")
            if checked["record_id"] in seen:
                raise ValidationError(
                    f"report.entries[{index}].record_id {checked['record_id']!r} is already "
                    "used by an earlier entry; every candidate is reported exactly once."
                )
            seen.add(checked["record_id"])
            entries.append(checked)
        object.__setattr__(self, "sampling_id", validate_record_id(self.sampling_id))
        object.__setattr__(
            self, "config_sha256", _require_sha256(self.config_sha256, "report.config_sha256")
        )
        object.__setattr__(self, "entries", tuple(entries))

    def count(self, status: str) -> int:
        """Return how many entries carry ``status``."""
        return sum(1 for entry in self.entries if entry["status"] == status)

    @property
    def counts(self) -> dict[str, int]:
        """Return the derived entry counts, total first."""
        counts = {"total": len(self.entries)}
        counts.update({status: self.count(status) for status in CANDIDATE_STATUSES})
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of the whole report."""
        return {
            "schema": self.schema,
            "sampling_id": self.sampling_id,
            "config_sha256": self.config_sha256,
            "counts": self.counts,
            "entries": [deepcopy(entry) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Any) -> GeometryQcReport:
        """Build a :class:`GeometryQcReport`, re-deriving the counts on the way."""
        mapping = require_mapping(data, "report")
        schema = mapping.get("schema")
        if schema != GEOMETRY_QC_SCHEMA:
            raise ValidationError(f"report.schema must be {GEOMETRY_QC_SCHEMA!r}; got {schema!r}.")
        reject_unknown_keys(mapping, _QC_KEYS, "report")
        report = cls(
            schema=schema,
            sampling_id=require_key(mapping, "sampling_id", "report"),
            config_sha256=require_key(mapping, "config_sha256", "report"),
            entries=tuple(
                require_sequence(require_key(mapping, "entries", "report"), "report.entries")
            ),
        )
        stored = mapping.get("counts")
        if stored is not None:
            counts = require_mapping(stored, "report.counts")
            for key, value in counts.items():
                require_int(value, f"report.counts.{key}")
            if counts != report.counts:
                raise ValidationError(
                    f"report.counts is {counts!r} but the entries derive {report.counts!r}. "
                    "The counts are a summary of the entries and are never stated apart "
                    "from them."
                )
        return report
