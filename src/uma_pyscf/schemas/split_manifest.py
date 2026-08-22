"""The dataset split manifest: which group of related records went where.

Schema ``uma-pyscf-split-manifest-v1``. A split manifest is the written answer
to one question -- *for this axis, which partition does each group of related
records belong to* -- and it is the file the training and evaluation stages
read. It never carries a wall-clock timestamp: regenerating a split from the
same config and the same candidate set has to reproduce the file byte for byte,
so the only things that identify it are its ``split_id``, its ``seed``, and the
``source`` it was computed from.

The record holds two assignment maps, and the pair is the point of the schema:

* ``group_assignments`` maps a *group key* to a partition. The group key is what
  the axis groups by -- a parent structure id, a canonical formula, a charge, a
  multiplicity -- and a group is indivisible: it appears once, and it names one
  partition.
* ``record_assignments`` maps a partition to the records that landed in it. The
  two maps describe the same assignment from opposite ends, so a leak shows up
  as an inconsistency rather than as a silently duplicated record. Record ids
  are therefore checked for uniqueness across *all* partitions, not within one.

Everything derived is re-derived on the way in. ``counts`` is a summary of the
assignments, and :meth:`SplitManifest.from_dict` recomputes it and refuses a
manifest whose totals disagree with its own maps, the same way the geometry QC
report refuses one whose counts disagree with its entries.

One honest note about ordering: ``partitions`` is a mapping and Python preserves
its insertion order, which is the order the split config declared -- that order
is what breaks a tie between two partitions with the same deficit during
assignment. The on-disk JSON is written key-sorted by
:func:`~uma_pyscf.core.io.write_json_atomic`, so a manifest read back reports
its partitions in name order. That costs nothing, because an assignment is
computed from the config and never re-derived from a manifest; the config is
where declaration order lives.

The house rules of :mod:`uma_pyscf.schemas.label_record` apply unchanged: frozen
kw-only dataclasses, explicit ``to_dict``/``from_dict``, validation in the
constructor, and unknown keys refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ._fields import (
    reject_unknown_keys,
    require_finite_float,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
)

__all__ = [
    "FRACTION_SUM_TOLERANCE",
    "SPLIT_AXES",
    "SPLIT_MANIFEST_SCHEMA",
    "SplitManifest",
    "validate_axis",
    "validate_partitions",
]

SPLIT_MANIFEST_SCHEMA = "uma-pyscf-split-manifest-v1"

#: The grouping axes a v1 split may use, in the order the implementation plan
#: lists them. ``parent`` and ``composition`` keep same-geometry charge/spin
#: siblings together; ``charge`` and ``multiplicity`` deliberately separate them
#: to measure generalization. There is no ``random`` axis, and adding one would
#: defeat the purpose of the record: a per-record random split is exactly the
#: leakage this schema exists to make impossible.
SPLIT_AXES: tuple[str, ...] = ("parent", "composition", "charge", "multiplicity")

#: How far the declared fractions may sum from 1.0 before the split is refused.
#: Fractions come from a hand-written config, so this tolerates the binary
#: representation of values like 0.6 + 0.2 + 0.2 and nothing more.
FRACTION_SUM_TOLERANCE = 1e-9

_MANIFEST_KEYS = (
    "schema",
    "split_id",
    "axis",
    "seed",
    "partitions",
    "source",
    "group_assignments",
    "record_assignments",
    "counts",
)
_SOURCE_KEYS = ("id", "sha256")
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, path: str) -> str:
    """Return ``value`` as a lowercase 64-character hex digest."""
    digest = require_str(value, path).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValidationError(f"{path} must be 64 hexadecimal characters; got {value!r}.")
    return digest


def validate_axis(value: object, path: str) -> str:
    """Return ``value`` as one of :data:`SPLIT_AXES`, or raise naming the choices."""
    axis = require_str(value, path)
    if axis not in SPLIT_AXES:
        raise ValidationError(
            f"{path} must be one of {', '.join(repr(name) for name in SPLIT_AXES)}; "
            f"got {axis!r}. There is deliberately no random axis: a per-record random "
            "split is the leakage this machinery exists to prevent."
        )
    return axis


def validate_partitions(value: object, path: str) -> dict[str, float]:
    """Return the declared partitions as a name-to-fraction mapping.

    A split needs at least two partitions, every fraction has to be strictly
    positive, and the fractions have to sum to 1 within
    :data:`FRACTION_SUM_TOLERANCE`. A zero fraction is refused rather than read
    as "declare it but leave it empty": a partition nobody puts anything into is
    a partition that should not have been declared, and letting it through would
    make the honoured fractions silently different from the written ones.
    Partition names must satisfy the record id pattern, because they end up in
    file and directory names.
    """
    mapping = require_mapping(value, path)
    if len(mapping) < 2:
        raise ValidationError(
            f"{path} must declare at least two partitions; got {len(mapping)}. "
            "A split with one partition holds nothing back."
        )
    fractions: dict[str, float] = {}
    for name, raw in mapping.items():
        try:
            validate_record_id(name)
        except ValidationError as exc:
            raise ValidationError(f"{path} has an unusable partition name: {exc}") from exc
        fraction = require_finite_float(raw, f"{path}.{name}")
        if fraction <= 0.0:
            raise ValidationError(
                f"{path}.{name} must be a positive fraction; got {fraction}. Remove the "
                "partition instead of declaring it empty."
            )
        fractions[name] = fraction
    total = sum(fractions.values())
    if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
        raise ValidationError(
            f"{path} fractions sum to {total!r}, not 1.0 (tolerance "
            f"{FRACTION_SUM_TOLERANCE}); every record is assigned exactly once, so the "
            "declared fractions have to account for the whole dataset."
        )
    return fractions


def _validated_source(value: object, path: str) -> dict[str, str]:
    """Return the source block naming the dataset this split was computed from."""
    mapping = require_mapping(value, path)
    reject_unknown_keys(mapping, _SOURCE_KEYS, path)
    source_id = require_str(require_key(mapping, "id", path), f"{path}.id")
    validate_record_id(source_id)
    digest = _require_sha256(require_key(mapping, "sha256", path), f"{path}.sha256")
    return {"id": source_id, "sha256": digest}


@dataclass(frozen=True, kw_only=True)
class SplitManifest:
    """One axis of one dataset, split into partitions by whole groups.

    ``seed`` and ``split_id`` together decide the order groups are considered
    in, so they are stored: a split that cannot say what produced it cannot be
    regenerated. ``source`` names the candidate manifest or dataset the items
    came from and fingerprints its content, which is what makes "this split
    belongs to that data" checkable rather than assumed.
    """

    schema: str = SPLIT_MANIFEST_SCHEMA
    split_id: str
    axis: str
    seed: int
    partitions: dict[str, float]
    source: dict[str, str]
    group_assignments: dict[str, str] = field(default_factory=dict)
    record_assignments: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != SPLIT_MANIFEST_SCHEMA:
            raise ValidationError(
                f"split.schema must be {SPLIT_MANIFEST_SCHEMA!r}; got {self.schema!r}."
            )
        partitions = validate_partitions(self.partitions, "split.partitions")
        groups = self._validated_group_assignments(partitions)
        records = self._validated_record_assignments(partitions)
        object.__setattr__(self, "split_id", validate_record_id(self.split_id))
        object.__setattr__(self, "axis", validate_axis(self.axis, "split.axis"))
        object.__setattr__(self, "seed", require_int(self.seed, "split.seed"))
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "source", _validated_source(self.source, "split.source"))
        object.__setattr__(self, "group_assignments", groups)
        object.__setattr__(self, "record_assignments", records)

    def _validated_group_assignments(self, partitions: dict[str, float]) -> dict[str, str]:
        """Return the group map, checking every group names a declared partition."""
        mapping = require_mapping(self.group_assignments, "split.group_assignments")
        if not mapping:
            raise ValidationError(
                "split.group_assignments must assign at least one group; a split that "
                "groups nothing is not a split."
            )
        assignments: dict[str, str] = {}
        for group_key, partition in mapping.items():
            path = f"split.group_assignments.{group_key}"
            require_str(group_key, f"split.group_assignments key {group_key!r}")
            name = require_str(partition, path)
            if name not in partitions:
                raise ValidationError(
                    f"{path} names the partition {name!r}, which is not declared in "
                    f"split.partitions ({', '.join(repr(key) for key in partitions)})."
                )
            assignments[group_key] = name
        return assignments

    def _validated_record_assignments(
        self, partitions: dict[str, float]
    ) -> dict[str, tuple[str, ...]]:
        """Return the record map: every declared partition, records sorted and disjoint.

        Each declared partition appears exactly once, even when the assignment
        left it empty, so a reader never has to distinguish "no records" from
        "the writer forgot". Record ids are sorted within a partition, which
        makes the written file independent of the order the items arrived in.
        """
        mapping = require_mapping(self.record_assignments, "split.record_assignments")
        undeclared = sorted(set(mapping) - set(partitions))
        if undeclared:
            raise ValidationError(
                f"split.record_assignments has partition(s) "
                f"{', '.join(repr(name) for name in undeclared)} that split.partitions does "
                "not declare."
            )
        missing = [name for name in partitions if name not in mapping]
        if missing:
            raise ValidationError(
                f"split.record_assignments is missing the declared partition(s) "
                f"{', '.join(repr(name) for name in missing)}; every partition is listed, "
                "an empty one as an empty list."
            )
        assignments: dict[str, tuple[str, ...]] = {}
        seen: dict[str, str] = {}
        for name in partitions:
            path = f"split.record_assignments.{name}"
            ids: list[str] = []
            for index, raw in enumerate(require_sequence(mapping[name], path)):
                record_id = validate_record_id(require_str(raw, f"{path}[{index}]"))
                if record_id in seen:
                    raise ValidationError(
                        f"{path}[{index}] repeats the record {record_id!r}, which is already "
                        f"in partition {seen[record_id]!r}. Partitions are disjoint: a record "
                        "that is in two of them leaks between train and evaluation."
                    )
                seen[record_id] = name
                ids.append(record_id)
            assignments[name] = tuple(sorted(ids))
        if not seen:
            raise ValidationError(
                "split.record_assignments assigns no records at all; a split manifest "
                "describes where records went."
            )
        return assignments

    @property
    def record_count(self) -> int:
        """Total number of assigned records, across every partition."""
        return sum(len(ids) for ids in self.record_assignments.values())

    def groups_in(self, partition: str) -> int:
        """Return how many groups landed in ``partition``."""
        return sum(1 for name in self.group_assignments.values() if name == partition)

    @property
    def counts(self) -> dict[str, Any]:
        """Return the derived summary of both assignment maps."""
        return {
            "records": self.record_count,
            "groups": len(self.group_assignments),
            "records_by_partition": {
                name: len(self.record_assignments[name]) for name in self.partitions
            },
            "groups_by_partition": {name: self.groups_in(name) for name in self.partitions},
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form of the whole manifest."""
        return {
            "schema": self.schema,
            "split_id": self.split_id,
            "axis": self.axis,
            "seed": self.seed,
            "partitions": dict(self.partitions),
            "source": dict(self.source),
            "group_assignments": dict(self.group_assignments),
            "record_assignments": {
                name: list(ids) for name, ids in self.record_assignments.items()
            },
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, data: Any) -> SplitManifest:
        """Build a :class:`SplitManifest`, re-deriving the counts on the way.

        The schema string is checked first, so a foreign or older manifest is
        refused by name rather than by a confusing field error further in.
        """
        mapping = require_mapping(data, "split")
        schema = mapping.get("schema")
        if schema != SPLIT_MANIFEST_SCHEMA:
            raise ValidationError(
                f"split.schema must be {SPLIT_MANIFEST_SCHEMA!r}; got {schema!r}."
            )
        reject_unknown_keys(mapping, _MANIFEST_KEYS, "split")
        manifest = cls(
            schema=schema,
            split_id=require_key(mapping, "split_id", "split"),
            axis=require_key(mapping, "axis", "split"),
            seed=require_key(mapping, "seed", "split"),
            partitions=require_key(mapping, "partitions", "split"),
            source=require_key(mapping, "source", "split"),
            group_assignments=require_key(mapping, "group_assignments", "split"),
            record_assignments=require_key(mapping, "record_assignments", "split"),
        )
        stored = mapping.get("counts")
        if stored is not None:
            counts = require_mapping(stored, "split.counts")
            if counts != manifest.counts:
                raise ValidationError(
                    f"split.counts is {counts!r} but the assignments derive "
                    f"{manifest.counts!r}. The counts summarize the assignments and are "
                    "never stated apart from them."
                )
        return manifest
