"""Group-wise dataset splits: assign related records together, never one by one.

This module exists to make one class of mistake impossible. Records in this
project are not independent samples: a bond scan produces twenty geometries of
the same molecule, a displacement set produces frames that differ by hundredths
of an angstrom, and a state expansion produces the same geometry in two charge
or spin states. Splitting such records individually puts near-copies of a
training structure into the evaluation set, and the resulting metric measures
memorization rather than generalization. So nothing here ever assigns a record.
It assigns a **group**, and every record of that group follows it. There is
deliberately no ``random`` axis: a per-record random split is precisely the
failure this machinery prevents, and the project plan rules it out for the
primary evaluation.

**The sibling rule, and why it is asymmetric.** Same-geometry charge/spin
siblings -- one structure computed as a neutral singlet and as a cation doublet
-- are related records, and under the ``parent`` and ``composition`` axes they
land in the same partition, because a model that saw the neutral has effectively
seen the cation's geometry. Under the ``charge`` and ``multiplicity`` axes those
same siblings are deliberately pulled apart: the whole purpose of a charge or
spin holdout is to ask whether the model generalizes to an electronic state it
never saw, and keeping siblings together would make that question unanswerable.
The asymmetry is by design, not an inconsistency. It follows the implementation
plan section 12 rule -- siblings share a group in the ordinary splits, and only
the generalization splits separate them -- and it means a charge or multiplicity
split's numbers must be reported as a generalization metric, never mixed into
the ordinary ones.

**Determinism.** Assignment is a pure function of the items, the axis, the
partitions, the seed, and the split id. Nothing reads the clock, the
environment, or a global random state. Group order comes from
``sha256(f"{split_id}|{seed}|{group_key}")``, which spreads related group keys
apart without any RNG, and each group in that order goes to whichever partition
is furthest behind its target item count. Regenerating a split from the same
inputs therefore reproduces the same manifest, byte for byte, on any host.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from ..core.elements import PERIODIC_SYMBOLS
from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ..schemas._fields import require_int, require_str
from ..schemas.candidate import CandidateManifest
from ..schemas.label_record import LabelRecord
from ..schemas.split_manifest import (
    SPLIT_AXES,
    SplitManifest,
    validate_axis,
    validate_partitions,
)

__all__ = [
    "AXES",
    "SplitItem",
    "assign_groups",
    "composition_formula",
    "generate_split",
    "group_key_for",
    "split_item_from_label_record",
    "split_items_from_candidate_manifest",
]

#: The grouping axes this module implements. The tuple is the schema's
#: :data:`~uma_pyscf.schemas.split_manifest.SPLIT_AXES`, re-exported here under
#: the name the split machinery uses: what counts as a valid axis is part of
#: what a split manifest *is*, so the schema layer owns the list and this module
#: consumes it rather than keeping a second copy that could drift.
AXES: tuple[str, ...] = SPLIT_AXES


def composition_formula(atomic_numbers: Sequence[int]) -> str:
    """Return the canonical formula of a set of atoms.

    Element symbols are sorted alphabetically and a count of one is omitted, so
    silane is ``H4Si`` and the SiH3 radical is ``H3Si``. This is the same
    convention the Part I parity export uses for its ``formula`` column, which
    keeps a composition string comparable across the two halves of the project.
    Sorting alphabetically rather than by Hill notation is a deliberate choice
    of one rule that never needs a special case for carbon.
    """
    counts: dict[str, int] = {}
    for index, number in enumerate(atomic_numbers):
        atomic = require_int(number, f"atomic_numbers[{index}]")
        if not 1 <= atomic < len(PERIODIC_SYMBOLS):
            raise ValidationError(
                f"atomic_numbers[{index}] must be an atomic number from 1 through "
                f"{len(PERIODIC_SYMBOLS) - 1}; got {atomic}."
            )
        symbol = PERIODIC_SYMBOLS[atomic]
        counts[symbol] = counts.get(symbol, 0) + 1
    if not counts:
        raise ValidationError("A composition needs at least one atom; got none.")
    return "".join(
        symbol + (str(count) if count > 1 else "") for symbol, count in sorted(counts.items())
    )


@dataclass(frozen=True, kw_only=True)
class SplitItem:
    """The few facts about a record that decide which group it belongs to.

    A split does not need geometries, energies, or methods -- only the identity
    of the record and the four properties the axes group by. Keeping the item
    this small is what lets one splitter serve candidate manifests and label
    records alike, and it keeps the split machinery independent of any storage
    format.

    Charge and multiplicity are carried, not re-checked: an item is built from a
    record that already passed the schema layer's electron/spin parity check, and
    repeating that check here would either duplicate the rule or, worse,
    disagree with it.
    """

    record_id: str
    parent_structure_id: str | None
    composition: str
    charge: int
    multiplicity: int

    def __post_init__(self) -> None:
        parent = self.parent_structure_id
        if parent is not None:
            validate_record_id(require_str(parent, "item.parent_structure_id"))
        multiplicity = require_int(self.multiplicity, "item.multiplicity")
        if multiplicity < 1:
            raise ValidationError(f"item.multiplicity must be at least 1; got {multiplicity}.")
        object.__setattr__(self, "record_id", validate_record_id(self.record_id))
        object.__setattr__(self, "composition", require_str(self.composition, "item.composition"))
        object.__setattr__(self, "charge", require_int(self.charge, "item.charge"))
        object.__setattr__(self, "multiplicity", multiplicity)


def split_items_from_candidate_manifest(manifest: CandidateManifest) -> tuple[SplitItem, ...]:
    """Return one :class:`SplitItem` per candidate, in manifest order."""
    if not isinstance(manifest, CandidateManifest):
        raise ValidationError(f"A candidate manifest is required; got {type(manifest).__name__}.")
    return tuple(
        SplitItem(
            record_id=record.record_id,
            parent_structure_id=record.structure.parent_structure_id,
            composition=composition_formula(record.structure.atomic_numbers),
            charge=record.state.charge,
            multiplicity=record.state.multiplicity,
        )
        for record in manifest.records
    )


def split_item_from_label_record(record: LabelRecord) -> SplitItem:
    """Return the :class:`SplitItem` describing one computed label record."""
    if not isinstance(record, LabelRecord):
        raise ValidationError(f"A label record is required; got {type(record).__name__}.")
    return SplitItem(
        record_id=record.record_id,
        parent_structure_id=record.structure.parent_structure_id,
        composition=composition_formula(record.structure.atomic_numbers),
        charge=record.state.charge,
        multiplicity=record.state.multiplicity,
    )


def group_key_for(item: SplitItem, axis: str) -> str:
    """Return the group key ``item`` has on ``axis``.

    A record with no ``parent_structure_id`` is its own group on the ``parent``
    axis: its record id becomes the key. That is the conservative reading --
    an unattributed structure is assumed to be related to nothing, so it moves
    alone rather than being pooled with every other parentless record into one
    giant group that would dominate a partition.
    """
    checked = validate_axis(axis, "axis")
    if checked == "parent":
        return item.parent_structure_id or item.record_id
    if checked == "composition":
        return item.composition
    if checked == "charge":
        return str(item.charge)
    return str(item.multiplicity)


def _grouped(items: Sequence[SplitItem], axis: str) -> dict[str, list[SplitItem]]:
    """Return the items bucketed by group key, in first-appearance order."""
    groups: dict[str, list[SplitItem]] = {}
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, SplitItem):
            raise ValidationError(f"Every item must be a SplitItem; got {type(item).__name__}.")
        if item.record_id in seen:
            raise ValidationError(
                f"The record {item.record_id!r} appears twice in the items to split; a "
                "record belongs to exactly one group and therefore to one partition."
            )
        seen.add(item.record_id)
        groups.setdefault(group_key_for(item, axis), []).append(item)
    return groups


def _ordering_digest(split_id: str, seed: int, group_key: str) -> str:
    """Return the digest that orders one group within its split."""
    return sha256(f"{split_id}|{seed}|{group_key}".encode()).hexdigest()


def assign_groups(
    items: Sequence[SplitItem],
    axis: str,
    partitions: Mapping[str, float],
    seed: int,
    split_id: str,
) -> dict[str, str]:
    """Assign every group of ``items`` on ``axis`` to one partition.

    The algorithm is fixed, and it is written out here because regenerating a
    split has to be stable across releases:

    1. Bucket the items by their group key on ``axis``.
    2. Order the groups by ``sha256(f"{split_id}|{seed}|{group_key}")``, with
       the group key itself breaking a digest tie.
    3. Walk that order and give each group to the partition with the largest
       *deficit*, where a partition's deficit is
       ``fraction * total_items - items_already_assigned_to_it``. A tie goes to
       whichever partition the config declared first.

    Assigning by deficit rather than by cumulative fraction boundaries is what
    lets groups have wildly different sizes -- a twenty-point scan and a single
    cation -- while the partitions still come out close to their declared
    fractions *by record count*. With equal-sized groups it is exact.

    Refusing to split fewer groups than there are partitions is not a
    limitation to work around: it is the guarantee doing its job. A dataset with
    one parent structure has one parent group, and no honest parent holdout can
    be cut from it -- the answer is more parent structures, not a smaller group.
    """
    checked_axis = validate_axis(axis, "axis")
    fractions = validate_partitions(partitions, "partitions")
    validate_record_id(split_id)
    require_int(seed, "seed")
    groups = _grouped(items, checked_axis)
    if len(groups) < len(fractions):
        raise ValidationError(
            f"Splitting on the {checked_axis!r} axis found {len(groups)} distinct group(s) "
            f"({', '.join(repr(key) for key in sorted(groups))}) but "
            f"{len(fractions)} partitions were declared "
            f"({', '.join(repr(name) for name in fractions)}). Groups are never divided, "
            "so every partition would have to receive at least one: either the dataset "
            "needs more distinct groups on this axis, or the split needs fewer partitions."
        )
    total = sum(len(members) for members in groups.values())
    ordered = sorted(groups, key=lambda key: (_ordering_digest(split_id, seed, key), key))
    assigned: dict[str, int] = {name: 0 for name in fractions}
    result: dict[str, str] = {}
    for group_key in ordered:
        # `max` returns the first maximal element, and `fractions` iterates in
        # declaration order, so an exact deficit tie falls to the partition the
        # config named first.
        chosen = max(fractions, key=lambda name: fractions[name] * total - assigned[name])
        result[group_key] = chosen
        assigned[chosen] += len(groups[group_key])
    return result


def generate_split(
    items: Sequence[SplitItem],
    *,
    split_id: str,
    axis: str,
    partitions: Mapping[str, float],
    seed: int,
    source_id: str,
    source_sha256: str,
) -> SplitManifest:
    """Assign the items and return the manifest describing the result.

    ``source_id`` and ``source_sha256`` name the candidate manifest or dataset
    the items were taken from, so a split can be checked against the data it
    claims to describe rather than being trusted to belong to it.
    """
    checked_axis = validate_axis(axis, "axis")
    group_assignments = assign_groups(items, checked_axis, partitions, seed, split_id)
    record_assignments: dict[str, list[str]] = {name: [] for name in partitions}
    for group_key, members in _grouped(items, checked_axis).items():
        partition = group_assignments[group_key]
        record_assignments[partition].extend(member.record_id for member in members)
    return SplitManifest(
        split_id=split_id,
        axis=checked_axis,
        seed=seed,
        partitions=dict(partitions),
        source={"id": source_id, "sha256": source_sha256},
        group_assignments=group_assignments,
        record_assignments={name: tuple(ids) for name, ids in record_assignments.items()},
    )
