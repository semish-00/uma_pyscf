"""Geometry QC on a computed label: the sampler's filters, applied again.

The three geometry questions -- are two atoms too close, did the structure fall
apart, is this geometry one already in the batch -- are exactly the ones
:mod:`uma_pyscf.sampling.filters` answers before a calculation is queued. This
module asks them again *after* the calculation, over ``record.structure``, and
it does so by calling that module rather than by carrying its own copy of the
collision, connectivity, and fingerprint arithmetic.

**Why the import goes sideways, and what would be worse.** The repository
structure plan orders module dependencies ``core -> schemas -> (sampling | qc |
datasets | ...) -> cli`` and rules out dependencies between those peers. This
import breaks that rule knowingly, in one direction only (``qc`` reads
``sampling.filters``; nothing in ``sampling`` reads ``qc``), because the
alternative is two implementations of the same geometry math that can disagree.
A structure accepted by the sampler and then rejected by QC for a reason the
sampler never applied -- or worse, the reverse -- would make the pipeline's
verdicts depend on which copy of the arithmetic ran, and that is a far more
expensive failure than one edge in the dependency graph.
:mod:`uma_pyscf.sampling.filters` is a leaf: it imports ``core`` and the
``Structure`` schema and nothing else, so the edge adds no cycle.

Records reaching QC are not candidates, so two differences from the sampling
stage are deliberate:

* There is no source-structure comparison. A label's geometry duplicating the
  seed it came from is normal (a state expansion computes exactly that
  geometry); what QC looks for is two *labels* that are the same calculation.
* Duplicate detection is **state-qualified**, the same rule sampling uses: the
  key is the composition, the rounded interatomic-distance fingerprint, the
  charge, and the multiplicity. The same geometry as a neutral singlet and as a
  cation doublet is two calculations and two labels, not a duplicate; the same
  geometry twice in the same state is one calculation billed twice and a leakage
  path between splits.

Each function returns a check dict -- ``name``, ``passed``, ``observed``,
``threshold`` -- and does not raise on a bad geometry, for the same reason the
filters do not: a rejection is a recorded verdict. A missing threshold does
raise, through the helpers in :mod:`uma_pyscf.qc.config`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from ..sampling.filters import (
    Fingerprint,
    fragment_count,
    minimum_distance_violation,
    pair_distance_fingerprint,
)
from ..schemas.label_record import LabelRecord
from .config import flag, non_negative_int, positive_threshold

__all__ = [
    "CHECK_DUPLICATE",
    "CHECK_FRAGMENTS",
    "CHECK_MINIMUM_DISTANCE",
    "GEOMETRY_CHECK_NAMES",
    "StateQualifiedKey",
    "check_duplicate",
    "check_fragments",
    "check_minimum_distance",
    "duplicate_map",
    "geometry_checks",
    "state_qualified_key",
]

CHECK_MINIMUM_DISTANCE = "minimum_distance"
CHECK_FRAGMENTS = "fragments"
CHECK_DUPLICATE = "duplicate"

#: The geometry checks, in the order :func:`geometry_checks` runs them.
GEOMETRY_CHECK_NAMES: tuple[str, ...] = (
    CHECK_MINIMUM_DISTANCE,
    CHECK_FRAGMENTS,
    CHECK_DUPLICATE,
)

#: Composition, rounded distance fingerprint, charge, multiplicity. Two records
#: are the same calculation exactly when these four agree.
StateQualifiedKey = tuple[tuple[int, ...], Fingerprint, int, int]

_SECTION = "geometry"


def _check(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    """Return one check result in the shape the QC report stores."""
    return {"name": name, "passed": passed, "observed": observed, "threshold": threshold}


def state_qualified_key(record: LabelRecord, decimals: int) -> StateQualifiedKey:
    """Return the key that decides whether two records are the same calculation."""
    return (
        tuple(sorted(record.structure.atomic_numbers)),
        pair_distance_fingerprint(record.structure, decimals),
        record.state.charge,
        record.state.multiplicity,
    )


def duplicate_map(records: Sequence[LabelRecord], section: Mapping[str, Any]) -> dict[str, str]:
    """Return each duplicate record's id mapped to the id of the record it repeats.

    The first record with a given key is the one kept, so the result depends on
    the order records are given in -- which is the order the caller resolved
    them in, and which :func:`uma_pyscf.qc.run.apply_qc` fixes by sorting file
    names. Only later duplicates appear in the mapping; a record that is not a
    key is not a duplicate of anything.
    """
    decimals = non_negative_int(section, "duplicate_decimals", _SECTION)
    kept: dict[StateQualifiedKey, str] = {}
    duplicates: dict[str, str] = {}
    for record in records:
        key = state_qualified_key(record, decimals)
        first = kept.get(key)
        if first is None:
            kept[key] = record.record_id
        else:
            duplicates[record.record_id] = first
    return duplicates


def check_minimum_distance(record: LabelRecord, section: Mapping[str, Any]) -> dict[str, Any]:
    """Check that no atom pair sits inside the covalent collision cutoff.

    ``observed`` is the worst offending pair -- both atom indices, both symbols,
    the distance, and the cutoff it failed -- or null when every pair is fine.
    """
    factor = positive_threshold(section, "covalent_factor", _SECTION)
    violation = minimum_distance_violation(record.structure, factor)
    return _check(CHECK_MINIMUM_DISTANCE, violation is None, deepcopy(violation), factor)


def check_fragments(record: LabelRecord, section: Mapping[str, Any]) -> dict[str, Any]:
    """Check whether the structure is connected, unless fragments are allowed.

    ``threshold`` names both config values that decide this one, because the
    fragment count alone does not say whether it is acceptable: a dissociation
    study sets ``allow_fragments`` and the same count then passes.
    """
    factor = positive_threshold(section, "bond_factor", _SECTION)
    allowed = flag(section, "allow_fragments", _SECTION)
    count = fragment_count(record.structure, factor)
    return _check(
        CHECK_FRAGMENTS,
        allowed or count == 1,
        count,
        {"bond_factor": factor, "allow_fragments": allowed},
    )


def check_duplicate(section: Mapping[str, Any], duplicate_of: str | None) -> dict[str, Any]:
    """Report whether this record repeats an earlier one in the same batch.

    ``observed`` is the id of the record that was kept, so a rejection names
    what it lost to; it is null when the record is the first of its kind.
    Duplicates are a property of a *batch*, which is why the verdict is computed
    once by :func:`duplicate_map` and passed in here rather than recomputed per
    record.
    """
    decimals = non_negative_int(section, "duplicate_decimals", _SECTION)
    return _check(CHECK_DUPLICATE, duplicate_of is None, duplicate_of, decimals)


def geometry_checks(
    record: LabelRecord, section: Mapping[str, Any], duplicate_of: str | None
) -> tuple[dict[str, Any], ...]:
    """Run every geometry check on one record, in :data:`GEOMETRY_CHECK_NAMES` order.

    All three run even after one fails, for the same reason the electronic
    checks do: a report is read to decide what to fix, and the whole list of
    reasons is what makes that possible.
    """
    return (
        check_minimum_distance(record, section),
        check_fragments(record, section),
        check_duplicate(section, duplicate_of),
    )
