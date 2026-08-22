"""Geometry filters that decide whether a candidate is worth a DFT calculation.

Three questions are asked of every candidate, and each is answered from the
covalent radii in :mod:`uma_pyscf.core.elements` and nothing else:

* Are two atoms closer than chemistry allows? A collision produces a wildly
  repulsive energy that teaches a model nothing and may not even converge.
* Did the structure fall apart into separate fragments? Sometimes that is the
  point (dissociation) and sometimes it is a runaway displacement, so the
  count is reported and the config decides.
* Is this geometry one already generated? A duplicate costs a DFT calculation
  and adds a leakage path between train and test splits.

Duplicate detection uses the multiset of interatomic distances, which is
invariant under rotation, translation, and reflection: the same molecule written
in a different frame has the same fingerprint, while a scanned bond does not.
Two structures can in principle share a distance multiset without being
congruent; that costs a candidate, never a wrong label, which is the direction
this project errs in.

None of these functions raise on a bad structure -- a violation is a return
value, because a rejected candidate is a recorded outcome and not an error. They
do raise for an element with no tabulated covalent radius, since that is a gap
in the table rather than a verdict about the geometry.
"""

from __future__ import annotations

from math import dist
from typing import Any

from ..core.elements import PERIODIC_SYMBOLS, covalent_radius
from ..core.errors import ValidationError
from ..schemas.label_record import Structure

__all__ = [
    "fragment_count",
    "is_duplicate",
    "minimum_distance_violation",
    "pair_distance_fingerprint",
]

Fingerprint = tuple[tuple[tuple[str, str], float], ...]


def _require_positive(value: object, name: str) -> float:
    """Return ``value`` as a positive float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{name} must be a number; got {value!r}.")
    number = float(value)
    if not number > 0.0:
        raise ValidationError(f"{name} must be positive; got {number}.")
    return number


def _symbols(structure: Structure) -> tuple[str, ...]:
    """Return the element symbols of ``structure`` in atom order."""
    return tuple(PERIODIC_SYMBOLS[number] for number in structure.atomic_numbers)


def _radii(structure: Structure) -> tuple[float, ...]:
    """Return the covalent radius of every atom, failing on an untabulated element."""
    return tuple(covalent_radius(number) for number in structure.atomic_numbers)


def minimum_distance_violation(
    structure: Structure, covalent_factor: float
) -> dict[str, Any] | None:
    """Return the worst too-close atom pair, or ``None`` when every pair is fine.

    A pair is too close when its distance is below
    ``covalent_factor * (r_i + r_j)``. "Worst" is the smallest distance relative
    to its own cutoff, so a pair of small atoms that are far inside their limit
    outranks a pair of large ones that are barely inside theirs; ties keep the
    lower atom indices. The returned dict names both atoms, the distance, and
    the cutoff it failed, so the QC report can state the verdict in full.
    """
    factor = _require_positive(covalent_factor, "covalent_factor")
    symbols = _symbols(structure)
    radii = _radii(structure)
    positions = structure.positions_angstrom
    worst: dict[str, Any] | None = None
    worst_ratio = 1.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            cutoff = factor * (radii[i] + radii[j])
            separation = dist(positions[i], positions[j])
            ratio = separation / cutoff if cutoff > 0.0 else 0.0
            if separation < cutoff and ratio < worst_ratio:
                worst_ratio = ratio
                worst = {
                    "atom_indices": [i, j],
                    "symbols": [symbols[i], symbols[j]],
                    "distance_angstrom": separation,
                    "cutoff_angstrom": cutoff,
                }
    return worst


def fragment_count(structure: Structure, bond_factor: float) -> int:
    """Return how many connected fragments ``structure`` falls into.

    Two atoms are bonded when their distance is at most
    ``bond_factor * (r_i + r_j)``; the fragments are the connected components of
    that graph, found with union-find. A single atom is one fragment.
    """
    factor = _require_positive(bond_factor, "bond_factor")
    radii = _radii(structure)
    positions = structure.positions_angstrom
    parent = list(range(len(positions)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if dist(positions[i], positions[j]) <= factor * (radii[i] + radii[j]):
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
    return len({find(index) for index in range(len(positions))})


def pair_distance_fingerprint(structure: Structure, decimals: int) -> Fingerprint:
    """Return the sorted, rounded interatomic distance multiset of ``structure``.

    Each entry pairs the two element symbols, ordered alphabetically so the pair
    does not depend on atom order, with the rounded distance in angstrom. The
    whole tuple is sorted, which makes it invariant under any relabelling of
    identical atoms as well as under rigid motion. ``decimals`` sets how close
    two geometries have to be to count as the same one.
    """
    if not isinstance(decimals, int) or isinstance(decimals, bool):
        raise ValidationError(f"decimals must be an integer; got {decimals!r}.")
    if decimals < 0:
        raise ValidationError(f"decimals must not be negative; got {decimals}.")
    symbols = _symbols(structure)
    positions = structure.positions_angstrom
    entries: list[tuple[tuple[str, str], float]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            left, right = symbols[i], symbols[j]
            pair = (left, right) if left <= right else (right, left)
            entries.append((pair, round(dist(positions[i], positions[j]), decimals)))
    return tuple(sorted(entries))


def is_duplicate(a: Structure, b: Structure, decimals: int) -> bool:
    """Return whether two structures are the same geometry to ``decimals`` places.

    Composition is compared first: a distance fingerprint says nothing about
    which atoms are present when there are no pairs to measure (a single atom),
    and two structures made of different elements are never the same structure
    regardless of how their distances line up.
    """
    if sorted(a.atomic_numbers) != sorted(b.atomic_numbers):
        return False
    return pair_distance_fingerprint(a, decimals) == pair_distance_fingerprint(b, decimals)
