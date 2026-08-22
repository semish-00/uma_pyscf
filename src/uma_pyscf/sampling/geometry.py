"""Deterministic geometry operations on a :class:`~uma_pyscf.schemas.Structure`.

Every function here is pure: it reads a structure and returns a new one, never
mutating its argument (a :class:`Structure` is frozen anyway, which is why the
operations are written this way rather than in place). Element identity and the
provenance fields ``parent_structure_id``, ``sampling_method``, and
``random_seed`` are carried through unchanged -- filling those in is the
generator's job, so that one place decides what a candidate claims about its
origin.

The two operations are the Part I ladder's perturbations, generalized: the
validation experiment could scale only the first ligand of a molecule centred on
the origin and could displace only whole structures, and both are special cases
of what is implemented here. The random displacement draws in exactly the Part I
nesting order -- per atom, then x, y, z, from ``random.Random(seed)`` -- so the
same seed reproduces the same numbers as before the port, and reproduces them on
any interpreter, which is what makes the candidate manifest regenerable.
"""

from __future__ import annotations

import random

from ..core.errors import ValidationError
from ..schemas.label_record import Structure

__all__ = ["gaussian_displacement", "scale_bond"]


def _require_index(value: object, name: str, atom_count: int) -> int:
    """Return ``value`` as an atom index that exists in a structure of this size."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer; got {value!r}.")
    if not 0 <= value < atom_count:
        raise ValidationError(
            f"{name} must be from 0 through {atom_count - 1} for a structure with "
            f"{atom_count} atoms; got {value}."
        )
    return value


def _rebuilt(structure: Structure, positions: list[tuple[float, float, float]]) -> Structure:
    """Return ``structure`` with new positions and its provenance carried over."""
    return Structure(
        atomic_numbers=structure.atomic_numbers,
        positions_angstrom=tuple(positions),
        parent_structure_id=structure.parent_structure_id,
        sampling_method=structure.sampling_method,
        random_seed=structure.random_seed,
    )


def scale_bond(
    structure: Structure, anchor_index: int, moved_index: int, factor: float
) -> Structure:
    """Return ``structure`` with one atom moved along its bond to another.

    Only ``moved_index`` moves, and it moves along the vector from the anchor:
    ``r_moved' = r_anchor + factor * (r_moved - r_anchor)``. The anchor and every
    other atom keep their coordinates exactly, so the scan changes one internal
    coordinate and nothing else -- no centre of mass correction, no rotation.
    A factor of 1 is a no-op geometrically and is allowed here; whether a
    candidate that reproduces its seed is worth computing is the generator's
    call, not this function's.
    """
    count = structure.atom_count
    anchor = _require_index(anchor_index, "anchor_index", count)
    moved = _require_index(moved_index, "moved_index", count)
    if anchor == moved:
        raise ValidationError(
            f"anchor_index and moved_index must name different atoms; both are {anchor}."
        )
    if isinstance(factor, bool) or not isinstance(factor, int | float):
        raise ValidationError(f"factor must be a number; got {factor!r}.")
    scale = float(factor)
    if not scale > 0.0:
        raise ValidationError(
            f"factor must be positive; got {scale}. A factor of zero collapses the bond "
            "and a negative one reflects the atom through the anchor."
        )
    base = structure.positions_angstrom[anchor]
    target = structure.positions_angstrom[moved]
    positions = list(structure.positions_angstrom)
    positions[moved] = (
        base[0] + scale * (target[0] - base[0]),
        base[1] + scale * (target[1] - base[1]),
        base[2] + scale * (target[2] - base[2]),
    )
    return _rebuilt(structure, positions)


def gaussian_displacement(
    structure: Structure,
    sigma_angstrom: float,
    seed: int,
    remove_translation: bool = True,
) -> Structure:
    """Return ``structure`` with every Cartesian component jittered independently.

    Each component is drawn from a normal distribution of width
    ``sigma_angstrom`` centred on zero. With ``remove_translation`` the mean
    shift per axis is subtracted afterwards, so the perturbation describes the
    internal geometry and does not also slide the molecule -- a translation
    would change nothing physically while making two identical structures look
    different to a coordinate comparison.

    The result depends on ``seed`` alone, and no geometry filter is applied
    here: a displacement that drives two atoms together produces a candidate
    that the generator rejects and reports, rather than an exception.
    """
    if isinstance(sigma_angstrom, bool) or not isinstance(sigma_angstrom, int | float):
        raise ValidationError(f"sigma_angstrom must be a number; got {sigma_angstrom!r}.")
    sigma = float(sigma_angstrom)
    if not sigma > 0.0:
        raise ValidationError(f"sigma_angstrom must be positive; got {sigma}.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError(f"seed must be an integer; got {seed!r}.")
    if not isinstance(remove_translation, bool):
        raise ValidationError(
            f"remove_translation must be true or false; got {remove_translation!r}."
        )

    rng = random.Random(seed)
    shifts = [[rng.gauss(0.0, sigma) for _ in range(3)] for _ in structure.atomic_numbers]
    mean = [0.0, 0.0, 0.0]
    if remove_translation:
        mean = [sum(row[axis] for row in shifts) / len(shifts) for axis in range(3)]
    positions = [
        (
            position[0] + shift[0] - mean[0],
            position[1] + shift[1] - mean[1],
            position[2] + shift[2] - mean[2],
        )
        for position, shift in zip(structure.positions_angstrom, shifts, strict=True)
    ]
    return _rebuilt(structure, positions)
