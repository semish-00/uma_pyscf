"""Charge and spin siblings: one geometry, several electronic states.

A large part of what a fine-tuned model has to learn about charged and open-shell
species comes from the *same* geometry computed in different states -- a neutral
singlet, its cation doublet, the anion -- because the geometry is then held fixed
and only the electronic structure differs. Expanding those siblings is this
module's whole job.

Every requested state is checked against the geometry's own electron count
before it is returned: ``2S`` unpaired electrons need ``2S`` electrons to occupy
and the rest have to pair up, so an impossible combination such as a
closed-shell count in a doublet is refused here rather than sent to the SCF to
fail hours later. Configs state charge and multiplicity explicitly, so an
impossible pair is a mistake in the config -- an error, not a rejected candidate.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.errors import ValidationError
from ..core.spin import electron_count, multiplicity_to_spin_2s, validate_electron_spin_parity
from ..schemas.label_record import ElectronicState, Structure

__all__ = ["expand_states"]


def expand_states(
    structure: Structure, states: Sequence[tuple[int, int]]
) -> list[ElectronicState]:
    """Return one validated :class:`ElectronicState` per ``(charge, multiplicity)`` pair.

    The list must be non-empty and free of repeats: the same state twice would
    produce two candidates describing one calculation, which is a duplicate the
    manifest could not even name apart. ``spin_2s`` is derived from the
    multiplicity, never taken as input.
    """
    pairs = list(states)
    if not pairs:
        raise ValidationError("states must list at least one (charge, multiplicity) pair.")
    seen: set[tuple[int, int]] = set()
    expanded: list[ElectronicState] = []
    for index, pair in enumerate(pairs):
        path = f"states[{index}]"
        if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValidationError(f"{path} must be a (charge, multiplicity) pair; got {pair!r}.")
        charge, multiplicity = pair
        for name, value in (("charge", charge), ("multiplicity", multiplicity)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValidationError(f"{path}.{name} must be an integer; got {value!r}.")
        if (charge, multiplicity) in seen:
            raise ValidationError(
                f"{path} repeats the state charge={charge}, multiplicity={multiplicity}; "
                "one geometry is expanded into each state at most once."
            )
        seen.add((charge, multiplicity))
        validate_electron_spin_parity(
            electron_count(structure.atomic_numbers, charge), multiplicity
        )
        expanded.append(
            ElectronicState(
                charge=charge,
                multiplicity=multiplicity,
                spin_2s=multiplicity_to_spin_2s(multiplicity),
                state_provenance="state_expansion",
            )
        )
    return expanded
