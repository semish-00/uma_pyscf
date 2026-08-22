"""Charge and spin bookkeeping with multiplicity as the single source of truth.

Records carry the spin multiplicity ``2S+1``. PySCF's ``spin`` argument is
``n_alpha - n_beta = 2S`` and is always a derived value; the reverse direction
is never accepted as record input. Behaviour matches the Part I validation
experiment so that labels produced before and after the port stay comparable.
"""

from __future__ import annotations

from collections.abc import Sequence

from .errors import ValidationError

__all__ = [
    "electron_count",
    "multiplicity_to_spin_2s",
    "spin_2s_to_multiplicity",
    "target_s2",
    "validate_electron_spin_parity",
]


def _require_int(value: object, name: str) -> int:
    """Return ``value`` as an int, rejecting bools and every non-integer type."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer.")
    return value


def multiplicity_to_spin_2s(multiplicity: int) -> int:
    """Convert a ``2S+1`` multiplicity to PySCF's ``n_alpha - n_beta = 2S``."""
    value = _require_int(multiplicity, "Multiplicity")
    if value < 1:
        raise ValidationError("Multiplicity must be at least 1.")
    return value - 1


def spin_2s_to_multiplicity(spin_2s: int) -> int:
    """Convert a derived ``2S`` back to the ``2S+1`` multiplicity of a record.

    Provided for reporting and round-trip checks only. Manifest and record
    inputs always state the multiplicity.
    """
    value = _require_int(spin_2s, "Spin 2S")
    if value < 0:
        raise ValidationError("Spin 2S must be zero or greater.")
    return value + 1


def target_s2(spin_2s: int) -> float:
    """Return the ideal ``<S^2>`` expectation value ``S(S+1)`` for ``2S``."""
    value = _require_int(spin_2s, "Spin 2S")
    if value < 0:
        raise ValidationError("Spin 2S must be zero or greater.")
    total_spin = value / 2.0
    return total_spin * (total_spin + 1.0)


def electron_count(atomic_numbers: Sequence[int], charge: int) -> int:
    """Return the electron count of a molecule from its atoms and total charge."""
    charge_value = _require_int(charge, "Charge")
    total = 0
    for index, atomic_number in enumerate(atomic_numbers):
        number = _require_int(atomic_number, f"Atomic number at index {index}")
        if number < 1:
            raise ValidationError(f"Atomic number at index {index} must be at least 1.")
        total += number
    count = total - charge_value
    if count < 1:
        raise ValidationError(
            f"Nuclear charge {total} and charge {charge_value} leave {count} electrons; "
            "a molecule must contain at least one electron."
        )
    return count


def validate_electron_spin_parity(electron_count: int, multiplicity: int) -> None:
    """Raise unless the electron count and the multiplicity can describe one state.

    ``2S`` unpaired electrons need at least ``2S`` electrons to occupy, and the
    remaining electrons must pair up, so ``electron_count - 2S`` must be even.
    """
    count = _require_int(electron_count, "Electron count")
    if count < 1:
        raise ValidationError("Electron count must be at least 1.")
    spin_2s = multiplicity_to_spin_2s(multiplicity)
    if spin_2s > count or (count - spin_2s) % 2:
        raise ValidationError(
            f"Electron count {count} and multiplicity {multiplicity} are inconsistent; "
            f"the derived PySCF spin 2S would be {spin_2s}."
        )
