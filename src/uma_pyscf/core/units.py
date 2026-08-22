"""CODATA unit constants and magnitude-only conversions.

Canonical records keep the calculation-native units (energy in Hartree,
gradients in Hartree/Bohr, coordinates in Angstrom) and name the unit in every
key. These helpers convert magnitudes only: the ``gradient`` to ``forces`` sign
inversion belongs to the dataset export layer and happens in exactly one place
there, never here.

The constants are the values already used by the Part I validation experiment,
so numbers produced before and after the port agree bit for bit.
"""

from __future__ import annotations

__all__ = [
    "BOHR_TO_ANGSTROM",
    "HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_MOL",
    "angstrom_to_bohr",
    "bohr_to_angstrom",
    "ev_per_angstrom_to_hartree_per_bohr",
    "ev_to_hartree",
    "hartree_per_bohr_to_ev_per_angstrom",
    "hartree_to_ev",
    "hartree_to_kcal_mol",
    "kcal_mol_to_hartree",
]

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.4220674763


def bohr_to_angstrom(value: float) -> float:
    """Convert a length from Bohr to Angstrom."""
    return value * BOHR_TO_ANGSTROM


def angstrom_to_bohr(value: float) -> float:
    """Convert a length from Angstrom to Bohr."""
    return value / BOHR_TO_ANGSTROM


def hartree_to_ev(value: float) -> float:
    """Convert an energy from Hartree to eV."""
    return value * HARTREE_TO_EV


def ev_to_hartree(value: float) -> float:
    """Convert an energy from eV to Hartree."""
    return value / HARTREE_TO_EV


def hartree_to_kcal_mol(value: float) -> float:
    """Convert an energy from Hartree to kcal/mol."""
    return value * HARTREE_TO_KCAL_MOL


def kcal_mol_to_hartree(value: float) -> float:
    """Convert an energy from kcal/mol to Hartree."""
    return value / HARTREE_TO_KCAL_MOL


def hartree_per_bohr_to_ev_per_angstrom(value: float) -> float:
    """Convert a gradient or force magnitude from Hartree/Bohr to eV/Angstrom."""
    return value * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM


def ev_per_angstrom_to_hartree_per_bohr(value: float) -> float:
    """Convert a gradient or force magnitude from eV/Angstrom to Hartree/Bohr."""
    return value / HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
