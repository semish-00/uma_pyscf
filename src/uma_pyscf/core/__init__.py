"""Cross-cutting primitives: units, spin, identifiers, atomic I/O, and errors.

``core`` is the bottom of the one-way dependency chain
``core -> schemas -> (sampling | calculators | qc | datasets | ...) -> cli``.
It depends on nothing inside the package and on the standard library only, so
any constant or conversion that more than one module needs belongs here rather
than being duplicated sideways.
"""

from __future__ import annotations

from .errors import ConfigError, ProvenanceError, UmaPyscfError, ValidationError
from .ids import (
    CASE_ID_PATTERN,
    canonical_json_fingerprint,
    sha256_of_file,
    validate_record_id,
)
from .io import read_json, write_json_atomic, write_text_atomic
from .spin import (
    electron_count,
    multiplicity_to_spin_2s,
    spin_2s_to_multiplicity,
    target_s2,
    validate_electron_spin_parity,
)
from .units import (
    BOHR_TO_ANGSTROM,
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    angstrom_to_bohr,
    bohr_to_angstrom,
    ev_per_angstrom_to_hartree_per_bohr,
    ev_to_hartree,
    hartree_per_bohr_to_ev_per_angstrom,
    hartree_to_ev,
    hartree_to_kcal_mol,
    kcal_mol_to_hartree,
)

__all__ = [
    "BOHR_TO_ANGSTROM",
    "CASE_ID_PATTERN",
    "HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_MOL",
    "ConfigError",
    "ProvenanceError",
    "UmaPyscfError",
    "ValidationError",
    "angstrom_to_bohr",
    "bohr_to_angstrom",
    "canonical_json_fingerprint",
    "electron_count",
    "ev_per_angstrom_to_hartree_per_bohr",
    "ev_to_hartree",
    "hartree_per_bohr_to_ev_per_angstrom",
    "hartree_to_ev",
    "hartree_to_kcal_mol",
    "kcal_mol_to_hartree",
    "multiplicity_to_spin_2s",
    "read_json",
    "sha256_of_file",
    "spin_2s_to_multiplicity",
    "target_s2",
    "validate_electron_spin_parity",
    "validate_record_id",
    "write_json_atomic",
    "write_text_atomic",
]
