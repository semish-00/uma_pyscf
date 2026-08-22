"""Element symbols and atomic numbers.

Atomic numbers are the canonical way this package names an element: records
store ``atomic_numbers`` and never symbols, so every reader of an external
structure file passes through :func:`canonical_symbol` and :func:`atomic_number`
once, at the boundary. The table is the cross-cutting constant that structure
parsing, sampling, and the schema layer would otherwise each redefine, so it
lives in ``core`` like the unit and spin constants next to it.

The symbol list is identical to the one the Part I validation experiment uses,
which keeps element identity stable across the port.
"""

from __future__ import annotations

from .errors import ValidationError

__all__ = [
    "ATOMIC_NUMBERS",
    "COVALENT_RADII_ANGSTROM",
    "MAX_ATOMIC_NUMBER",
    "PERIODIC_SYMBOLS",
    "atomic_number",
    "canonical_symbol",
    "covalent_radius",
]

# Index equals the atomic number, so element 0 is an empty placeholder.
# fmt: off
PERIODIC_SYMBOLS: tuple[str, ...] = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
# fmt: on

ATOMIC_NUMBERS: dict[str, int] = {
    symbol: number for number, symbol in enumerate(PERIODIC_SYMBOLS) if symbol
}
MAX_ATOMIC_NUMBER = len(PERIODIC_SYMBOLS) - 1


#: Single-bond covalent radii in angstrom, from Cordero et al., "Covalent radii
#: revisited", Dalton Trans. 2008, 2832-2838 (the sp3 value is used for carbon).
#: Only the elements this project actually samples are listed, and the table is
#: deliberately not extended by interpolation or by a second source: a geometry
#: filter that silently uses a made-up radius would accept or reject structures
#: for reasons nobody can reproduce. :func:`covalent_radius` therefore fails
#: closed on every element that is missing here, and adding one means adding the
#: published value. H, Si, Ge, and Cl match the radii the Part I validation
#: experiment used for its minimum-distance filter, so ladder geometries keep
#: the same verdict after the port.
COVALENT_RADII_ANGSTROM: dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
}


def canonical_symbol(raw: str) -> str:
    """Return the canonical spelling of an element symbol.

    Surrounding whitespace is dropped and the capitalization is normalized, so
    ``"cl "`` and ``"CL"`` both become ``"Cl"``. Anything that is not an element
    fails closed rather than being passed on as an unknown label.
    """
    if not isinstance(raw, str):
        raise ValidationError(f"Element symbol must be a string; got {type(raw).__name__}.")
    symbol = raw.strip().capitalize()
    if symbol not in ATOMIC_NUMBERS:
        raise ValidationError(f"Unknown element symbol {raw!r}.")
    return symbol


def atomic_number(symbol: str) -> int:
    """Return the atomic number of an element symbol, canonicalizing it first."""
    return ATOMIC_NUMBERS[canonical_symbol(symbol)]


def covalent_radius(symbol_or_z: str | int) -> float:
    """Return the covalent radius in angstrom of an element symbol or atomic number.

    An element outside :data:`COVALENT_RADII_ANGSTROM` raises
    :class:`~uma_pyscf.core.errors.ValidationError` instead of receiving a
    default. Distance filters built on this value decide which structures reach
    the DFT queue, so an unlisted element has to stop the run and be added to
    the table with its published radius.
    """
    if isinstance(symbol_or_z, bool) or not isinstance(symbol_or_z, str | int):
        raise ValidationError(
            f"Element must be a symbol or an atomic number; got {symbol_or_z!r}."
        )
    if isinstance(symbol_or_z, int):
        if not 1 <= symbol_or_z <= MAX_ATOMIC_NUMBER:
            raise ValidationError(
                f"Atomic number must be from 1 through {MAX_ATOMIC_NUMBER}; got {symbol_or_z}."
            )
        symbol = PERIODIC_SYMBOLS[symbol_or_z]
    else:
        symbol = canonical_symbol(symbol_or_z)
    if symbol not in COVALENT_RADII_ANGSTROM:
        raise ValidationError(
            f"No covalent radius is tabulated for {symbol!r}. The table carries the "
            "Cordero (2008) values for the elements this project samples only; add the "
            "published radius for this element rather than estimating one."
        )
    return COVALENT_RADII_ANGSTROM[symbol]
