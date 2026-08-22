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
    "MAX_ATOMIC_NUMBER",
    "PERIODIC_SYMBOLS",
    "atomic_number",
    "canonical_symbol",
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
