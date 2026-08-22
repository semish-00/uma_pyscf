"""Element symbols and atomic numbers.

The expected values are hard-coded rather than imported from the Part I
validation experiment: the package must not depend on `validation/`, so the
agreement between the two tables is pinned by literals here.
"""

from __future__ import annotations

import unittest

from uma_pyscf.core.elements import (
    ATOMIC_NUMBERS,
    MAX_ATOMIC_NUMBER,
    PERIODIC_SYMBOLS,
    atomic_number,
    canonical_symbol,
)
from uma_pyscf.core.errors import ValidationError


class TableTests(unittest.TestCase):
    def test_table_covers_element_1_through_118(self) -> None:
        self.assertEqual(len(PERIODIC_SYMBOLS), 119)
        self.assertEqual(PERIODIC_SYMBOLS[0], "")
        self.assertEqual(MAX_ATOMIC_NUMBER, 118)
        self.assertEqual(len(ATOMIC_NUMBERS), 118)

    def test_index_equals_atomic_number(self) -> None:
        for number, symbol in enumerate(PERIODIC_SYMBOLS):
            if symbol:
                with self.subTest(symbol=symbol):
                    self.assertEqual(ATOMIC_NUMBERS[symbol], number)

    def test_symbols_are_unique(self) -> None:
        symbols = [symbol for symbol in PERIODIC_SYMBOLS if symbol]
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_project_elements_are_present(self) -> None:
        for symbol, number in (("H", 1), ("Si", 14), ("Cl", 17), ("Ge", 32)):
            with self.subTest(symbol=symbol):
                self.assertEqual(PERIODIC_SYMBOLS[number], symbol)


class AtomicNumberTests(unittest.TestCase):
    def test_known_elements(self) -> None:
        for symbol, number in (
            ("H", 1),
            ("Si", 14),
            ("Cl", 17),
            ("Ge", 32),
            ("Og", 118),
        ):
            with self.subTest(symbol=symbol):
                self.assertEqual(atomic_number(symbol), number)

    def test_lookup_canonicalizes_first(self) -> None:
        self.assertEqual(atomic_number("cl "), 17)
        self.assertEqual(atomic_number(" GE"), 32)

    def test_unknown_symbol_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            atomic_number("Xx")


class CanonicalSymbolTests(unittest.TestCase):
    def test_whitespace_and_case_are_normalized(self) -> None:
        for raw, expected in (
            ("cl ", "Cl"),
            ("  CL", "Cl"),
            ("h", "H"),
            ("si", "Si"),
            ("gE", "Ge"),
            ("OG", "Og"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(canonical_symbol(raw), expected)

    def test_canonical_symbols_are_returned_unchanged(self) -> None:
        for symbol in ("H", "He", "Si", "Cl", "Ge", "Og"):
            with self.subTest(symbol=symbol):
                self.assertEqual(canonical_symbol(symbol), symbol)

    def test_unknown_symbols_are_rejected(self) -> None:
        for raw in ("Xx", "", "  ", "Silicon", "Q", "H2"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValidationError):
                    canonical_symbol(raw)

    def test_error_names_the_offending_symbol(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            canonical_symbol("Xx")
        self.assertIn("'Xx'", str(caught.exception))

    def test_non_string_input_is_rejected(self) -> None:
        values: tuple[object, ...] = (14, None, ["H"], 1.0)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    canonical_symbol(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
