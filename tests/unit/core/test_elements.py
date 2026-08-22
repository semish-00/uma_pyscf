"""Element symbols and atomic numbers.

The expected values are hard-coded rather than imported from the Part I
validation experiment: the package must not depend on `validation/`, so the
agreement between the two tables is pinned by literals here.
"""

from __future__ import annotations

import unittest

from uma_pyscf.core.elements import (
    ATOMIC_NUMBERS,
    COVALENT_RADII_ANGSTROM,
    MAX_ATOMIC_NUMBER,
    PERIODIC_SYMBOLS,
    atomic_number,
    canonical_symbol,
    covalent_radius,
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


class CovalentRadiusTests(unittest.TestCase):
    """The Cordero (2008) radii the geometry filters are built on.

    The values are pinned as literals here for the same reason as the symbol
    table: `validation/` uses the same four numbers for H, Si, Ge, and Cl, and
    the agreement has to survive without importing anything from it.
    """

    def test_the_tabulated_values(self) -> None:
        expected = {
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
        self.assertEqual(COVALENT_RADII_ANGSTROM, expected)

    def test_the_table_covers_only_the_project_elements(self) -> None:
        self.assertEqual(len(COVALENT_RADII_ANGSTROM), 14)
        for symbol in COVALENT_RADII_ANGSTROM:
            with self.subTest(symbol=symbol):
                self.assertEqual(canonical_symbol(symbol), symbol)

    def test_lookup_by_symbol(self) -> None:
        for symbol, radius in (("H", 0.31), ("Si", 1.11), ("Cl", 1.02), ("Ge", 1.20)):
            with self.subTest(symbol=symbol):
                self.assertEqual(covalent_radius(symbol), radius)

    def test_lookup_by_atomic_number(self) -> None:
        for number, radius in ((1, 0.31), (14, 1.11), (17, 1.02), (32, 1.20)):
            with self.subTest(number=number):
                self.assertEqual(covalent_radius(number), radius)

    def test_lookup_canonicalizes_the_symbol_first(self) -> None:
        self.assertEqual(covalent_radius("cl "), 1.02)
        self.assertEqual(covalent_radius(" GE"), 1.20)

    def test_an_element_outside_the_table_fails_closed(self) -> None:
        for value in ("Fe", "Xe", 26, 54):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    covalent_radius(value)

    def test_the_error_names_the_element_and_the_source(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            covalent_radius("Fe")
        message = str(caught.exception)
        self.assertIn("'Fe'", message)
        self.assertIn("Cordero", message)

    def test_an_unknown_symbol_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            covalent_radius("Xx")

    def test_an_out_of_range_atomic_number_is_rejected(self) -> None:
        for number in (0, -1, MAX_ATOMIC_NUMBER + 1):
            with self.subTest(number=number):
                with self.assertRaises(ValidationError):
                    covalent_radius(number)

    def test_a_non_element_input_is_rejected(self) -> None:
        values: tuple[object, ...] = (None, 1.0, True, ["H"])
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    covalent_radius(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
