"""Charge and spin bookkeeping.

The expected values are hard-coded rather than imported from the Part I
validation experiment: the package must not depend on `validation/`, so the
agreement between the two implementations is pinned by literals here.
"""

from __future__ import annotations

import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.spin import (
    electron_count,
    multiplicity_to_spin_2s,
    spin_2s_to_multiplicity,
    target_s2,
    validate_electron_spin_parity,
)

SILICON = 14
HYDROGEN = 1
CHLORINE = 17


class MultiplicityConversionTests(unittest.TestCase):
    def test_known_multiplicities_match_pyscf_spin(self) -> None:
        for multiplicity, spin_2s in ((1, 0), (2, 1), (3, 2), (4, 3), (7, 6)):
            with self.subTest(multiplicity=multiplicity):
                self.assertEqual(multiplicity_to_spin_2s(multiplicity), spin_2s)
                self.assertEqual(spin_2s_to_multiplicity(spin_2s), multiplicity)

    def test_conversions_round_trip(self) -> None:
        for multiplicity in range(1, 12):
            with self.subTest(multiplicity=multiplicity):
                spin_2s = multiplicity_to_spin_2s(multiplicity)
                self.assertEqual(spin_2s_to_multiplicity(spin_2s), multiplicity)
        for spin_2s in range(0, 12):
            with self.subTest(spin_2s=spin_2s):
                multiplicity = spin_2s_to_multiplicity(spin_2s)
                self.assertEqual(multiplicity_to_spin_2s(multiplicity), spin_2s)

    def test_multiplicity_zero_and_negative_are_rejected(self) -> None:
        for multiplicity in (0, -1, -3):
            with self.subTest(multiplicity=multiplicity):
                with self.assertRaises(ValidationError):
                    multiplicity_to_spin_2s(multiplicity)

    def test_booleans_are_not_accepted_as_integers(self) -> None:
        booleans: tuple[object, ...] = (True, False)
        for value in booleans:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    multiplicity_to_spin_2s(value)  # type: ignore[arg-type]
                with self.assertRaises(ValidationError):
                    spin_2s_to_multiplicity(value)  # type: ignore[arg-type]
                with self.assertRaises(ValidationError):
                    target_s2(value)  # type: ignore[arg-type]

    def test_non_integer_multiplicity_is_rejected(self) -> None:
        values: tuple[object, ...] = (2.0, "2", None, [2])
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    multiplicity_to_spin_2s(value)  # type: ignore[arg-type]

    def test_negative_spin_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            spin_2s_to_multiplicity(-1)
        with self.assertRaises(ValidationError):
            target_s2(-1)


class TargetS2Tests(unittest.TestCase):
    def test_known_expectation_values(self) -> None:
        for spin_2s, expected in ((0, 0.0), (1, 0.75), (2, 2.0), (3, 3.75), (4, 6.0)):
            with self.subTest(spin_2s=spin_2s):
                self.assertAlmostEqual(target_s2(spin_2s), expected, places=12)

    def test_target_s2_follows_from_multiplicity(self) -> None:
        self.assertAlmostEqual(target_s2(multiplicity_to_spin_2s(2)), 0.75, places=12)


class ElectronCountTests(unittest.TestCase):
    def test_neutral_and_charged_silane_fragments(self) -> None:
        sih3 = (SILICON, HYDROGEN, HYDROGEN, HYDROGEN)
        self.assertEqual(electron_count(sih3, 0), 17)
        self.assertEqual(electron_count(sih3, 1), 16)
        self.assertEqual(electron_count(sih3, -1), 18)

    def test_larger_molecule(self) -> None:
        sicl4 = (SILICON, CHLORINE, CHLORINE, CHLORINE, CHLORINE)
        self.assertEqual(electron_count(sicl4, 0), 82)

    def test_invalid_atomic_numbers_are_rejected(self) -> None:
        cases: tuple[object, ...] = ((0,), (-1,), (14, 0), (14.0,), ("Si",), (True,))
        for atomic_numbers in cases:
            with self.subTest(atomic_numbers=atomic_numbers):
                with self.assertRaises(ValidationError):
                    electron_count(atomic_numbers, 0)  # type: ignore[arg-type]

    def test_non_integer_charge_is_rejected(self) -> None:
        charges: tuple[object, ...] = (0.0, "0", True)
        for charge in charges:
            with self.subTest(charge=charge):
                with self.assertRaises(ValidationError):
                    electron_count((SILICON,), charge)  # type: ignore[arg-type]

    def test_molecule_without_electrons_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            electron_count((HYDROGEN,), 1)
        with self.assertRaises(ValidationError):
            electron_count((HYDROGEN,), 5)

    def test_atomic_number_error_names_the_index(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            electron_count((SILICON, 0), 0)
        self.assertIn("index 1", str(caught.exception))


class ElectronSpinParityTests(unittest.TestCase):
    def test_odd_electron_count_accepts_even_multiplicities(self) -> None:
        for multiplicity in (2, 4):
            with self.subTest(multiplicity=multiplicity):
                validate_electron_spin_parity(17, multiplicity)

    def test_odd_electron_count_rejects_odd_multiplicity(self) -> None:
        with self.assertRaises(ValidationError):
            validate_electron_spin_parity(17, 3)

    def test_even_electron_count_accepts_odd_multiplicities(self) -> None:
        for multiplicity in (1, 3):
            with self.subTest(multiplicity=multiplicity):
                validate_electron_spin_parity(16, multiplicity)

    def test_even_electron_count_rejects_even_multiplicity(self) -> None:
        with self.assertRaises(ValidationError):
            validate_electron_spin_parity(16, 2)

    def test_spin_larger_than_electron_count_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            validate_electron_spin_parity(1, 4)
        with self.assertRaises(ValidationError):
            validate_electron_spin_parity(2, 5)

    def test_error_message_states_electrons_multiplicity_and_spin(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            validate_electron_spin_parity(17, 3)
        message = str(caught.exception)
        self.assertIn("17", message)
        self.assertIn("multiplicity 3", message)
        self.assertIn("2S would be 2", message)

    def test_invalid_electron_count_is_rejected(self) -> None:
        counts: tuple[object, ...] = (0, -1, 2.0, True)
        for count in counts:
            with self.subTest(count=count):
                with self.assertRaises(ValidationError):
                    validate_electron_spin_parity(count, 1)  # type: ignore[arg-type]

    def test_invalid_multiplicity_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            validate_electron_spin_parity(16, 0)

    def test_composed_with_electron_count(self) -> None:
        sih3 = (SILICON, HYDROGEN, HYDROGEN, HYDROGEN)
        validate_electron_spin_parity(electron_count(sih3, 0), 2)
        validate_electron_spin_parity(electron_count(sih3, 1), 1)
        with self.assertRaises(ValidationError):
            validate_electron_spin_parity(electron_count(sih3, 1), 2)


if __name__ == "__main__":
    unittest.main()
