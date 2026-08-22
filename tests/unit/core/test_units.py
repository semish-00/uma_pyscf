"""Unit constants and magnitude conversions.

The constants are compared against literals written out here on purpose. Part I
results were produced with these exact values, so a silent edit on either side
has to fail this test rather than shift labels.
"""

from __future__ import annotations

import unittest

from uma_pyscf.core import units


class ConstantTests(unittest.TestCase):
    def test_constants_match_the_values_used_by_part_one(self) -> None:
        self.assertEqual(units.BOHR_TO_ANGSTROM, 0.529177210903)
        self.assertEqual(units.HARTREE_TO_EV, 27.211386245988)
        self.assertEqual(units.HARTREE_TO_KCAL_MOL, 627.5094740631)
        self.assertEqual(units.HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, 51.4220674763)

    def test_gradient_constant_is_consistent_with_energy_and_length(self) -> None:
        derived = units.HARTREE_TO_EV / units.BOHR_TO_ANGSTROM
        self.assertAlmostEqual(derived / units.HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, 1.0, places=9)


class ConversionTests(unittest.TestCase):
    def test_single_unit_conversions(self) -> None:
        self.assertAlmostEqual(units.bohr_to_angstrom(1.0), 0.529177210903, places=12)
        self.assertAlmostEqual(units.hartree_to_ev(1.0), 27.211386245988, places=12)
        self.assertAlmostEqual(units.hartree_to_kcal_mol(1.0), 627.5094740631, places=9)
        self.assertAlmostEqual(
            units.hartree_per_bohr_to_ev_per_angstrom(1.0), 51.4220674763, places=9
        )

    def test_conversions_are_linear(self) -> None:
        self.assertAlmostEqual(units.hartree_to_ev(0.0), 0.0, places=15)
        self.assertAlmostEqual(units.hartree_to_ev(-2.5), -2.5 * units.HARTREE_TO_EV, places=12)

    def test_round_trips(self) -> None:
        pairs = (
            (units.bohr_to_angstrom, units.angstrom_to_bohr),
            (units.hartree_to_ev, units.ev_to_hartree),
            (units.hartree_to_kcal_mol, units.kcal_mol_to_hartree),
            (
                units.hartree_per_bohr_to_ev_per_angstrom,
                units.ev_per_angstrom_to_hartree_per_bohr,
            ),
        )
        for forward, backward in pairs:
            for value in (0.0, 1.0, -3.25, 1.2345678901234e-3):
                with self.subTest(forward=forward.__name__, value=value):
                    self.assertAlmostEqual(backward(forward(value)), value, delta=1e-12)
                    self.assertAlmostEqual(forward(backward(value)), value, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
