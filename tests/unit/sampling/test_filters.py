"""Collision, fragment, and duplicate filters.

Distances are chosen against the Cordero radii the package tabulates (H 0.31,
Si 1.11), and the expected cutoffs are written out in the tests so a change to
either the radii or the factor convention shows up here.
"""

from __future__ import annotations

import math
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.sampling.filters import (
    fragment_count,
    is_duplicate,
    minimum_distance_violation,
    pair_distance_fingerprint,
)
from uma_pyscf.schemas.label_record import Structure

BOND = 1.480
COMPONENT = BOND / math.sqrt(3.0)


def sih4(bond: float = BOND) -> Structure:
    """Return tetrahedral SiH4 with the given Si-H bond length."""
    component = bond / math.sqrt(3.0)
    directions = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
    return Structure(
        atomic_numbers=(14, 1, 1, 1, 1),
        positions_angstrom=(
            (0.0, 0.0, 0.0),
            *((component * x, component * y, component * z) for x, y, z in directions),
        ),
    )


def h2_pair(separation: float) -> Structure:
    """Return two H2 molecules whose centres are ``separation`` angstrom apart."""
    return Structure(
        atomic_numbers=(1, 1, 1, 1),
        positions_angstrom=(
            (0.0, 0.0, 0.0),
            (0.74, 0.0, 0.0),
            (0.0, 0.0, separation),
            (0.74, 0.0, separation),
        ),
    )


def rotated(structure: Structure, angle: float) -> Structure:
    """Return ``structure`` rotated about z and translated, by hand."""
    cos, sin = math.cos(angle), math.sin(angle)
    return Structure(
        atomic_numbers=structure.atomic_numbers,
        positions_angstrom=tuple(
            (
                cos * x - sin * y + 3.5,
                sin * x + cos * y - 1.25,
                z + 10.0,
            )
            for x, y, z in structure.positions_angstrom
        ),
    )


class MinimumDistanceTests(unittest.TestCase):
    def test_a_reasonable_molecule_has_no_violation(self) -> None:
        self.assertIsNone(minimum_distance_violation(sih4(), 0.65))

    def test_a_collision_is_reported_with_both_atoms(self) -> None:
        # 0.65 * (1.11 + 0.31) = 0.923 angstrom.
        structure = Structure(
            atomic_numbers=(14, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.90)),
        )
        violation = minimum_distance_violation(structure, 0.65)
        assert violation is not None
        self.assertEqual(violation["atom_indices"], [0, 1])
        self.assertEqual(violation["symbols"], ["Si", "H"])
        self.assertAlmostEqual(violation["distance_angstrom"], 0.90, places=12)
        self.assertAlmostEqual(violation["cutoff_angstrom"], 0.923, places=12)

    def test_a_pair_just_above_the_cutoff_passes(self) -> None:
        structure = Structure(
            atomic_numbers=(14, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.93)),
        )
        self.assertIsNone(minimum_distance_violation(structure, 0.65))

    def test_the_factor_decides(self) -> None:
        structure = Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.50)),
        )
        self.assertIsNone(minimum_distance_violation(structure, 0.65))
        self.assertIsNotNone(minimum_distance_violation(structure, 0.90))

    def test_the_worst_violation_is_the_one_reported(self) -> None:
        structure = Structure(
            atomic_numbers=(1, 1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.36, 0.0, 0.0), (0.36, 0.10, 0.0)),
        )
        violation = minimum_distance_violation(structure, 0.65)
        assert violation is not None
        self.assertEqual(violation["atom_indices"], [1, 2])

    def test_an_untabulated_element_is_an_error_not_a_verdict(self) -> None:
        structure = Structure(
            atomic_numbers=(26, 26),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
        )
        with self.assertRaises(ValidationError):
            minimum_distance_violation(structure, 0.65)

    def test_a_non_positive_factor_is_rejected(self) -> None:
        for factor in (0.0, -0.65):
            with self.subTest(factor=factor):
                with self.assertRaises(ValidationError):
                    minimum_distance_violation(sih4(), factor)


class FragmentCountTests(unittest.TestCase):
    def test_a_bonded_molecule_is_one_fragment(self) -> None:
        self.assertEqual(fragment_count(sih4(), 1.3), 1)

    def test_two_distant_h2_pairs_are_two_fragments(self) -> None:
        self.assertEqual(fragment_count(h2_pair(6.0), 1.3), 2)

    def test_the_same_pairs_close_together_are_one_fragment(self) -> None:
        self.assertEqual(fragment_count(h2_pair(0.75), 1.3), 1)

    def test_a_single_atom_is_one_fragment(self) -> None:
        lone = Structure(atomic_numbers=(1,), positions_angstrom=((0.0, 0.0, 0.0),))
        self.assertEqual(fragment_count(lone, 1.3), 1)

    def test_a_stretched_bond_splits_once_the_cutoff_is_passed(self) -> None:
        # A Si-H bond of 1.924 angstrom against a cutoff of 1.3 * 1.42 = 1.846.
        stretched = Structure(
            atomic_numbers=(14, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 1.924)),
        )
        self.assertEqual(fragment_count(stretched, 1.3), 2)
        self.assertEqual(fragment_count(stretched, 1.45), 1)

    def test_a_chain_counts_as_one_fragment(self) -> None:
        chain = Structure(
            atomic_numbers=(1, 1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.74, 0.0, 0.0), (1.48, 0.0, 0.0)),
        )
        self.assertEqual(fragment_count(chain, 1.3), 1)

    def test_a_non_positive_factor_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            fragment_count(sih4(), 0.0)


class FingerprintTests(unittest.TestCase):
    def test_a_rotated_and_translated_copy_has_the_same_fingerprint(self) -> None:
        base = sih4()
        moved = rotated(base, math.radians(37.0))
        self.assertEqual(pair_distance_fingerprint(base, 3), pair_distance_fingerprint(moved, 3))
        self.assertTrue(is_duplicate(base, moved, 3))

    def test_a_scaled_bond_changes_the_fingerprint(self) -> None:
        self.assertNotEqual(
            pair_distance_fingerprint(sih4(), 3), pair_distance_fingerprint(sih4(1.6), 3)
        )
        self.assertFalse(is_duplicate(sih4(), sih4(1.6), 3))

    def test_the_entries_are_symbol_pairs_and_rounded_distances(self) -> None:
        structure = Structure(
            atomic_numbers=(14, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 1.4804999)),
        )
        self.assertEqual(pair_distance_fingerprint(structure, 3), ((("H", "Si"), 1.48),))

    def test_the_pair_order_does_not_depend_on_atom_order(self) -> None:
        left = Structure(
            atomic_numbers=(14, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 1.48)),
        )
        right = Structure(
            atomic_numbers=(1, 14),
            positions_angstrom=((0.0, 0.0, 1.48), (0.0, 0.0, 0.0)),
        )
        self.assertEqual(pair_distance_fingerprint(left, 3), pair_distance_fingerprint(right, 3))

    def test_decimals_set_how_close_counts_as_the_same(self) -> None:
        near = sih4(BOND + 0.0004)
        self.assertTrue(is_duplicate(sih4(), near, 3))
        self.assertFalse(is_duplicate(sih4(), near, 6))

    def test_different_compositions_are_never_duplicates(self) -> None:
        silicon = Structure(atomic_numbers=(14,), positions_angstrom=((0.0, 0.0, 0.0),))
        hydrogen = Structure(atomic_numbers=(1,), positions_angstrom=((0.0, 0.0, 0.0),))
        self.assertEqual(pair_distance_fingerprint(silicon, 3), ())
        self.assertEqual(pair_distance_fingerprint(hydrogen, 3), ())
        self.assertFalse(is_duplicate(silicon, hydrogen, 3))

    def test_a_permuted_atom_order_of_one_structure_is_a_duplicate(self) -> None:
        base = sih4()
        positions = list(base.positions_angstrom)
        permuted = Structure(
            atomic_numbers=(14, 1, 1, 1, 1),
            positions_angstrom=(
                positions[0],
                positions[3],
                positions[1],
                positions[4],
                positions[2],
            ),
        )
        self.assertTrue(is_duplicate(base, permuted, 3))

    def test_negative_or_non_integer_decimals_are_rejected(self) -> None:
        values: tuple[object, ...] = (-1, 1.0, "3", None, True)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    pair_distance_fingerprint(sih4(), value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
