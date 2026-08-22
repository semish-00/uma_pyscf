"""Pure geometry operations on a structure.

The test molecule is the same tetrahedral SiH4 the reference config uses, built
here from literals so the tests do not depend on the committed XYZ file.
"""

from __future__ import annotations

import math
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.sampling.geometry import gaussian_displacement, scale_bond
from uma_pyscf.schemas.label_record import Structure

BOND = 1.480
COMPONENT = BOND / math.sqrt(3.0)


def sih4() -> Structure:
    """Return tetrahedral SiH4 with Si at the origin and a 1.480 angstrom bond."""
    directions = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
    return Structure(
        atomic_numbers=(14, 1, 1, 1, 1),
        positions_angstrom=(
            (0.0, 0.0, 0.0),
            *((COMPONENT * x, COMPONENT * y, COMPONENT * z) for x, y, z in directions),
        ),
    )


def separation(structure: Structure, left: int, right: int) -> float:
    """Return the distance between two atoms of ``structure``."""
    return math.dist(structure.positions_angstrom[left], structure.positions_angstrom[right])


class ScaleBondTests(unittest.TestCase):
    def test_only_the_target_distance_changes(self) -> None:
        base = sih4()
        scaled = scale_bond(base, 0, 1, 1.15)
        self.assertAlmostEqual(separation(scaled, 0, 1), BOND * 1.15, places=12)
        for index in (2, 3, 4):
            with self.subTest(atom=index):
                self.assertEqual(scaled.positions_angstrom[index], base.positions_angstrom[index])
        self.assertEqual(scaled.positions_angstrom[0], base.positions_angstrom[0])

    def test_the_moved_atom_stays_on_the_bond_axis(self) -> None:
        scaled = scale_bond(sih4(), 0, 1, 0.85)
        moved = scaled.positions_angstrom[1]
        original = sih4().positions_angstrom[1]
        for axis in range(3):
            with self.subTest(axis=axis):
                self.assertAlmostEqual(moved[axis], original[axis] * 0.85, places=12)

    def test_an_off_origin_anchor_is_respected(self) -> None:
        base = Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((1.0, 2.0, 3.0), (1.0, 2.0, 4.0)),
        )
        scaled = scale_bond(base, 0, 1, 2.0)
        self.assertEqual(scaled.positions_angstrom[0], (1.0, 2.0, 3.0))
        self.assertEqual(scaled.positions_angstrom[1], (1.0, 2.0, 5.0))

    def test_the_anchor_may_be_the_second_atom(self) -> None:
        base = Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
        )
        scaled = scale_bond(base, 1, 0, 0.5)
        self.assertEqual(scaled.positions_angstrom[1], (0.0, 0.0, 2.0))
        self.assertEqual(scaled.positions_angstrom[0], (0.0, 0.0, 1.0))

    def test_the_input_structure_is_not_modified(self) -> None:
        base = sih4()
        positions = base.positions_angstrom
        scale_bond(base, 0, 1, 1.5)
        self.assertEqual(base.positions_angstrom, positions)

    def test_provenance_and_elements_are_carried_over(self) -> None:
        base = Structure(
            atomic_numbers=(14, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 1.48)),
            parent_structure_id="sih4_seed",
            sampling_method="bond_scan",
            random_seed=7,
        )
        scaled = scale_bond(base, 0, 1, 1.1)
        self.assertEqual(scaled.atomic_numbers, (14, 1))
        self.assertEqual(scaled.parent_structure_id, "sih4_seed")
        self.assertEqual(scaled.sampling_method, "bond_scan")
        self.assertEqual(scaled.random_seed, 7)

    def test_identical_indices_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            scale_bond(sih4(), 1, 1, 1.1)

    def test_out_of_range_indices_are_rejected(self) -> None:
        for anchor, moved in ((0, 5), (5, 0), (-1, 0), (0, -1)):
            with self.subTest(anchor=anchor, moved=moved):
                with self.assertRaises(ValidationError):
                    scale_bond(sih4(), anchor, moved, 1.1)

    def test_the_error_names_the_offending_argument(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            scale_bond(sih4(), 0, 9, 1.1)
        self.assertIn("moved_index", str(caught.exception))

    def test_non_integer_indices_are_rejected(self) -> None:
        values: tuple[object, ...] = (1.0, "1", None, True)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    scale_bond(sih4(), 0, value, 1.1)  # type: ignore[arg-type]

    def test_non_positive_factors_are_rejected(self) -> None:
        for factor in (0.0, -1.0, -0.5):
            with self.subTest(factor=factor):
                with self.assertRaises(ValidationError):
                    scale_bond(sih4(), 0, 1, factor)

    def test_non_numeric_factors_are_rejected(self) -> None:
        values: tuple[object, ...] = ("1.1", None, True)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    scale_bond(sih4(), 0, 1, value)  # type: ignore[arg-type]


class GaussianDisplacementTests(unittest.TestCase):
    def test_the_same_seed_reproduces_the_same_geometry(self) -> None:
        first = gaussian_displacement(sih4(), 0.04, 20260813)
        second = gaussian_displacement(sih4(), 0.04, 20260813)
        self.assertEqual(first.positions_angstrom, second.positions_angstrom)

    def test_the_next_seed_gives_a_different_geometry(self) -> None:
        first = gaussian_displacement(sih4(), 0.04, 20260813)
        second = gaussian_displacement(sih4(), 0.04, 20260814)
        self.assertNotEqual(first.positions_angstrom, second.positions_angstrom)

    def test_every_atom_actually_moves(self) -> None:
        base = sih4()
        displaced = gaussian_displacement(base, 0.04, 1)
        for index in range(base.atom_count):
            with self.subTest(atom=index):
                self.assertNotEqual(
                    displaced.positions_angstrom[index], base.positions_angstrom[index]
                )

    def test_removing_translation_leaves_the_centroid_where_it_was(self) -> None:
        base = sih4()
        displaced = gaussian_displacement(base, 0.12, 4242, remove_translation=True)
        for axis in range(3):
            with self.subTest(axis=axis):
                before = sum(row[axis] for row in base.positions_angstrom)
                after = sum(row[axis] for row in displaced.positions_angstrom)
                self.assertAlmostEqual(
                    after / base.atom_count, before / base.atom_count, places=12
                )

    def test_keeping_translation_shifts_the_centroid(self) -> None:
        base = sih4()
        displaced = gaussian_displacement(base, 0.12, 4242, remove_translation=False)
        shifts = [
            sum(row[axis] for row in displaced.positions_angstrom)
            - sum(row[axis] for row in base.positions_angstrom)
            for axis in range(3)
        ]
        self.assertTrue(any(abs(shift) > 1e-9 for shift in shifts))

    def test_the_input_structure_is_not_modified(self) -> None:
        base = sih4()
        positions = base.positions_angstrom
        gaussian_displacement(base, 0.04, 1)
        self.assertEqual(base.positions_angstrom, positions)

    def test_provenance_and_elements_are_carried_over(self) -> None:
        base = Structure(
            atomic_numbers=(14, 1, 1, 1, 1),
            positions_angstrom=sih4().positions_angstrom,
            parent_structure_id="sih4_seed",
            sampling_method="cartesian_displacement",
            random_seed=3,
        )
        displaced = gaussian_displacement(base, 0.04, 11)
        self.assertEqual(displaced.atomic_numbers, base.atomic_numbers)
        self.assertEqual(displaced.parent_structure_id, "sih4_seed")
        self.assertEqual(displaced.random_seed, 3)

    def test_a_wider_sigma_displaces_further(self) -> None:
        base = sih4()
        narrow = gaussian_displacement(base, 0.01, 99)
        wide = gaussian_displacement(base, 0.50, 99)
        narrow_shift = max(
            math.dist(row, other)
            for row, other in zip(narrow.positions_angstrom, base.positions_angstrom, strict=True)
        )
        wide_shift = max(
            math.dist(row, other)
            for row, other in zip(wide.positions_angstrom, base.positions_angstrom, strict=True)
        )
        self.assertGreater(wide_shift, narrow_shift)

    def test_non_positive_sigma_is_rejected(self) -> None:
        for sigma in (0.0, -0.04):
            with self.subTest(sigma=sigma):
                with self.assertRaises(ValidationError):
                    gaussian_displacement(sih4(), sigma, 1)

    def test_a_non_integer_seed_is_rejected(self) -> None:
        values: tuple[object, ...] = (1.0, "1", None, True)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    gaussian_displacement(sih4(), 0.04, value)  # type: ignore[arg-type]

    def test_a_non_boolean_translation_flag_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            gaussian_displacement(sih4(), 0.04, 1, remove_translation=1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
