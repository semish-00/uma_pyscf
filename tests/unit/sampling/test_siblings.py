"""Charge and spin sibling expansion.

SiH4 has 18 electrons, so the neutral molecule can only be a singlet (or a
triplet), and the cation with 17 electrons can only be a doublet (or a quartet).
Those counts are what the parity check is made of.
"""

from __future__ import annotations

import math
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.sampling.siblings import expand_states
from uma_pyscf.schemas.label_record import Structure

COMPONENT = 1.480 / math.sqrt(3.0)


def sih4() -> Structure:
    """Return tetrahedral SiH4, 18 electrons when neutral."""
    directions = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
    return Structure(
        atomic_numbers=(14, 1, 1, 1, 1),
        positions_angstrom=(
            (0.0, 0.0, 0.0),
            *((COMPONENT * x, COMPONENT * y, COMPONENT * z) for x, y, z in directions),
        ),
    )


class ExpandStatesTests(unittest.TestCase):
    def test_valid_states_become_electronic_states(self) -> None:
        states = expand_states(sih4(), [(0, 1), (1, 2)])
        self.assertEqual(
            [(state.charge, state.multiplicity) for state in states], [(0, 1), (1, 2)]
        )

    def test_spin_2s_is_derived_from_the_multiplicity(self) -> None:
        states = expand_states(sih4(), [(0, 1), (0, 3), (1, 2)])
        self.assertEqual([state.spin_2s for state in states], [0, 2, 1])

    def test_the_expansion_records_where_the_state_came_from(self) -> None:
        state = expand_states(sih4(), [(0, 1)])[0]
        self.assertEqual(state.state_provenance, "state_expansion")

    def test_the_requested_order_is_kept(self) -> None:
        states = expand_states(sih4(), [(1, 2), (0, 1), (-1, 2)])
        self.assertEqual([state.charge for state in states], [1, 0, -1])

    def test_an_even_electron_count_cannot_be_a_doublet(self) -> None:
        with self.assertRaises(ValidationError):
            expand_states(sih4(), [(0, 2)])

    def test_an_odd_electron_count_cannot_be_a_singlet(self) -> None:
        with self.assertRaises(ValidationError):
            expand_states(sih4(), [(1, 1)])

    def test_the_parity_error_names_the_multiplicity(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            expand_states(sih4(), [(0, 2)])
        self.assertIn("multiplicity 2", str(caught.exception))

    def test_a_repeated_state_is_rejected(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            expand_states(sih4(), [(0, 1), (1, 2), (0, 1)])
        self.assertIn("states[2]", str(caught.exception))

    def test_an_empty_list_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            expand_states(sih4(), [])

    def test_a_malformed_pair_is_rejected(self) -> None:
        values: tuple[object, ...] = ((0,), (0, 1, 2), "01", 0, None)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    expand_states(sih4(), [value])  # type: ignore[list-item]

    def test_non_integer_charge_or_multiplicity_is_rejected(self) -> None:
        values: tuple[tuple[object, object], ...] = ((0.0, 1), (0, 1.0), (0, True), ("0", 1))
        for pair in values:
            with self.subTest(pair=pair):
                with self.assertRaises(ValidationError):
                    expand_states(sih4(), [pair])  # type: ignore[list-item]

    def test_a_multiplicity_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            expand_states(sih4(), [(0, 0)])

    def test_stripping_every_electron_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            expand_states(sih4(), [(18, 1)])


if __name__ == "__main__":
    unittest.main()
