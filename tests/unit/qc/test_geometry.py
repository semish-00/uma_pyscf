"""The geometry checks, applied to computed records instead of to candidates.

The arithmetic itself belongs to `uma_pyscf.sampling.filters` and is tested
there; what is tested here is the part QC adds -- that a record's structure is
run through those filters with the QC config's factors, and that duplicate
detection is state-qualified, so the same geometry in two electronic states is
two calculations rather than one calculation reported twice.
"""

from __future__ import annotations

import json
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.qc.geometry import (
    GEOMETRY_CHECK_NAMES,
    check_duplicate,
    check_fragments,
    check_minimum_distance,
    duplicate_map,
    geometry_checks,
    state_qualified_key,
)
from uma_pyscf.schemas.label_record import (
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)

SECTION = {
    "covalent_factor": 0.65,
    "bond_factor": 1.3,
    "allow_fragments": False,
    "duplicate_decimals": 3,
}

Vectors = tuple[tuple[float, float, float], ...]

#: A bonded H2 at its equilibrium separation.
H2: Vectors = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74144))
#: Two H atoms far inside the covalent collision cutoff of 0.403 A.
COLLIDED: Vectors = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.2))
#: Two separate H2 molecules, five angstrom apart.
TWO_MOLECULES: Vectors = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.74144),
    (5.0, 0.0, 0.0),
    (5.0, 0.0, 0.74144),
)


def make_record(
    *,
    record_id: str = "h2_a",
    positions: Vectors = H2,
    charge: int = 0,
    multiplicity: int = 1,
) -> LabelRecord:
    """Return a hydrogen-only label record with the geometry a test cares about."""
    atoms = len(positions)
    return LabelRecord(
        record_id=record_id,
        structure=Structure(atomic_numbers=(1,) * atoms, positions_angstrom=positions),
        state=ElectronicState(charge=charge, multiplicity=multiplicity, spin_2s=multiplicity - 1),
        method=Method(
            functional="wb97m-v",
            basis="def2-tzvpd",
            ecp=None,
            aux_basis=None,
            grid_level=3,
            nlc_grid_level=1,
            grid_response=True,
            density_fit=False,
            scf_conv_tol=1e-10,
            scf_max_cycle=200,
        ),
        engine=Engine(name="gpu4pyscf", versions={"pyscf": "2.6.2"}),
        results=Results(
            energy_hartree=-1.1730407,
            gradient_hartree_per_bohr=((0.0, 0.0, 0.01),) * atoms,
            converged=True,
        ),
        raw=RawArtifact(),
        qc=QcState(status="pending"),
    )


def section(**overrides: object) -> dict[str, object]:
    """Return the reference geometry section with `overrides` applied."""
    return {**SECTION, **overrides}


class MinimumDistanceTests(unittest.TestCase):
    def test_a_bonded_molecule_passes_with_nothing_observed(self) -> None:
        check = check_minimum_distance(make_record(), SECTION)
        self.assertEqual(check["name"], "minimum_distance")
        self.assertTrue(check["passed"])
        self.assertIsNone(check["observed"])
        self.assertEqual(check["threshold"], 0.65)

    def test_a_collided_pair_fails_and_names_both_atoms(self) -> None:
        check = check_minimum_distance(make_record(positions=COLLIDED), SECTION)
        self.assertFalse(check["passed"])
        observed = check["observed"]
        self.assertEqual(observed["atom_indices"], [0, 1])
        self.assertEqual(observed["symbols"], ["H", "H"])
        self.assertAlmostEqual(observed["distance_angstrom"], 0.2)
        self.assertAlmostEqual(observed["cutoff_angstrom"], 0.65 * 0.62)

    def test_a_smaller_covalent_factor_can_accept_the_same_geometry(self) -> None:
        check = check_minimum_distance(
            make_record(positions=COLLIDED), section(covalent_factor=0.3)
        )
        self.assertTrue(check["passed"])

    def test_a_section_without_the_factor_is_an_error_not_a_pass(self) -> None:
        broken = {key: value for key, value in SECTION.items() if key != "covalent_factor"}
        with self.assertRaises(ValidationError) as caught:
            check_minimum_distance(make_record(positions=COLLIDED), broken)
        self.assertIn("geometry.covalent_factor", str(caught.exception))


class FragmentTests(unittest.TestCase):
    def test_a_connected_molecule_passes_with_one_fragment(self) -> None:
        check = check_fragments(make_record(), SECTION)
        self.assertEqual(check["name"], "fragments")
        self.assertTrue(check["passed"])
        self.assertEqual(check["observed"], 1)
        self.assertEqual(check["threshold"], {"bond_factor": 1.3, "allow_fragments": False})

    def test_two_fragments_fail_when_fragments_are_not_allowed(self) -> None:
        check = check_fragments(make_record(positions=TWO_MOLECULES), SECTION)
        self.assertFalse(check["passed"])
        self.assertEqual(check["observed"], 2)

    def test_the_same_two_fragments_pass_when_fragments_are_allowed(self) -> None:
        check = check_fragments(
            make_record(positions=TWO_MOLECULES), section(allow_fragments=True)
        )
        self.assertTrue(check["passed"])
        self.assertEqual(check["observed"], 2)
        self.assertEqual(check["threshold"], {"bond_factor": 1.3, "allow_fragments": True})

    def test_a_section_without_the_flag_is_an_error_not_a_pass(self) -> None:
        broken = {key: value for key, value in SECTION.items() if key != "allow_fragments"}
        with self.assertRaises(ValidationError) as caught:
            check_fragments(make_record(positions=TWO_MOLECULES), broken)
        self.assertIn("geometry.allow_fragments", str(caught.exception))


class DuplicateTests(unittest.TestCase):
    def test_the_state_qualified_key_separates_two_charges(self) -> None:
        neutral = make_record(record_id="h2_neutral")
        cation = make_record(record_id="h2_cation", charge=1, multiplicity=2)
        self.assertNotEqual(state_qualified_key(neutral, 3), state_qualified_key(cation, 3))

    def test_the_state_qualified_key_matches_the_same_geometry_and_state(self) -> None:
        first = make_record(record_id="h2_first")
        second = make_record(record_id="h2_second")
        self.assertEqual(state_qualified_key(first, 3), state_qualified_key(second, 3))

    def test_the_later_of_two_identical_records_is_the_duplicate(self) -> None:
        records = [make_record(record_id="h2_first"), make_record(record_id="h2_second")]
        self.assertEqual(duplicate_map(records, SECTION), {"h2_second": "h2_first"})

    def test_a_duplicate_check_names_the_record_that_was_kept(self) -> None:
        check = check_duplicate(SECTION, "h2_first")
        self.assertEqual(check["name"], "duplicate")
        self.assertFalse(check["passed"])
        self.assertEqual(check["observed"], "h2_first")
        self.assertEqual(check["threshold"], 3)

    def test_the_first_record_of_its_kind_is_not_a_duplicate(self) -> None:
        check = check_duplicate(SECTION, None)
        self.assertTrue(check["passed"])
        self.assertIsNone(check["observed"])

    def test_the_same_geometry_in_a_different_charge_is_not_a_duplicate(self) -> None:
        records = [
            make_record(record_id="h2_neutral"),
            make_record(record_id="h2_cation", charge=1, multiplicity=2),
        ]
        self.assertEqual(duplicate_map(records, SECTION), {})

    def test_the_same_geometry_in_a_different_multiplicity_is_not_a_duplicate(self) -> None:
        records = [
            make_record(record_id="h4_singlet", positions=TWO_MOLECULES),
            make_record(record_id="h4_triplet", positions=TWO_MOLECULES, multiplicity=3),
        ]
        self.assertEqual(duplicate_map(records, SECTION), {})

    def test_a_third_copy_points_at_the_first_record_not_the_second(self) -> None:
        records = [make_record(record_id=f"h2_{name}") for name in ("a", "b", "c")]
        self.assertEqual(duplicate_map(records, SECTION), {"h2_b": "h2_a", "h2_c": "h2_a"})

    def test_rounding_decides_how_close_two_geometries_have_to_be(self) -> None:
        records = [
            make_record(record_id="h2_a"),
            make_record(record_id="h2_b", positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.7418))),
        ]
        self.assertEqual(duplicate_map(records, SECTION), {})
        self.assertEqual(duplicate_map(records, section(duplicate_decimals=2)), {"h2_b": "h2_a"})

    def test_a_section_without_the_rounding_is_an_error(self) -> None:
        broken = {key: value for key, value in SECTION.items() if key != "duplicate_decimals"}
        with self.assertRaises(ValidationError) as caught:
            duplicate_map([make_record()], broken)
        self.assertIn("geometry.duplicate_decimals", str(caught.exception))


class GeometryCheckSetTests(unittest.TestCase):
    def test_every_check_runs_in_the_declared_order(self) -> None:
        checks = geometry_checks(make_record(), SECTION, None)
        self.assertEqual(tuple(check["name"] for check in checks), GEOMETRY_CHECK_NAMES)

    def test_all_checks_run_even_after_one_fails(self) -> None:
        checks = geometry_checks(make_record(positions=COLLIDED), SECTION, "h2_first")
        failed = [check["name"] for check in checks if not check["passed"]]
        self.assertEqual(failed, ["minimum_distance", "duplicate"])

    def test_every_check_is_json_safe(self) -> None:
        checks = geometry_checks(make_record(positions=COLLIDED), SECTION, "h2_first")
        for check in checks:
            with self.subTest(check=check["name"]):
                self.assertEqual(sorted(check), ["name", "observed", "passed", "threshold"])
                self.assertEqual(json.loads(json.dumps(check)), check)


if __name__ == "__main__":
    unittest.main()
