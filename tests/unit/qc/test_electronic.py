"""The electronic-structure checks, one record and one config section at a time.

Every check is exercised through its own function rather than through a whole
QC run, because that is how the thresholds are meant to be readable: a check is
a pure function of a record and the section of the config that names its
condition, and it returns a verdict rather than raising one.
"""

from __future__ import annotations

import json
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.qc.electronic import (
    ELECTRONIC_CHECK_NAMES,
    check_converged,
    check_gradient_max_component,
    check_gradient_norm,
    check_s2_deviation,
    electronic_checks,
    gradient_max_abs,
    gradient_norm,
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
    "require_converged": True,
    "s2_max_abs_deviation": 0.05,
    "require_s2_for_open_shell": True,
    "gradient_max_abs_hartree_per_bohr": 1.0,
    "gradient_norm_max_hartree_per_bohr": 2.0,
}

Vectors = tuple[tuple[float, float, float], ...]

SMALL_GRADIENT: Vectors = ((0.0, 0.0, -0.01), (0.0, 0.0, 0.01))


def make_record(
    *,
    record_id: str = "h2_a",
    charge: int = 0,
    multiplicity: int = 1,
    gradient: Vectors = SMALL_GRADIENT,
    converged: bool = True,
    s2_deviation: float | None = None,
) -> LabelRecord:
    """Return an H2 label record with the electronic facts a test cares about."""
    return LabelRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74144)),
        ),
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
            gradient_hartree_per_bohr=gradient,
            converged=converged,
            s2=None if s2_deviation is None else 0.75 + s2_deviation,
            s2_target=None if s2_deviation is None else 0.75,
            s2_deviation=s2_deviation,
        ),
        raw=RawArtifact(),
        qc=QcState(status="pending"),
    )


def section(**overrides: object) -> dict[str, object]:
    """Return the reference electronic section with `overrides` applied."""
    return {**SECTION, **overrides}


class ConvergenceTests(unittest.TestCase):
    def test_a_converged_record_passes(self) -> None:
        check = check_converged(make_record(), SECTION)
        self.assertEqual(check["name"], "converged")
        self.assertTrue(check["passed"])
        self.assertIs(check["observed"], True)
        self.assertIs(check["threshold"], True)

    def test_an_unconverged_record_fails(self) -> None:
        check = check_converged(make_record(converged=False), SECTION)
        self.assertFalse(check["passed"])
        self.assertIs(check["observed"], False)

    def test_the_check_honours_a_section_that_does_not_require_convergence(self) -> None:
        check = check_converged(make_record(converged=False), section(require_converged=False))
        self.assertTrue(check["passed"])
        self.assertIs(check["observed"], False)

    def test_a_section_without_the_flag_is_an_error_not_a_pass(self) -> None:
        broken = {key: value for key, value in SECTION.items() if key != "require_converged"}
        with self.assertRaises(ValidationError) as caught:
            check_converged(make_record(converged=False), broken)
        self.assertIn("electronic.require_converged", str(caught.exception))


class SpinContaminationTests(unittest.TestCase):
    def test_a_singlet_is_skipped_with_nothing_observed(self) -> None:
        check = check_s2_deviation(make_record(multiplicity=1), SECTION)
        self.assertEqual(check["name"], "s2_deviation")
        self.assertTrue(check["passed"])
        self.assertIsNone(check["observed"])
        self.assertEqual(check["threshold"], 0.05)

    def test_a_singlet_is_skipped_even_when_it_reports_a_deviation(self) -> None:
        check = check_s2_deviation(make_record(multiplicity=1, s2_deviation=0.9), SECTION)
        self.assertTrue(check["passed"])
        self.assertIsNone(check["observed"])

    def test_an_open_shell_record_under_the_tolerance_passes(self) -> None:
        check = check_s2_deviation(
            make_record(charge=1, multiplicity=2, s2_deviation=0.02), SECTION
        )
        self.assertTrue(check["passed"])
        self.assertAlmostEqual(float(check["observed"]), 0.02)

    def test_an_open_shell_record_over_the_tolerance_fails(self) -> None:
        check = check_s2_deviation(
            make_record(charge=1, multiplicity=2, s2_deviation=0.2), SECTION
        )
        self.assertFalse(check["passed"])
        self.assertAlmostEqual(float(check["observed"]), 0.2)

    def test_a_negative_deviation_is_compared_by_magnitude(self) -> None:
        check = check_s2_deviation(
            make_record(charge=1, multiplicity=2, s2_deviation=-0.2), SECTION
        )
        self.assertFalse(check["passed"])
        self.assertAlmostEqual(float(check["observed"]), 0.2)

    def test_a_deviation_exactly_at_the_tolerance_passes(self) -> None:
        check = check_s2_deviation(
            make_record(charge=1, multiplicity=2, s2_deviation=0.05), SECTION
        )
        self.assertTrue(check["passed"])

    def test_an_open_shell_record_without_s2_fails_when_it_is_required(self) -> None:
        check = check_s2_deviation(make_record(charge=1, multiplicity=2), SECTION)
        self.assertFalse(check["passed"])
        self.assertEqual(check["observed"], "missing")

    def test_an_open_shell_record_without_s2_passes_when_it_is_not_required(self) -> None:
        check = check_s2_deviation(
            make_record(charge=1, multiplicity=2), section(require_s2_for_open_shell=False)
        )
        self.assertTrue(check["passed"])
        self.assertEqual(check["observed"], "missing")

    def test_a_section_without_the_tolerance_is_an_error(self) -> None:
        broken = {key: value for key, value in SECTION.items() if key != "s2_max_abs_deviation"}
        with self.assertRaises(ValidationError) as caught:
            check_s2_deviation(make_record(charge=1, multiplicity=2), broken)
        self.assertIn("electronic.s2_max_abs_deviation", str(caught.exception))


class GradientMagnitudeTests(unittest.TestCase):
    def test_the_largest_absolute_component_is_reported(self) -> None:
        record = make_record(gradient=((0.0, -0.4, 0.1), (0.2, 0.0, 0.3)))
        self.assertAlmostEqual(gradient_max_abs(record), 0.4)

    def test_the_norm_is_the_frobenius_norm_of_the_matrix(self) -> None:
        record = make_record(gradient=((0.0, 0.0, 3.0), (4.0, 0.0, 0.0)))
        self.assertAlmostEqual(gradient_norm(record), 5.0)

    def test_a_component_exactly_at_the_ceiling_passes(self) -> None:
        record = make_record(gradient=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)))
        check = check_gradient_max_component(record, SECTION)
        self.assertEqual(check["name"], "gradient_max_component")
        self.assertTrue(check["passed"])
        self.assertEqual(check["observed"], 1.0)
        self.assertEqual(check["threshold"], 1.0)

    def test_a_component_just_over_the_ceiling_fails(self) -> None:
        record = make_record(gradient=((0.0, 0.0, 1.0000001), (0.0, 0.0, 0.0)))
        self.assertFalse(check_gradient_max_component(record, SECTION)["passed"])

    def test_a_negative_component_is_compared_by_magnitude(self) -> None:
        record = make_record(gradient=((0.0, 0.0, -1.5), (0.0, 0.0, 0.0)))
        check = check_gradient_max_component(record, SECTION)
        self.assertFalse(check["passed"])
        self.assertEqual(check["observed"], 1.5)

    def test_a_norm_exactly_at_the_ceiling_passes(self) -> None:
        record = make_record(gradient=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)))
        check = check_gradient_norm(record, section(gradient_norm_max_hartree_per_bohr=1.0))
        self.assertEqual(check["name"], "gradient_norm")
        self.assertTrue(check["passed"])
        self.assertEqual(check["observed"], 1.0)

    def test_the_norm_can_fail_while_every_component_passes(self) -> None:
        record = make_record(gradient=((0.0, 0.0, 0.9), (0.0, 0.0, 0.9)))
        self.assertTrue(check_gradient_max_component(record, SECTION)["passed"])
        self.assertFalse(
            check_gradient_norm(record, section(gradient_norm_max_hartree_per_bohr=1.0))["passed"]
        )

    def test_a_section_without_a_ceiling_is_an_error(self) -> None:
        broken = {
            key: value
            for key, value in SECTION.items()
            if key != "gradient_norm_max_hartree_per_bohr"
        }
        with self.assertRaises(ValidationError) as caught:
            check_gradient_norm(make_record(), broken)
        self.assertIn("electronic.gradient_norm_max_hartree_per_bohr", str(caught.exception))


class ElectronicCheckSetTests(unittest.TestCase):
    def test_every_check_runs_in_the_declared_order(self) -> None:
        checks = electronic_checks(make_record(), SECTION)
        self.assertEqual(tuple(check["name"] for check in checks), ELECTRONIC_CHECK_NAMES)

    def test_all_checks_run_even_after_one_fails(self) -> None:
        record = make_record(converged=False, gradient=((0.0, 0.0, 9.0), (0.0, 0.0, 0.0)))
        checks = electronic_checks(record, SECTION)
        failed = [check["name"] for check in checks if not check["passed"]]
        self.assertEqual(failed, ["converged", "gradient_max_component", "gradient_norm"])

    def test_every_check_is_json_safe(self) -> None:
        checks = electronic_checks(make_record(charge=1, multiplicity=2), SECTION)
        for check in checks:
            with self.subTest(check=check["name"]):
                self.assertEqual(sorted(check), ["name", "observed", "passed", "threshold"])
                self.assertEqual(json.loads(json.dumps(check)), check)


if __name__ == "__main__":
    unittest.main()
