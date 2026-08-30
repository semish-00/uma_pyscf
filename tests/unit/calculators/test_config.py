"""Production DFT config, Conditional GO scope, and resource tiers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from uma_pyscf.calculators.config import (
    load_dft_config,
    method_from_config,
    resource_for_candidate,
    scope_violations,
    validate_dft_config,
)
from uma_pyscf.core.errors import ValidationError
from uma_pyscf.schemas.candidate import CandidateRecord
from uma_pyscf.schemas.label_record import ElectronicState, Structure

REPO_ROOT = Path(__file__).resolve().parents[3]
DFT_CONFIG = REPO_ROOT / "configs" / "dft" / "omol_wb97mv_tzvpd_v1.yaml"


def candidate(
    *,
    atomic_numbers: tuple[int, ...] = (1, 1),
    charge: int = 0,
    multiplicity: int = 1,
    state_provenance: str | None = "ground_state_default",
) -> CandidateRecord:
    """Return a geometry/state pair whose shape is valid before scope checks."""
    return CandidateRecord(
        record_id="scope_case",
        structure=Structure(
            atomic_numbers=atomic_numbers,
            positions_angstrom=tuple(
                (float(index), 0.0, 0.0) for index in range(len(atomic_numbers))
            ),
        ),
        state=ElectronicState(
            charge=charge,
            multiplicity=multiplicity,
            spin_2s=multiplicity - 1,
            state_provenance=state_provenance,
        ),
    )


class ProductionConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_dft_config(DFT_CONFIG)

    def test_committed_config_is_the_gate1_protocol(self) -> None:
        self.assertEqual(self.config["protocol_id"], "omol_wb97mv_tzvpd_v1")
        method = method_from_config(self.config)
        self.assertEqual(method.functional, "wb97m-v")
        self.assertEqual(method.basis, "def2-tzvpd")
        self.assertEqual((method.grid_level, method.nlc_grid_level), (5, 5))
        self.assertTrue(method.grid_response)
        self.assertTrue(method.density_fit)
        self.assertEqual(method.scf_conv_tol, 1e-10)
        self.assertEqual(method.scf_max_cycle, 250)

    def test_direct_method_changes_only_density_fitting(self) -> None:
        primary = method_from_config(self.config)
        direct = method_from_config(self.config, density_fit=False)
        primary_dict = primary.to_dict()
        direct_dict = direct.to_dict()
        self.assertNotEqual(primary_dict.pop("density_fit"), direct_dict.pop("density_fit"))
        self.assertEqual(primary_dict, direct_dict)

    def test_release_remains_fail_closed(self) -> None:
        changed = deepcopy(self.config)
        changed["release_controls"]["release_allowed"] = True
        with self.assertRaises(ValidationError) as caught:
            validate_dft_config(changed)
        self.assertIn("release_allowed", str(caught.exception))

    def test_gate1_scope_cannot_be_silently_expanded(self) -> None:
        changed = deepcopy(self.config)
        changed["scope"]["allowed_elements"].append("C")
        with self.assertRaises(ValidationError) as caught:
            validate_dft_config(changed)
        self.assertIn("expands Gate 1", str(caught.exception))

    def test_explicit_minao_is_not_optional(self) -> None:
        changed = deepcopy(self.config)
        changed["initial_density"]["guess"] = "atom"
        with self.assertRaises(ValidationError):
            validate_dft_config(changed)


class ScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_dft_config(DFT_CONFIG)

    def test_default_h2_is_ready(self) -> None:
        self.assertEqual(scope_violations(candidate(), self.config), ())

    def test_an_element_outside_gate1_is_blocked(self) -> None:
        violations = scope_violations(candidate(atomic_numbers=(6, 1, 1, 1, 1)), self.config)
        self.assertIn("atomic_numbers_outside_gate1:6", violations)

    def test_more_than_eight_atoms_is_blocked(self) -> None:
        violations = scope_violations(
            candidate(
                atomic_numbers=(1,) * 9,
                multiplicity=2,
                state_provenance="state_registry:h9_doublet_v1",
            ),
            self.config,
        )
        self.assertIn("atom_count_exceeds_gate1:9>8", violations)

    def test_pending_scientific_state_is_blocked(self) -> None:
        violations = scope_violations(
            candidate(
                atomic_numbers=(14, 1, 1, 1),
                multiplicity=2,
                state_provenance="pending_scientific_review",
            ),
            self.config,
        )
        self.assertIn("state_provenance_blocked:pending_scientific_review", violations)
        self.assertIn("non_default_state_missing_approved_registry_provenance", violations)

    def test_registered_non_default_state_is_ready(self) -> None:
        value = candidate(
            atomic_numbers=(14, 1, 1, 1),
            multiplicity=2,
            state_provenance="state_registry:sih3_doublet_v1",
        )
        self.assertEqual(scope_violations(value, self.config), ())

    def test_resource_tiers_match_the_validated_ladder(self) -> None:
        small = resource_for_candidate(candidate(atomic_numbers=(14, 1, 1, 1, 1)), self.config)
        large = resource_for_candidate(
            candidate(atomic_numbers=(14, 32, 17, 17, 17, 1, 1, 1)), self.config
        )
        self.assertEqual((small["ncpus"], small["max_memory_mb"]), (8, 24000))
        self.assertEqual((large["ncpus"], large["max_memory_mb"]), (16, 48000))


if __name__ == "__main__":
    unittest.main()
