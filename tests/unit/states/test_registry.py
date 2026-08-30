"""Versioned non-default state approval registry."""

from __future__ import annotations

from pathlib import Path
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.schemas.candidate import CandidateRecord
from uma_pyscf.schemas.label_record import ElectronicState, Structure
from uma_pyscf.schemas.state_registry import StateRegistry, StateRegistryEntry
from uma_pyscf.states.registry import load_state_registry, state_registry_violations

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_REGISTRY = REPO_ROOT / "configs" / "states" / "h_si_ge_cl_states_v1.yaml"


def sih3(provenance: str | None) -> CandidateRecord:
    return CandidateRecord(
        record_id="sih3_doublet",
        structure=Structure(
            atomic_numbers=(14, 1, 1, 1),
            positions_angstrom=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        ),
        state=ElectronicState(
            charge=0,
            multiplicity=2,
            spin_2s=1,
            state_provenance=provenance,
        ),
    )


def approved_registry() -> StateRegistry:
    return StateRegistry(
        registry_id="unit_states_v1",
        created="2026-08-31",
        description="Approved unit fixture.",
        entries=(
            StateRegistryEntry(
                entry_id="sih3_neutral_doublet",
                composition="H3Si",
                charge=0,
                multiplicity=2,
                status="approved",
                evidence=("unit-reference",),
                reviewer="unit-reviewer",
                decision="unit-decision",
            ),
        ),
    )


class StateRegistryTests(unittest.TestCase):
    def test_committed_registry_lists_twelve_states_and_approves_none(self) -> None:
        registry = load_state_registry(COMMITTED_REGISTRY)
        self.assertEqual(len(registry.entries), 12)
        self.assertEqual(
            {entry.status for entry in registry.entries},
            {"pending_scientific_review"},
        )

    def test_an_exact_approved_entry_and_provenance_pass(self) -> None:
        registry = approved_registry()
        candidate = sih3("state_registry:unit_states_v1:sih3_neutral_doublet")
        self.assertEqual(state_registry_violations(candidate, registry), ())

    def test_prefix_alone_no_longer_counts_as_approval(self) -> None:
        candidate = sih3("state_registry:anything")
        self.assertEqual(
            state_registry_violations(candidate, None),
            ("non_default_state_registry_not_supplied",),
        )

    def test_pending_entry_remains_blocked_even_with_exact_provenance(self) -> None:
        registry = load_state_registry(COMMITTED_REGISTRY)
        candidate = sih3("state_registry:h_si_ge_cl_states_v1:sih3_neutral_doublet")
        self.assertIn(
            "non_default_state_registry_status:pending_scientific_review",
            state_registry_violations(candidate, registry),
        )

    def test_wrong_provenance_is_named(self) -> None:
        violations = state_registry_violations(sih3("state_registry:wrong"), approved_registry())
        self.assertTrue(
            any(
                value.startswith("state_provenance_registry_mismatch")
                for value in violations
            )
        )

    def test_approved_entry_without_accountable_review_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            StateRegistryEntry(
                entry_id="unreviewed",
                composition="H3Si",
                charge=0,
                multiplicity=2,
                status="approved",
            )

    def test_duplicate_scientific_state_is_refused(self) -> None:
        first = approved_registry().entries[0]
        with self.assertRaises(ValidationError):
            StateRegistry(
                registry_id="duplicate_v1",
                created="2026-08-31",
                description="Broken duplicate.",
                entries=(
                    first,
                    StateRegistryEntry(
                        entry_id="same_state_again",
                        composition=first.composition,
                        charge=first.charge,
                        multiplicity=first.multiplicity,
                        status="rejected",
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
