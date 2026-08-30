"""Train-only atomic composition baseline fitting and leakage refusals."""

from __future__ import annotations

from dataclasses import replace
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.datasets.baseline import (
    fit_atomic_composition_baseline,
    predict_baseline_energy,
)
from uma_pyscf.schemas.composition_baseline import CompositionBaseline
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
from uma_pyscf.schemas.split_manifest import SplitManifest

REFERENCES = {"Cl": -1.0, "Ge": -3.0, "H": -0.5, "Si": -2.0}
COMPOSITIONS = {
    "sih4": (14, 1, 1, 1, 1),
    "geh4": (32, 1, 1, 1, 1),
    "gecl4": (32, 17, 17, 17, 17),
    "mixed": (14, 32, 17, 17, 17, 1, 1, 1),
    "sicl4": (14, 17, 17, 17, 17),
}


def record(record_id: str, atomic_numbers: tuple[int, ...]) -> LabelRecord:
    counts = {symbol: 0 for symbol in REFERENCES}
    symbols = {1: "H", 14: "Si", 17: "Cl", 32: "Ge"}
    for number in atomic_numbers:
        counts[symbols[number]] += 1
    energy = sum(counts[symbol] * value for symbol, value in REFERENCES.items())
    return LabelRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=atomic_numbers,
            positions_angstrom=tuple(
                (float(index), 0.0, 0.0) for index in range(len(atomic_numbers))
            ),
            parent_structure_id=f"{record_id}_parent",
        ),
        state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
        method=Method(
            functional="wb97m-v",
            basis="def2-tzvpd",
            ecp=None,
            aux_basis=None,
            grid_level=5,
            nlc_grid_level=5,
            grid_response=True,
            density_fit=True,
            scf_conv_tol=1e-10,
            scf_max_cycle=250,
        ),
        engine=Engine(name="gpu4pyscf", versions={"gpu4pyscf": "1.8.1"}),
        results=Results(
            energy_hartree=energy,
            gradient_hartree_per_bohr=tuple((0.0, 0.0, 0.0) for _ in atomic_numbers),
            converged=True,
        ),
        raw=RawArtifact(),
        qc=QcState(status="accepted"),
    )


def full_rank_fixture() -> tuple[tuple[LabelRecord, ...], SplitManifest]:
    records = tuple(record(name, COMPOSITIONS[name]) for name in COMPOSITIONS)
    split = SplitManifest(
        split_id="baseline_split_v1",
        axis="parent",
        seed=1,
        partitions={"train": 0.8, "holdout": 0.2},
        source={"id": "source_v1", "sha256": "a" * 64},
        group_assignments={
            "sih4_parent": "train",
            "geh4_parent": "train",
            "gecl4_parent": "train",
            "mixed_parent": "train",
            "sicl4_parent": "holdout",
        },
        record_assignments={
            "train": ("sih4", "geh4", "gecl4", "mixed"),
            "holdout": ("sicl4",),
        },
    )
    return records, split


class AtomicBaselineTests(unittest.TestCase):
    def test_exact_full_rank_fit_recovers_atomic_references(self) -> None:
        records, split = full_rank_fixture()
        baseline = fit_atomic_composition_baseline(
            records,
            split,
            baseline_id="unit_atomic_baseline_v1",
            fit_partition="train",
            record_checksums_sha256={entry.record_id: "b" * 64 for entry in records},
        )
        for symbol, expected in REFERENCES.items():
            self.assertAlmostEqual(
                baseline.atomic_reference_energy_hartree[symbol], expected, places=12
            )
        self.assertEqual(baseline.fit_record_ids, ("gecl4", "geh4", "mixed", "sih4"))
        self.assertAlmostEqual(
            baseline.metrics_by_partition["holdout"]["max_abs_error_hartree"], 0.0
        )

    def test_prediction_refuses_an_element_absent_from_the_baseline(self) -> None:
        value = record("sih4", COMPOSITIONS["sih4"])
        with self.assertRaises(ValidationError) as caught:
            predict_baseline_energy(value, {"H": -0.5})
        self.assertIn("absent", str(caught.exception))

    def test_rank_deficient_training_compositions_fail_closed(self) -> None:
        records, split = full_rank_fixture()
        broken = SplitManifest(
            split_id="rank_deficient_v1",
            axis="parent",
            seed=1,
            partitions={"train": 0.6, "holdout": 0.4},
            source=split.source,
            group_assignments={
                "sih4_parent": "train",
                "geh4_parent": "train",
                "gecl4_parent": "train",
                "mixed_parent": "holdout",
                "sicl4_parent": "holdout",
            },
            record_assignments={
                "train": ("sih4", "geh4", "gecl4"),
                "holdout": ("mixed", "sicl4"),
            },
        )
        with self.assertRaises(ValidationError) as caught:
            fit_atomic_composition_baseline(
                records,
                broken,
                baseline_id="broken_v1",
                fit_partition="train",
                record_checksums_sha256={entry.record_id: "b" * 64 for entry in records},
            )
        self.assertIn("rank", str(caught.exception))

    def test_a_nonaccepted_record_is_never_used(self) -> None:
        records, split = full_rank_fixture()
        records = tuple(
            replace(entry, qc=QcState(status="rejected"))
            if entry.record_id == "sih4"
            else entry
            for entry in records
        )
        with self.assertRaises(ValidationError) as caught:
            fit_atomic_composition_baseline(
                records,
                split,
                baseline_id="broken_v1",
                fit_partition="train",
                record_checksums_sha256={entry.record_id: "b" * 64 for entry in records},
            )
        self.assertIn("accepted", str(caught.exception))

    def test_baseline_schema_round_trips_and_rejects_a_changed_rank(self) -> None:
        records, split = full_rank_fixture()
        baseline = fit_atomic_composition_baseline(
            records,
            split,
            baseline_id="roundtrip_v1",
            fit_partition="train",
            record_checksums_sha256={entry.record_id: "b" * 64 for entry in records},
        )
        self.assertEqual(CompositionBaseline.from_dict(baseline.to_dict()), baseline)
        changed = baseline.to_dict()
        changed["design_rank"] = 3
        with self.assertRaises(ValidationError):
            CompositionBaseline.from_dict(changed)


if __name__ == "__main__":
    unittest.main()
