"""Unlabeled UMA prediction schema tests."""

from __future__ import annotations

from copy import deepcopy
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.schemas.label_record import ElectronicState, Structure
from uma_pyscf.schemas.model_prediction import (
    ModelPredictionManifest,
    ModelPredictionRecord,
)


def record(record_id: str = "h2_prediction") -> ModelPredictionRecord:
    return ModelPredictionRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)),
            parent_structure_id="h2_seed",
            sampling_method="bond_scan",
        ),
        state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
        energy_ev=-31.0,
        forces_ev_per_angstrom=((0.0, 0.0, 0.2), (0.0, 0.0, -0.2)),
    )


class ModelPredictionSchemaTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        original = ModelPredictionManifest(
            prediction_id="h2_base_uma_v1",
            source={"id": "h2_pool_v1", "sha256": "a" * 64},
            model={"name": "uma-s-1p2", "task": "omol"},
            runtime={"fairchem_core": "2.22.0"},
            records=(record(),),
        )
        self.assertEqual(
            ModelPredictionManifest.from_dict(original.to_dict()).to_dict(),
            original.to_dict(),
        )

    def test_force_count_units_and_unknown_results_fail(self) -> None:
        with self.assertRaises(ValidationError):
            ModelPredictionRecord(
                record_id="bad",
                structure=record().structure,
                state=record().state,
                energy_ev=0.0,
                forces_ev_per_angstrom=((0.0, 0.0, 0.0),),
            )
        manifest = ModelPredictionManifest(
            prediction_id="valid",
            source={"id": "pool", "sha256": "b" * 64},
            model={"name": "uma"},
            runtime={"version": "1"},
            records=(record(),),
        ).to_dict()
        manifest["units"]["energy"] = "hartree"
        with self.assertRaises(ValidationError):
            ModelPredictionManifest.from_dict(manifest)
        tampered = deepcopy(record().to_dict())
        tampered["results"]["reference_energy_ev"] = -1.0
        with self.assertRaises(ValidationError):
            ModelPredictionRecord.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
