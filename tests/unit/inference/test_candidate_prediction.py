"""Unlabeled candidate-manifest UMA prediction without fairchem imports."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.inference.uma import predict_candidate_manifest
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord
from uma_pyscf.schemas.label_record import ElectronicState, Structure
from uma_pyscf.schemas.model_prediction import ModelPredictionManifest


class FakeAtoms:
    def __init__(self, *, numbers, positions, pbc) -> None:
        self.numbers = tuple(numbers)
        self.positions = tuple(tuple(row) for row in positions)
        self.pbc = pbc
        self.info: dict[str, int] = {}
        self.calc = None

    def get_potential_energy(self) -> float:
        return -float(sum(self.numbers))

    def get_forces(self):
        return [[0.1, 0.2, 0.3] for _ in self.numbers]


class FakeAse:
    __version__ = "test"
    Atoms = FakeAtoms


def candidates() -> CandidateManifest:
    config = {"schema_version": 1, "sampling_id": "pool_v1"}
    return CandidateManifest(
        sampling_id="pool_v1",
        config_sha256=canonical_json_fingerprint(config),
        config=config,
        records=(
            CandidateRecord(
                record_id="h2_candidate",
                structure=Structure(
                    atomic_numbers=(1, 1),
                    positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)),
                    parent_structure_id="h2_seed",
                    sampling_method="bond_scan",
                ),
                state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
            ),
        ),
    )


class CandidatePredictionTests(unittest.TestCase):
    def test_prediction_contains_no_reference_fields_and_round_trips(self) -> None:
        context = {
            "ase": FakeAse,
            "calculator": object(),
            "local_checkpoint": None,
            "cache_files": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.json"
            with (
                patch("uma_pyscf.inference.uma._initialize_uma", return_value=context),
                patch(
                    "uma_pyscf.inference.uma._model_metadata",
                    return_value={"name": "uma-s-1p2", "task": "omol"},
                ),
                patch(
                    "uma_pyscf.inference.uma._runtime_metadata",
                    return_value={"fairchem_core": "2.22.0"},
                ),
            ):
                artifact = predict_candidate_manifest(
                    candidates(),
                    manifest_sha256="a" * 64,
                    prediction_id="pool_base_uma_v1",
                    model_name="uma-s-1p2",
                    model_source="test",
                    model_license="test",
                    checkpoint_path=None,
                    model_cache_dir=directory,
                    task="omol",
                    device="cuda",
                    inference_settings="default",
                    seed=41,
                    fairchem_core_version="2.22.0",
                    output_path=output,
                    repository=directory,
                    container_sha256="b" * 64,
                )

            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                ModelPredictionManifest.from_dict(loaded).to_dict(), artifact.to_dict()
            )
            self.assertEqual(loaded["records"][0]["results"]["energy_ev"], -2.0)
            self.assertNotIn("reference", json.dumps(loaded))


if __name__ == "__main__":
    unittest.main()
