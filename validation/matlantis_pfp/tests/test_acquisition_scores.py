from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "build_acquisition_scores.py"
SPEC = importlib.util.spec_from_file_location("matlantis_pfp_acquisition", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
acquisition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquisition)


class AcquisitionScoreTests(unittest.TestCase):
    def test_scores_use_predictions_but_not_reference_results(self) -> None:
        candidates = {
            "schema": "uma-pyscf-candidate-manifest-v1",
            "sampling_id": "pool_v1",
            "records": [self._candidate("r1", 0.0), self._candidate("r2", 0.1)],
        }
        evaluation = {
            "schema": "uma-pyscf-uma-evaluation-v1",
            "evaluation_id": "base_v1",
            "model": {"name": "uma-s-1p2"},
            "predictions": [self._uma("r1", 1.0), self._uma("r2", 2.0)],
        }
        pfp = {"r1": self._pfp("r1", 0.0, 3.0), "r2": self._pfp("r2", 0.1, 4.0)}

        manifest = acquisition.build_score_manifest(candidates, evaluation, pfp)

        self.assertEqual(manifest["schema"], "uma-pyscf-acquisition-scores-v1")
        self.assertEqual(manifest["score_id"], "pool_v1_pfp_uma_scores_v1")
        self.assertEqual(len(manifest["records"]), 2)
        first = manifest["records"][0]
        self.assertFalse(first["provenance"]["reference_fields_used"])
        self.assertAlmostEqual(first["scores"]["pfp_uma_force_rms"], 2.0)
        self.assertIn("pfp_uma_combined_rank", first["scores"])

    def test_trajectory_provenance_is_forwarded_to_selection_scores(self) -> None:
        candidate = self._candidate("r1", 0.0)
        candidate["generation_parameters"] = {
            "trajectory_id": "reaction_forward",
            "frame_index": 17,
        }
        candidates = {
            "schema": "uma-pyscf-candidate-manifest-v1",
            "sampling_id": "pool_v1",
            "records": [candidate],
        }
        evaluation = {
            "schema": "uma-pyscf-uma-evaluation-v1",
            "evaluation_id": "base_v1",
            "model": {"name": "uma-s-1p2"},
            "predictions": [self._uma("r1", 1.0)],
        }

        manifest = acquisition.build_score_manifest(
            candidates, evaluation, {"r1": self._pfp("r1", 0.0, 3.0)}
        )

        self.assertEqual(manifest["records"][0]["trajectory_id"], "reaction_forward")
        self.assertEqual(manifest["records"][0]["frame_index"], 17)

    def test_unlabeled_model_prediction_manifest_is_supported(self) -> None:
        candidates = {
            "schema": "uma-pyscf-candidate-manifest-v1",
            "sampling_id": "pool_v1",
            "records": [self._candidate("r1", 0.0)],
        }
        predictions = {
            "schema": "uma-pyscf-model-predictions-v1",
            "prediction_id": "pool_base_v1",
            "model": {"name": "uma-s-1p2"},
            "records": [
                {
                    "record_id": "r1",
                    "results": {
                        "energy_ev": 1.0,
                        "forces_ev_per_angstrom": [[1.0, 1.0, 1.0]],
                    },
                }
            ],
        }
        pfp = {"r1": self._pfp("r1", 0.0, 3.0)}

        manifest = acquisition.build_score_manifest(candidates, predictions, pfp)

        self.assertEqual(
            manifest["records"][0]["provenance"]["uma_prediction_id"],
            "pool_base_v1",
        )

    @staticmethod
    def _candidate(record_id: str, x: float) -> dict:
        return {
            "record_id": record_id,
            "structure": {
                "atomic_numbers": [1],
                "positions_angstrom": [[x, 0.0, 0.0]],
                "parent_structure_id": f"p{record_id[-1]}",
            },
            "state": {"charge": 0, "multiplicity": 1},
        }

    @staticmethod
    def _uma(record_id: str, force: float) -> dict:
        return {
            "record_id": record_id,
            "predicted_energy_ev": force,
            "predicted_forces_ev_per_angstrom": [[force, force, force]],
            "reference_energy_ev": 999999.0,
            "reference_forces_ev_per_angstrom": [[999999.0, 999999.0, 999999.0]],
        }

    @staticmethod
    def _pfp(record_id: str, x: float, force: float) -> dict:
        return {
            "schema": "uma-pyscf-pfp-single-point-v1",
            "record_id": record_id,
            "input": {
                "atomic_numbers": [1],
                "positions_angstrom": [[x, 0.0, 0.0]],
            },
            "model": {
                "name": "PFP",
                "model_version": "v9.0.0",
                "calc_mode": "R2SCAN_PLUS_D3",
            },
            "provenance": {"runtime_versions": {"pfp_api_client": "1.21.3"}},
            "results": {
                "energy_ev": force + 10.0,
                "forces_ev_per_angstrom": [[force, force, force]],
            },
        }


if __name__ == "__main__":
    unittest.main()
