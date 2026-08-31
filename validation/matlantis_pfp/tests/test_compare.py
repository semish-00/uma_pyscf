from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "compare.py"
SPEC = importlib.util.spec_from_file_location("matlantis_pfp_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class CompareRecordsTest(unittest.TestCase):
    def test_same_composition_offset_is_removed_and_gradient_sign_is_inverted(self) -> None:
        pfp = {
            "a": self._pfp("a", 10.0, 1.0),
            "b": self._pfp("b", 12.0, 1.0),
        }
        reference = {
            "a": self._reference("a", 1.0 / compare.HARTREE_TO_EV, -0.5),
            "b": self._reference("b", 3.0 / compare.HARTREE_TO_EV, -0.5),
        }

        rows, summary = compare.compare_records(pfp, reference)

        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["geometry_max_abs_delta_angstrom"], 0.0)
        self.assertAlmostEqual(summary["energy_same_composition_centered_mae_ev"], 0.0)
        expected_force_error = 1.0 - 0.5 * compare.HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        self.assertAlmostEqual(
            summary["force_component_mae_ev_per_angstrom"], abs(expected_force_error)
        )
        self.assertEqual([row["record_id"] for row in rows], ["a", "b"])

    def test_partition_metrics_require_exact_non_overlapping_coverage(self) -> None:
        pfp = {
            "a": self._pfp("a", 10.0, 1.0),
            "b": self._pfp("b", 12.0, 1.0),
        }
        reference = {
            "a": self._reference("a", 1.0, -0.5),
            "b": self._reference("b", 2.0, -0.5),
        }
        split = {
            "schema": "uma-pyscf-split-manifest-v1",
            "record_assignments": {"train": ["a"], "holdout": ["b"]},
        }

        metrics = compare._partition_metrics(pfp, reference, split)

        self.assertEqual(metrics["train"]["records"], 1)
        self.assertEqual(metrics["holdout"]["records"], 1)

    @staticmethod
    def _pfp(record_id: str, energy_ev: float, force: float) -> dict:
        return {
            "schema": "uma-pyscf-pfp-single-point-v1",
            "record_id": record_id,
            "input": {
                "atomic_numbers": [1],
                "positions_angstrom": [[0.0, 0.0, 0.0]],
                "charge": 0,
                "multiplicity": 1,
            },
            "results": {
                "energy_ev": energy_ev,
                "forces_ev_per_angstrom": [[force, force, force]],
            },
        }

    @staticmethod
    def _reference(record_id: str, energy_hartree: float, gradient: float) -> dict:
        return {
            "schema": "uma-pyscf-label-record-v1",
            "record_id": record_id,
            "structure": {
                "atomic_numbers": [1],
                "positions_angstrom": [[0.0, 0.0, 0.0]],
            },
            "state": {"charge": 0, "multiplicity": 1},
            "results": {
                "energy_hartree": energy_hartree,
                "gradient_hartree_per_bohr": [[gradient, gradient, gradient]],
            },
        }


if __name__ == "__main__":
    unittest.main()
