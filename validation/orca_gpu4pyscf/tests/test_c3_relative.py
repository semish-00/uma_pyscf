from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from analyze_c3_relative import HARTREE_TO_KCAL_MOL, analyze
from generate_c3_relative_suite import SENTINELS, SUITE_ID, generate

ROOT = Path(__file__).resolve().parents[1]


def _result(energy: float, gradient: float, wall: float, density_fit: bool) -> dict:
    return {
        "schema": "crosscode-result-v1",
        "engine": "pyscf-cpu",
        "converged": True,
        "case": {
            "atoms": [{"element": "H", "xyz_angstrom": [0.0, 0.0, 0.0]}],
            "charge": 0,
            "multiplicity": 1,
            "pyscf_spin_2s": 0,
            "electron_count": 1,
            "functional": "wb97m-v",
            "basis": "def2-tzvpd",
        },
        "settings": {
            "grid_level": 5,
            "nlc_grid_level": 5,
            "grid_response": True,
            "density_fit": density_fit,
        },
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": [[gradient, 0.0, 0.0]],
        "wall_time_seconds": wall,
    }


class C3RelativeGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "structures").mkdir()
            (root / "suites").mkdir()
            shutil.copy(
                ROOT / "suites" / "si_ge_h_cl_ladder_v1.json", root / "suites"
            )
            for case_id, _ in SENTINELS:
                config = ROOT / "configs" / f"{case_id}.json"
                data = json.loads(config.read_text(encoding="utf-8"))
                shutil.copy(config, root / "configs")
                shutil.copy((config.parent / data["structure"]).resolve(), root / "structures")

            suite_path, written = generate(root)
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            self.assertEqual(suite["suite_id"], SUITE_ID)
            self.assertEqual(suite["case_count"], len(SENTINELS))
            self.assertEqual(len(written), len(SENTINELS))
            for entry in suite["cases"]:
                config = json.loads((root / entry["config"]).read_text(encoding="utf-8"))
                self.assertTrue(config["pyscf"]["density_fit"])
                self.assertEqual(config["case_id"], entry["case_id"])

            self.assertEqual(
                suite_path.read_bytes(),
                (ROOT / "suites" / f"{SUITE_ID}.json").read_bytes(),
            )
            for path in written:
                self.assertEqual(
                    path.read_bytes(),
                    (ROOT / "configs" / path.name).read_bytes(),
                )


class C3RelativeAnalysisTests(unittest.TestCase):
    def test_relative_error_cancels_common_absolute_shift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = {
                "seed": _result(-10.0, 0.0, 10.0, False),
                "distorted": _result(-9.0, 0.0, 12.0, False),
                "seed_df": _result(-10.00005, 1e-6, 5.0, True),
                "distorted_df": _result(-9.000048, 2e-6, 6.0, True),
            }
            for case_id, result in results.items():
                path = root / "runs" / case_id / "pyscf-cpu" / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(result), encoding="utf-8")
            suite = {
                "suite_id": "test",
                "cases": [
                    {
                        "case_id": "distorted_df",
                        "structure_case_id": "distorted",
                        "reference_case_id": "seed",
                        "reference_density_fit_case_id": "seed_df",
                    }
                ],
            }
            report = analyze(suite, root)
            row = report["rows"][0]
            self.assertAlmostEqual(row["relative_energy_signed_error_hartree"], 2e-6)
            self.assertAlmostEqual(
                row["relative_energy_absolute_error_kcal_mol"],
                2e-6 * HARTREE_TO_KCAL_MOL,
            )
            self.assertAlmostEqual(row["gradient_component_rmse_hartree_per_bohr"], 2e-6 / 3**0.5)
            self.assertEqual(report["aggregate_speedup_vs_direct"], 2.0)


if __name__ == "__main__":
    unittest.main()
