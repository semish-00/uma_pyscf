from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from analyze_c4_candidate import _pair_metrics, analyze
from generate_c4_density_fit_suite import SOURCE_SUITE_ID, SUITE_ID, generate

ROOT = Path(__file__).resolve().parents[1]


def _result(engine: str, energy: float, gradient: float, wall: float) -> dict:
    return {
        "schema": "crosscode-result-v1",
        "engine": engine,
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
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": [[gradient, 0.0, 0.0]],
        "s2": 0.0,
        "wall_time_seconds": wall,
        "tolerances": {
            "energy_abs_hartree": 5e-5,
            "gradient_rms_hartree_per_bohr": 2e-4,
            "gradient_max_hartree_per_bohr": 5e-4,
        },
        "settings": {
            "grid_level": 5,
            "nlc_grid_level": 5,
            "grid_response": True,
            "density_fit": engine == "gpu4pyscf",
        },
    }


class C4AnalysisTests(unittest.TestCase):
    def test_pair_metrics(self) -> None:
        left = _result("gpu4pyscf", -1.0, 3e-6, 1.0)
        right = _result("pyscf-cpu", -1.000002, 0.0, 10.0)
        metrics = _pair_metrics(left, right)
        self.assertAlmostEqual(metrics["energy_absolute_difference_hartree"], 2e-6)
        self.assertAlmostEqual(
            metrics["gradient_component_rmse_hartree_per_bohr"], 3e-6 / 3**0.5
        )
        self.assertEqual(
            metrics["gradient_max_absolute_difference_hartree_per_bohr"], 3e-6
        )

    def test_analysis_reports_both_pairs_and_speed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                ("candidate", "gpu4pyscf"): _result("gpu4pyscf", -1.0, 1e-6, 2.0),
                ("base", "pyscf-cpu"): _result("pyscf-cpu", -1.000001, 0.0, 20.0),
                ("base", "orca"): _result("orca", -1.0001, 2e-6, 30.0),
            }
            for (case_id, engine), result in values.items():
                path = root / "runs" / case_id / engine / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(result), encoding="utf-8")
            suite = {
                "suite_id": "test",
                "cases": [
                    {
                        "case_id": "candidate",
                        "base_case_id": "base",
                        "category": "test",
                    }
                ],
            }
            report = analyze(suite, root)
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(len(report["rows"]), 2)
            self.assertEqual(report["aggregate_speedup_vs_cpu_direct"], 10.0)


class C4GeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_and_changes_only_density_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "structures").mkdir()
            (root / "suites").mkdir()
            source_path = ROOT / "suites" / f"{SOURCE_SUITE_ID}.json"
            shutil.copy(source_path, root / "suites")
            source = json.loads(source_path.read_text(encoding="utf-8"))
            for entry in source["cases"]:
                config_path = ROOT / entry["config"]
                config = json.loads(config_path.read_text(encoding="utf-8"))
                shutil.copy(config_path, root / "configs")
                shutil.copy(
                    (config_path.parent / config["structure"]).resolve(),
                    root / "structures",
                )

            suite_path, written = generate(root)
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            self.assertEqual(suite["suite_id"], SUITE_ID)
            self.assertEqual(suite["case_count"], 29)
            self.assertEqual(len(written), 29)
            for entry in suite["cases"]:
                candidate = json.loads((root / entry["config"]).read_text(encoding="utf-8"))
                baseline = json.loads(
                    (root / "configs" / f"{entry['base_case_id']}.json").read_text(
                        encoding="utf-8"
                    )
                )
                baseline["case_id"] = entry["case_id"]
                baseline["pyscf"]["density_fit"] = True
                self.assertEqual(candidate, baseline)
            self.assertEqual(
                suite_path.read_bytes(),
                (ROOT / "suites" / f"{SUITE_ID}.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
