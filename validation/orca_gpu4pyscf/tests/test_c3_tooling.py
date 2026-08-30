from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from analyze_c3_matrix import compare_variant
from generate_c3_matrix import (
    BASE_CASE_IDS,
    BASELINE_SETTINGS,
    DENSITY_FIT_SUITE_ID,
    SUITE_ID,
    VARIANTS,
    generate,
)

ROOT = Path(__file__).resolve().parents[1]


def _copy_inputs(destination: Path) -> None:
    (destination / "configs").mkdir(parents=True)
    (destination / "structures").mkdir()
    (destination / "suites").mkdir()
    shutil.copy(ROOT / "suites" / "gpu_smoke_v1.json", destination / "suites")
    for case_id in BASE_CASE_IDS:
        config = ROOT / "configs" / f"{case_id}.json"
        data = json.loads(config.read_text(encoding="utf-8"))
        shutil.copy(config, destination / "configs")
        structure = (config.parent / data["structure"]).resolve()
        shutil.copy(structure, destination / "structures")


def _result(case_id: str, *, energy: float = -10.0) -> dict:
    return {
        "case": {
            "case_id": case_id,
            "atoms": [{"element": "H", "xyz_angstrom": [0.0, 0.0, 0.0]}],
            "charge": 0,
            "multiplicity": 1,
            "pyscf_spin_2s": 0,
            "electron_count": 1,
            "functional": "wb97m-v",
            "basis": "def2-tzvpd",
        },
        "engine": "gpu4pyscf",
        "converged": True,
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": [[0.0, 0.0, 0.0]],
        "s2": 0.0,
        "wall_time_seconds": 10.0,
        "settings": {
            "scf": {"conv_tol": 1e-10, "max_cycle": 200},
            "grid_level": 5,
            "nlc_grid_level": 5,
            "grid_response": True,
            "density_fit": False,
        },
    }


class C3GeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_and_changes_one_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _copy_inputs(root)
            suite_path, written = generate(root)
            self.assertEqual(len(written), len(BASE_CASE_IDS) * len(VARIANTS))
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            self.assertEqual(suite["suite_id"], SUITE_ID)
            self.assertEqual(suite["case_count"], len(written))
            self.assertEqual(suite["baseline_settings"], BASELINE_SETTINGS)

            for entry in suite["cases"]:
                baseline = json.loads(
                    (root / "configs" / f"{entry['base_case_id']}.json").read_text(
                        encoding="utf-8"
                    )
                )
                candidate = json.loads(
                    (root / entry["config"]).read_text(encoding="utf-8")
                )
                expected = deepcopy(baseline)
                expected["case_id"] = entry["case_id"]
                expected["pyscf"][entry["setting_key"]] = entry["candidate_value"]
                self.assertEqual(candidate, expected)

            committed_suite = ROOT / "suites" / f"{SUITE_ID}.json"
            self.assertEqual(suite_path.read_bytes(), committed_suite.read_bytes())
            density_fit_suite_path = (
                root / "suites" / f"{DENSITY_FIT_SUITE_ID}.json"
            )
            density_fit_suite = json.loads(
                density_fit_suite_path.read_text(encoding="utf-8")
            )
            self.assertEqual(density_fit_suite["case_count"], len(BASE_CASE_IDS))
            self.assertEqual(
                density_fit_suite["engine_jobs_per_case"],
                ["pyscf-cpu", "gpu4pyscf"],
            )
            self.assertTrue(
                all(
                    entry["setting_key"] == "density_fit"
                    and entry["candidate_value"] is True
                    for entry in density_fit_suite["cases"]
                )
            )
            committed_density_fit_suite = (
                ROOT / "suites" / f"{DENSITY_FIT_SUITE_ID}.json"
            )
            self.assertEqual(
                density_fit_suite_path.read_bytes(),
                committed_density_fit_suite.read_bytes(),
            )
            for generated in written:
                self.assertEqual(
                    generated.read_bytes(),
                    (ROOT / "configs" / generated.name).read_bytes(),
                )


class C3AnalysisTests(unittest.TestCase):
    def test_compare_variant_reports_error_and_speed(self) -> None:
        baseline = _result("base")
        candidate = _result("candidate", energy=-9.999999)
        candidate["gradient_hartree_per_bohr"] = [[1e-6, -2e-6, 3e-6]]
        candidate["settings"]["grid_level"] = 4
        candidate["wall_time_seconds"] = 5.0
        row = {
            "base_case_id": "base",
            "case_id": "candidate",
            "axis": "ordinary_grid",
            "setting_key": "grid_level",
            "baseline_value": 5,
            "candidate_value": 4,
        }
        metrics = compare_variant(baseline, candidate, row)
        self.assertAlmostEqual(metrics["energy_absolute_difference_hartree"], 1e-6)
        self.assertEqual(metrics["gradient_max_absolute_difference_hartree_per_bohr"], 3e-6)
        self.assertEqual(metrics["speedup_vs_baseline"], 2.0)
        self.assertTrue(metrics["within_provisional_cpu_gpu_thresholds"])

    def test_compare_variant_rejects_multiple_setting_changes(self) -> None:
        baseline = _result("base")
        candidate = _result("candidate")
        candidate["settings"]["grid_level"] = 4
        candidate["settings"]["nlc_grid_level"] = 4
        row = {
            "base_case_id": "base",
            "case_id": "candidate",
            "axis": "ordinary_grid",
            "setting_key": "grid_level",
            "baseline_value": 5,
            "candidate_value": 4,
        }
        with self.assertRaisesRegex(ValueError, "Expected only settings.grid_level"):
            compare_variant(baseline, candidate, row)


if __name__ == "__main__":
    unittest.main()
