from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from common import case_record, load_case, write_json
import gate1_metrics
import generate_charge_spin_mini_suite
from generate_charge_spin_mini_suite import generate

ROOT = Path(__file__).resolve().parents[1]
MINI_SUITE = ROOT / "suites" / "charge_spin_mini_v1.json"

# The charge/spin matrix the plan asks for, stated independently of the
# generator tables so a change in either side is visible here.
EXPECTED_MATRIX = {
    ("sih3", 0, 2),
    ("sih3", 0, 4),
    ("sih3", 1, 1),
    ("sih3", 1, 3),
    ("sih3", -1, 1),
    ("sih3", -1, 3),
    ("geh3", 0, 2),
    ("geh3", 0, 4),
    ("geh3", 1, 1),
    ("geh3", 1, 3),
    ("geh3", -1, 1),
    ("geh3", -1, 3),
}
SEED_CASE_IDS = ("sih3_doublet_planar_seed", "geh3_doublet_planar_seed")

CASE_METRIC_HEADERS = [
    "case_id",
    "category",
    "left_engine",
    "right_engine",
    "energy_signed_difference_hartree",
    "energy_absolute_difference_hartree",
    "gradient_component_rmse_hartree_per_bohr",
    "gradient_component_mae_hartree_per_bohr",
    "gradient_max_absolute_difference_hartree_per_bohr",
    "gradient_max_atom_index_zero_based",
    "gradient_max_axis",
]
PERFORMANCE_HEADERS = [
    "case_id",
    "category",
    "engine",
    "converged",
    "energy_hartree",
    "s2",
    "s2_target",
    "s2_deviation",
    "wall_time_seconds",
    "scf_wall_time_seconds",
    "gradient_wall_time_seconds",
]


def _copy_seed_tree(destination: Path) -> None:
    """Copy only what the generator is allowed to read: seed configs/structures."""
    (destination / "configs").mkdir(parents=True)
    (destination / "structures").mkdir()
    for case_id in SEED_CASE_IDS:
        shutil.copy(ROOT / "configs" / f"{case_id}.json", destination / "configs")
        shutil.copy(ROOT / "structures" / f"{case_id}.xyz", destination / "structures")


class ChargeSpinMiniSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = json.loads(MINI_SUITE.read_text(encoding="utf-8"))

    def test_suite_shape_and_pending_review_status(self) -> None:
        self.assertEqual(self.suite["schema"], "crosscode-suite-v1")
        self.assertEqual(self.suite["suite_id"], "charge_spin_mini_v1")
        self.assertEqual(self.suite["engine_jobs_per_case"], ["pyscf-cpu", "gpu4pyscf"])
        self.assertEqual(
            self.suite["state_selection_status"], "pending_scientific_review"
        )
        self.assertIn("NOT yet approved as training-label states", self.suite["description"])
        self.assertEqual(len(self.suite["cases"]), 12)
        self.assertEqual(self.suite["case_count"], 12)

    def test_every_case_manifest_loads_and_matches_the_suite_entry(self) -> None:
        for entry in self.suite["cases"]:
            with self.subTest(case_id=entry["case_id"]):
                case = load_case(ROOT / entry["config"])
                self.assertEqual(case.case_id, entry["case_id"])
                self.assertEqual(case.charge, entry["charge"])
                self.assertEqual(case.multiplicity, entry["multiplicity"])
                self.assertEqual(entry["category"], "charge_spin_matrix")
                self.assertEqual(entry["state"]["base"], entry["case_id"].split("_")[0])

    def test_expected_charge_spin_matrix_is_exactly_present(self) -> None:
        observed = {
            (entry["state"]["base"], entry["charge"], entry["multiplicity"])
            for entry in self.suite["cases"]
        }
        self.assertEqual(observed, EXPECTED_MATRIX)
        # The neutral doublet seeds enter as the reference state, not as new configs.
        references = [
            entry["case_id"]
            for entry in self.suite["cases"]
            if entry["state"]["label"] == "neutral_doublet_reference"
        ]
        self.assertEqual(sorted(references), sorted(SEED_CASE_IDS))

    def test_generated_manifests_only_change_the_electronic_state(self) -> None:
        for entry in self.suite["cases"]:
            if entry["case_id"] in SEED_CASE_IDS:
                continue
            base = entry["state"]["base"]
            seed = json.loads(
                (ROOT / "configs" / f"{base}_doublet_planar_seed.json").read_text(
                    encoding="utf-8"
                )
            )
            generated = json.loads(
                (ROOT / entry["config"]).read_text(encoding="utf-8")
            )
            with self.subTest(case_id=entry["case_id"]):
                self.assertEqual(
                    generated["structure"], f"../structures/{base}_doublet_planar_seed.xyz"
                )
                changed = {
                    key
                    for key in set(seed) | set(generated)
                    if seed.get(key) != generated.get(key)
                }
                expected = {"case_id"} | {
                    key
                    for key in ("charge", "multiplicity")
                    if entry[key] != seed[key]
                }
                self.assertEqual(changed, expected)
                self.assertEqual(generated["charge"], entry["charge"])
                self.assertEqual(generated["multiplicity"], entry["multiplicity"])

    def test_generation_is_deterministic_and_matches_the_committed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _copy_seed_tree(root)
            structures_before = {
                path.name: path.read_bytes() for path in (root / "structures").iterdir()
            }

            suite_path, written = generate(root)
            first_suite = suite_path.read_bytes()
            first_configs = {path.name: path.read_bytes() for path in written}

            second_suite_path, second_written = generate(root)
            self.assertEqual(second_suite_path, suite_path)
            self.assertEqual(second_suite_path.read_bytes(), first_suite)
            self.assertEqual(
                {path.name: path.read_bytes() for path in second_written}, first_configs
            )

            self.assertEqual(len(first_configs), 10)
            self.assertEqual(first_suite, MINI_SUITE.read_bytes())
            for name, payload in first_configs.items():
                self.assertEqual(payload, (ROOT / "configs" / name).read_bytes())

            # The generator must never create or rewrite a structure file.
            self.assertEqual(
                {path.name: path.read_bytes() for path in (root / "structures").iterdir()},
                structures_before,
            )

    def test_generator_refuses_to_write_without_the_seed_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "structures").mkdir()
            with self.assertRaises(FileNotFoundError):
                generate(root)
            self.assertFalse((root / "suites" / "charge_spin_mini_v1.json").is_file())

    def test_module_tables_cover_the_documented_bases(self) -> None:
        self.assertEqual(generate_charge_spin_mini_suite.BASES, ("sih3", "geh3"))
        self.assertEqual(len(generate_charge_spin_mini_suite.STATES), 5)


CASES = {
    "h2_wb97mv_def2tzvpd": [[0.0, 0.0, 0.001], [0.0, 0.0, -0.001]],
    "sih4_td_seed": [
        [0.0, 0.0, 0.002],
        [0.001, 0.001, 0.001],
        [-0.001, -0.001, 0.001],
        [-0.001, 0.001, -0.001],
        [0.001, -0.001, -0.001],
    ],
}
BASE_ENERGY = {"h2_wb97mv_def2tzvpd": -1.16, "sih4_td_seed": -291.5}


def _shifted(gradient: list[list[float]], delta: float) -> list[list[float]]:
    """Shift every component so RMSE, MAE, and max all equal ``delta``."""
    return [[value + delta for value in row] for row in gradient]


def _result(case_id: str, engine: str, energy_delta: float, gradient_delta: float) -> dict:
    case = load_case(ROOT / "configs" / f"{case_id}.json")
    return {
        "schema": "crosscode-result-v1",
        "engine": engine,
        "case": case_record(case),
        "converged": True,
        "energy_hartree": BASE_ENERGY[case_id] + energy_delta,
        "gradient_hartree_per_bohr": _shifted(CASES[case_id], gradient_delta),
        "s2": None,
        "s2_target": 0.0,
        "s2_deviation": None,
        "wall_time_seconds": 12.5,
        "tolerances": case.tolerances,
        "tolerance_status": case.raw["tolerance_status"],
    }


def _write_tree(root: Path, plan: dict[str, dict[str, dict]]) -> Path:
    """Write fabricated results plus a suite manifest; return the suite path."""
    for case_id, engines in plan.items():
        for engine, result in engines.items():
            write_json(root / "runs" / case_id / engine / "result.json", result)
    suite_path = root / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema": "crosscode-suite-v1",
                "suite_id": "gate1_test",
                "case_count": len(plan),
                "cases": [
                    {
                        "case_id": case_id,
                        "category": "gate1_fixture",
                        "config": f"configs/{case_id}.json",
                    }
                    for case_id in plan
                ],
            }
        ),
        encoding="utf-8",
    )
    return suite_path


def _default_plan(
    gpu_energy_delta: float = 1e-9, gpu_gradient_delta: float = 1e-9
) -> dict[str, dict[str, dict]]:
    """Both cases with all three engines; ORCA sits far outside the CPU-GPU noise."""
    return {
        case_id: {
            "pyscf-cpu": _result(case_id, "pyscf-cpu", 0.0, 0.0),
            "gpu4pyscf": _result(
                case_id, "gpu4pyscf", gpu_energy_delta, gpu_gradient_delta
            ),
            "orca": _result(case_id, "orca", 1e-4, 1e-4),
        }
        for case_id in CASES
    }


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


class Gate1MetricsTests(unittest.TestCase):
    def _run(self, plan: dict[str, dict[str, dict]]) -> tuple[Path, dict]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        suite_path = _write_tree(root, plan)
        output_dir = root / "analysis"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "gate1_metrics.py"),
                str(suite_path),
                "--root",
                str(root),
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(
            (output_dir / "gate1_summary_gate1_test.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["schema"], "gate1-metrics-summary-v1")
        return output_dir, summary

    def test_all_engines_present_and_cpu_gpu_within_thresholds(self) -> None:
        output_dir, summary = self._run(_default_plan())
        self.assertEqual(summary["suite_id"], "gate1_test")
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(
            summary["engine_result_counts"],
            {"orca": 2, "pyscf-cpu": 2, "gpu4pyscf": 2},
        )
        self.assertEqual(summary["missing"], {"orca": [], "pyscf-cpu": [], "gpu4pyscf": []})

        self.assertEqual(
            [(pair["left_engine"], pair["right_engine"]) for pair in summary["pairs"]],
            [
                ("gpu4pyscf", "pyscf-cpu"),
                ("pyscf-cpu", "orca"),
                ("gpu4pyscf", "orca"),
            ],
        )
        for pair in summary["pairs"]:
            self.assertEqual(pair["paired_case_count"], 2)
            self.assertIn(
                pair["worst_energy_absolute_difference_hartree"]["case_id"], CASES
            )
        cpu_gpu = summary["pairs"][0]
        self.assertLess(
            cpu_gpu["worst_gradient_component_rmse_hartree_per_bohr"]["value"], 1e-8
        )

        gate = summary["provisional_cpu_gpu_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["evaluated_case_count"], 2)
        self.assertEqual(gate["passed_case_count"], 2)
        self.assertEqual(gate["failed_case_count"], 0)
        self.assertTrue(gate["final_calculation_success"]["passed"])
        self.assertEqual(gate["final_calculation_success"]["gpu4pyscf_result_count"], 2)
        self.assertEqual(
            gate["thresholds"],
            {
                "energy_abs_hartree": 5e-6,
                "gradient_rms_hartree_per_bohr": 2e-5,
                "gradient_max_hartree_per_bohr": 1e-4,
            },
        )
        self.assertEqual(
            gate["tolerance_status"], "provisional_not_scientifically_frozen"
        )

        relative = summary["relative_condition"]
        self.assertTrue(relative["passed"])
        self.assertEqual(relative["compared_case_count"], 2)
        self.assertEqual(relative["energy_absolute_difference_violations"], [])
        self.assertEqual(relative["gradient_component_rmse_violations"], [])

        headers, rows = _read_csv(output_dir / "gate1_case_metrics_gate1_test.csv")
        self.assertEqual(headers, CASE_METRIC_HEADERS)
        self.assertEqual(len(rows), 6)
        signed = {
            (row[0], row[2], row[3]): float(row[4])
            for row in rows
        }
        self.assertAlmostEqual(
            signed[("h2_wb97mv_def2tzvpd", "gpu4pyscf", "pyscf-cpu")], 1e-9, places=12
        )
        self.assertAlmostEqual(
            signed[("h2_wb97mv_def2tzvpd", "pyscf-cpu", "orca")], -1e-4, places=8
        )

        headers, rows = _read_csv(output_dir / "gate1_performance_gate1_test.csv")
        self.assertEqual(headers, PERFORMANCE_HEADERS)
        self.assertEqual(len(rows), 6)
        # The split timers are absent from these fabricated results.
        for row in rows:
            self.assertEqual(row[PERFORMANCE_HEADERS.index("scf_wall_time_seconds")], "")
            self.assertEqual(
                row[PERFORMANCE_HEADERS.index("gradient_wall_time_seconds")], ""
            )
            self.assertEqual(row[PERFORMANCE_HEADERS.index("converged")], "True")
            self.assertEqual(
                row[PERFORMANCE_HEADERS.index("wall_time_seconds")], "12.5"
            )

    def test_energy_difference_above_threshold_fails_that_case(self) -> None:
        plan = _default_plan()
        plan["sih4_td_seed"]["gpu4pyscf"] = _result("sih4_td_seed", "gpu4pyscf", 1e-5, 1e-9)
        _, summary = self._run(plan)
        gate = summary["provisional_cpu_gpu_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["passed_case_count"], 1)
        self.assertEqual(gate["failed_case_count"], 1)
        by_case = {case["case_id"]: case for case in gate["cases"]}
        self.assertEqual(by_case["sih4_td_seed"]["failed_metrics"], ["energy_abs_hartree"])
        self.assertFalse(by_case["sih4_td_seed"]["passed"])
        self.assertTrue(by_case["h2_wb97mv_def2tzvpd"]["passed"])
        # Success criterion and relative condition are unaffected by this fault.
        self.assertTrue(gate["final_calculation_success"]["passed"])
        self.assertTrue(summary["relative_condition"]["passed"])

    def test_missing_gpu_result_fails_the_success_criterion(self) -> None:
        plan = _default_plan()
        del plan["sih4_td_seed"]["gpu4pyscf"]
        output_dir, summary = self._run(plan)
        gate = summary["provisional_cpu_gpu_gate"]
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["final_calculation_success"]["passed"])
        self.assertEqual(gate["final_calculation_success"]["gpu4pyscf_result_count"], 1)
        self.assertEqual(gate["final_calculation_success"]["required_case_count"], 2)
        # The one comparable case still passes on numbers alone.
        self.assertEqual(gate["evaluated_case_count"], 1)
        self.assertEqual(gate["passed_case_count"], 1)
        self.assertEqual(summary["missing"]["gpu4pyscf"], ["sih4_td_seed"])
        self.assertEqual(summary["engine_result_counts"]["gpu4pyscf"], 1)

        _, rows = _read_csv(output_dir / "gate1_case_metrics_gate1_test.csv")
        self.assertEqual(len(rows), 4)
        _, rows = _read_csv(output_dir / "gate1_performance_gate1_test.csv")
        self.assertEqual(len(rows), 5)

    def test_missing_cpu_counterpart_blocks_a_vacuous_gate_pass(self) -> None:
        # All GPU results exist and every comparable case passes, but one case
        # has no CPU counterpart: the gate must fail on pairing coverage
        # instead of passing vacuously.
        plan = _default_plan()
        del plan["sih4_td_seed"]["pyscf-cpu"]
        _, summary = self._run(plan)
        gate = summary["provisional_cpu_gpu_gate"]
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["final_calculation_success"]["passed"])
        self.assertEqual(gate["evaluated_case_count"], 1)
        self.assertEqual(gate["passed_case_count"], 1)
        self.assertFalse(gate["pairing_coverage"]["passed"])
        self.assertEqual(gate["pairing_coverage"]["required_case_count"], 2)
        self.assertEqual(gate["pairing_coverage"]["paired_case_count"], 1)

    def test_cpu_gpu_difference_larger_than_cpu_orca_is_flagged(self) -> None:
        plan = _default_plan()
        plan["h2_wb97mv_def2tzvpd"]["gpu4pyscf"] = _result(
            "h2_wb97mv_def2tzvpd", "gpu4pyscf", 1e-9, 1e-3
        )
        _, summary = self._run(plan)
        relative = summary["relative_condition"]
        self.assertFalse(relative["passed"])
        self.assertEqual(relative["compared_case_count"], 2)
        self.assertEqual(relative["energy_absolute_difference_violations"], [])
        self.assertEqual(
            relative["gradient_component_rmse_violations"], ["h2_wb97mv_def2tzvpd"]
        )

    def test_empty_runs_tree_still_writes_tables_and_warns(self) -> None:
        output_dir, summary = self._run({case_id: {} for case_id in CASES})
        self.assertEqual(
            summary["engine_result_counts"], {"orca": 0, "pyscf-cpu": 0, "gpu4pyscf": 0}
        )
        for pair in summary["pairs"]:
            self.assertEqual(pair["paired_case_count"], 0)
            self.assertIsNone(pair["worst_energy_absolute_difference_hartree"]["value"])
        self.assertFalse(summary["provisional_cpu_gpu_gate"]["passed"])
        self.assertEqual(sorted(summary["missing"]["gpu4pyscf"]), sorted(CASES))
        for name in ("gate1_case_metrics_gate1_test.csv", "gate1_performance_gate1_test.csv"):
            headers, rows = _read_csv(output_dir / name)
            self.assertTrue(headers)
            self.assertEqual(rows, [])

    def test_warning_is_printed_for_a_suite_without_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_path = _write_tree(root, {case_id: {} for case_id in CASES})
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "gate1_metrics.py"),
                    str(suite_path),
                    "--root",
                    str(root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("WARNING", completed.stdout)
            self.assertTrue(
                (root / "analysis" / "gate1_summary_gate1_test.json").is_file()
            )

    def test_fingerprint_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _default_plan()
            tampered = _result("h2_wb97mv_def2tzvpd", "gpu4pyscf", 1e-9, 1e-9)
            tampered["case"]["input_fingerprint_sha256"] = "0" * 64
            plan["h2_wb97mv_def2tzvpd"]["gpu4pyscf"] = tampered
            suite_path = _write_tree(root, plan)
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "input_fingerprint_sha256"):
                gate1_metrics.build_report(suite, root / "runs")

    def test_unconverged_result_is_never_compared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _default_plan()
            plan["h2_wb97mv_def2tzvpd"]["gpu4pyscf"]["converged"] = False
            suite_path = _write_tree(root, plan)
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "converged"):
                gate1_metrics.build_report(suite, root / "runs")

    def test_pair_metrics_reports_the_worst_component_location(self) -> None:
        left = _result("h2_wb97mv_def2tzvpd", "gpu4pyscf", 0.0, 0.0)
        right = _result("h2_wb97mv_def2tzvpd", "pyscf-cpu", 0.0, 0.0)
        left["gradient_hartree_per_bohr"][1][1] += 3e-5
        metrics = gate1_metrics.pair_metrics(left, right)
        self.assertEqual(metrics["gradient_max_atom_index_zero_based"], 1)
        self.assertEqual(metrics["gradient_max_axis"], "y")
        self.assertAlmostEqual(
            metrics["gradient_max_absolute_difference_hartree_per_bohr"], 3e-5
        )
        self.assertAlmostEqual(
            metrics["gradient_component_mae_hartree_per_bohr"], 3e-5 / 6
        )
        self.assertAlmostEqual(
            metrics["gradient_component_rmse_hartree_per_bohr"],
            (3e-5) / 6 ** 0.5,
        )


if __name__ == "__main__":
    unittest.main()
