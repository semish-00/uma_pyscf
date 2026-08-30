from __future__ import annotations

import copy
from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import RESULT_SCHEMA, case_record, load_case, multiplicity_to_pyscf_spin
from compare import compare_results
from parse_orca import normalized_result, parse_engrad
from prepare_orca import render_orca_input
from run_pyscf import build_plan, package_version


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "h2_wb97mv_def2tzvpd.json"
ENGRAD = ROOT / "tests" / "fixtures" / "h2.engrad"


class ManifestTests(unittest.TestCase):
    def test_multiplicity_conversion(self) -> None:
        self.assertEqual(multiplicity_to_pyscf_spin(1), 0)
        self.assertEqual(multiplicity_to_pyscf_spin(2), 1)
        self.assertEqual(multiplicity_to_pyscf_spin(3), 2)

    def test_manifest_resolves_structure_and_spin(self) -> None:
        case = load_case(CONFIG)
        self.assertEqual(case.spin_2s, 0)
        self.assertEqual(len(case.atoms), 2)
        self.assertEqual(case.basis, "def2-tzvpd")
        self.assertEqual(case.raw["orca"]["version"], "6.0.0")

    def test_inconsistent_electron_spin_parity_is_rejected(self) -> None:
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["multiplicity"] = 2
        raw["structure"] = str((ROOT / "structures" / "h2.xyz").resolve())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative"):
                load_case(path)

        raw["structure"] = "h2.xyz"
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            path = directory_path / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            (directory_path / "h2.xyz").write_text(
                (ROOT / "structures" / "h2.xyz").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_case(path)


class GenerationTests(unittest.TestCase):
    def test_orca_input_contains_reviewed_settings(self) -> None:
        text = render_orca_input(load_case(CONFIG))
        self.assertIn("! WB97M-V def2-TZVPD EnGrad", text)
        self.assertIn("VeryTightSCF", text)
        self.assertIn("SCNL", text)
        self.assertIn("NORI", text)
        self.assertIn("NOCOSX", text)
        self.assertIn("%pal\n  nprocs 4\nend", text)
        self.assertIn("* xyz 0 1", text)

    def test_pyscf_dry_run_does_not_import_pyscf(self) -> None:
        plan = build_plan(CONFIG, "cpu")
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["settings"]["grid_level"], 5)
        self.assertEqual(plan["settings"]["nlc_grid_level"], 5)

    def test_package_version_tries_distribution_aliases(self) -> None:
        def fake_version(name: str) -> str:
            if name == "gpu4pyscf-cuda12x":
                return "1.8.1"
            raise metadata.PackageNotFoundError(name)

        with patch("run_pyscf.metadata.version", side_effect=fake_version):
            self.assertEqual(
                package_version("gpu4pyscf", "gpu4pyscf-cuda12x"),
                "1.8.1",
            )


class ParserAndComparisonTests(unittest.TestCase):
    def test_parse_engrad(self) -> None:
        result = parse_engrad(ENGRAD)
        self.assertEqual(result["atom_count"], 2)
        self.assertAlmostEqual(result["energy_hartree"], -1.160768048981414)
        self.assertAlmostEqual(result["gradient_hartree_per_bohr"][1][2], -0.001072355070069)

    def test_parse_engrad_without_nonstandard_end_marker(self) -> None:
        text = ENGRAD.read_text(encoding="utf-8").replace(
            "#\n# The end of the file\n#\n",
            "",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orca_6_0_0.engrad"
            path.write_text(text, encoding="utf-8")
            result = parse_engrad(path)
        self.assertEqual(result["atom_count"], 2)
        self.assertAlmostEqual(result["energy_hartree"], -1.160768048981414)

    def test_normalized_orca_result_checks_coordinates(self) -> None:
        result = normalized_result(CONFIG, ENGRAD, None)
        self.assertEqual(result["schema"], RESULT_SCHEMA)
        self.assertEqual(result["engine"], "orca")
        self.assertIsNone(result["converged"])

    def test_normalized_orca_result_rejects_wrong_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.out"
            output.write_text(
                "Program Version 6.0.1\nORCA TERMINATED NORMALLY\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                normalized_result(CONFIG, ENGRAD, output)

    def test_identical_results_pass(self) -> None:
        case = load_case(CONFIG)
        base = {
            "schema": RESULT_SCHEMA,
            "engine": "left",
            "case": case_record(case),
            "converged": True,
            "energy_hartree": -1.0,
            "gradient_hartree_per_bohr": [[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
            "tolerances": case.tolerances,
            "tolerance_status": case.raw["tolerance_status"],
        }
        other = copy.deepcopy(base)
        other["engine"] = "right"
        report = compare_results(base, other)
        self.assertTrue(report["passed"])
        self.assertEqual(report["gradient_rms_difference_hartree_per_bohr"], 0.0)

    def test_large_energy_difference_fails(self) -> None:
        case = load_case(CONFIG)
        left = {
            "schema": RESULT_SCHEMA,
            "engine": "left",
            "case": case_record(case),
            "converged": True,
            "energy_hartree": -1.0,
            "gradient_hartree_per_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "tolerances": case.tolerances,
            "tolerance_status": case.raw["tolerance_status"],
        }
        right = copy.deepcopy(left)
        right["engine"] = "right"
        right["energy_hartree"] = -1.1
        self.assertFalse(compare_results(left, right)["passed"])


if __name__ == "__main__":
    unittest.main()
