from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from collect_environment import collect, default_output, emit_yaml
from common import case_record, load_case, write_json
from gpu_smoke_check import CHECKS, MATMUL_TRACE_EXPECTED, run_checks
import run_suite

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUITE = ROOT / "suites" / "gpu_smoke_v1.json"
SMOKE_ORDER = (
    "h2_wb97mv_def2tzvpd",
    "sih4_td_seed",
    "sicl4_td_seed",
    "sih3_doublet_planar_seed",
    "h3si_gecl3_staggered_seed",
)


class AtomicWriteTests(unittest.TestCase):
    def test_write_json_writes_and_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "result.json"
            write_json(destination, {"value": 1})
            write_json(destination, {"value": 2})
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_failed_serialization_leaves_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            write_json(destination, {"value": 1})
            with self.assertRaises(TypeError):
                write_json(destination, {"value": object()})
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"value": 1})
            self.assertEqual(list(destination.parent.iterdir()), [destination])


class SmokeSuiteManifestTests(unittest.TestCase):
    def test_smoke_suite_matches_plan_c1(self) -> None:
        suite = json.loads(SMOKE_SUITE.read_text(encoding="utf-8"))
        self.assertEqual(suite["schema"], "crosscode-suite-v1")
        self.assertEqual(suite["engine_jobs_per_case"], ["gpu4pyscf"])
        self.assertEqual(suite["case_count"], len(suite["cases"]))
        self.assertEqual(
            tuple(case["case_id"] for case in suite["cases"]), SMOKE_ORDER
        )
        self.assertTrue(suite["execution_policy"]["stop_on_first_failure"])
        for entry in suite["cases"]:
            case = load_case(ROOT / entry["config"])
            self.assertEqual(case.case_id, entry["case_id"])
            self.assertEqual(case.multiplicity, entry["multiplicity"])

    def test_open_shell_case_is_present(self) -> None:
        suite = json.loads(SMOKE_SUITE.read_text(encoding="utf-8"))
        multiplicities = [case["multiplicity"] for case in suite["cases"]]
        self.assertIn(2, multiplicities)


def _copy_smoke_tree(destination: Path) -> Path:
    suite = json.loads(SMOKE_SUITE.read_text(encoding="utf-8"))
    (destination / "configs").mkdir(parents=True)
    (destination / "structures").mkdir()
    (destination / "suites").mkdir()
    for entry in suite["cases"]:
        config_path = ROOT / entry["config"]
        shutil.copy(config_path, destination / "configs" / config_path.name)
        structure = json.loads(config_path.read_text(encoding="utf-8"))["structure"]
        structure_path = (config_path.parent / structure).resolve()
        shutil.copy(structure_path, destination / "structures" / structure_path.name)
    suite_path = destination / "suites" / "gpu_smoke_v1.json"
    shutil.copy(SMOKE_SUITE, suite_path)
    return suite_path


def _runner_args(suite: Path, root: Path, **overrides: object) -> Namespace:
    values = {
        "suite": suite,
        "device": "gpu",
        "root": root,
        "dry_run": False,
        "keep_going": False,
        "overwrite": False,
        "case_timeout_minutes": None,
        "summary_output": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _succeed(config, device, output, timeout_seconds):
    write_json(output, {"stub": True})
    return subprocess.CompletedProcess(args=["stub"], returncode=0, stdout="ok", stderr="")


class RunSuiteTests(unittest.TestCase):
    def test_dry_run_validates_all_cases(self) -> None:
        code = run_suite.run_suite(_runner_args(SMOKE_SUITE, ROOT, dry_run=True))
        self.assertEqual(code, 0)

    def test_successful_run_writes_results_ledger_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_path = _copy_smoke_tree(root)
            summary_path = root / "session.json"
            with mock.patch.object(run_suite, "execute_case", side_effect=_succeed):
                code = run_suite.run_suite(
                    _runner_args(suite_path, root, summary_output=summary_path)
                )
            self.assertEqual(code, 0)
            for case_id in SMOKE_ORDER:
                run_dir = root / "runs" / case_id / "gpu4pyscf"
                self.assertTrue((run_dir / "result.json").is_file())
                ledger = (run_dir / "attempts.jsonl").read_text(encoding="utf-8")
                (entry,) = [json.loads(line) for line in ledger.splitlines()]
                self.assertEqual(entry["status"], "succeeded")
                self.assertEqual(entry["attempt_index"], 1)
            session = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(session["status_counts"], {"succeeded": 5})

    def test_failure_stops_suite_and_preserves_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_path = _copy_smoke_tree(root)
            summary_path = root / "session.json"
            calls = {"count": 0}

            def flaky(config, device, output, timeout_seconds):
                calls["count"] += 1
                if calls["count"] == 2:
                    return subprocess.CompletedProcess(
                        args=["stub"], returncode=1, stdout="", stderr="SCF blew up"
                    )
                return _succeed(config, device, output, timeout_seconds)

            with mock.patch.object(run_suite, "execute_case", side_effect=flaky):
                code = run_suite.run_suite(
                    _runner_args(suite_path, root, summary_output=summary_path)
                )
            self.assertEqual(code, 1)
            session = json.loads(summary_path.read_text(encoding="utf-8"))
            statuses = {row["case_id"]: row["status"] for row in session["cases"]}
            self.assertEqual(statuses[SMOKE_ORDER[0]], "succeeded")
            self.assertEqual(statuses[SMOKE_ORDER[1]], "failed")
            for case_id in SMOKE_ORDER[2:]:
                self.assertEqual(statuses[case_id], "not_attempted")
            failed_dir = root / "runs" / SMOKE_ORDER[1] / "gpu4pyscf"
            self.assertFalse((failed_dir / "result.json").exists())
            (entry,) = [
                json.loads(line)
                for line in (failed_dir / "attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(entry["status"], "failed")
            self.assertIn("SCF blew up", entry["stderr_tail"])

    def test_existing_results_are_skipped_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_path = _copy_smoke_tree(root)
            first = root / "runs" / SMOKE_ORDER[0] / "gpu4pyscf" / "result.json"
            write_json(first, {"original": True})
            summary_path = root / "session.json"
            with mock.patch.object(run_suite, "execute_case", side_effect=_succeed):
                code = run_suite.run_suite(
                    _runner_args(suite_path, root, summary_output=summary_path)
                )
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(first.read_text(encoding="utf-8")), {"original": True}
            )
            session = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                session["status_counts"],
                {"skipped_existing_result": 1, "succeeded": 4},
            )


class CollectEnvironmentTests(unittest.TestCase):
    def test_inventory_collects_with_explicit_nulls(self) -> None:
        errors: dict[str, str] = {}
        inventory = collect(errors)
        self.assertEqual(inventory["schema"], "gpu4pyscf-environment-v1")
        self.assertIn("pyscf", inventory["packages"])
        self.assertIn("cupy", inventory["packages"])
        self.assertIn("cutensor", inventory["packages"])
        # Missing probes must be recorded, never silently dropped.
        for key in ("nvidia_driver", "gpus", "libxc"):
            self.assertIn(key, inventory)

    def test_yaml_emission_round_trips(self) -> None:
        data = {
            "schema": "x",
            "empty": None,
            "flag": True,
            "nested": {"a": 1, "b": "text with: colon and \"quotes\""},
            "gpus": [{"index": 0, "name": "GPU A"}, {"index": 1, "name": "GPU B"}],
            "plain_list": [1, 2, 3],
        }
        text = emit_yaml(data)
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed here.")
        self.assertEqual(yaml.safe_load(text), data)

    def test_default_output_sanitizes_host(self) -> None:
        path = default_output("GPU Host #1")
        self.assertEqual(path.name, "gpu4pyscf-gpu-host-1.yaml")
        self.assertEqual(path.parent.name, "environments")
        self.assertEqual(default_output(None).name, "gpu4pyscf-unknown-host.yaml")


class GpuSmokeCheckTests(unittest.TestCase):
    def test_matmul_trace_reference(self) -> None:
        self.assertEqual(MATMUL_TRACE_EXPECTED, 1060.0)

    def test_report_structure_and_fail_fast(self) -> None:
        report = run_checks(device_id=0, grid_level=1, nlc_grid_level=1)
        self.assertEqual(
            [row["name"] for row in report["checks"]], [name for name, _ in CHECKS]
        )
        for row in report["checks"]:
            self.assertIn(row["status"], ("passed", "failed", "skipped"))
        failed_indexes = [
            index
            for index, row in enumerate(report["checks"])
            if row["status"] == "failed"
        ]
        if failed_indexes:
            self.assertFalse(report["passed"])
            # Everything after the first failure must be skipped, not attempted.
            for row in report["checks"][failed_indexes[0] + 1 :]:
                self.assertEqual(row["status"], "skipped")


class SummarizeEnginePairTests(unittest.TestCase):
    def test_gpu_vs_cpu_pairing(self) -> None:
        config = ROOT / "configs" / "h2_wb97mv_def2tzvpd.json"
        case = load_case(config)
        base = {
            "schema": "crosscode-result-v1",
            "case": case_record(case),
            "converged": True,
            "energy_hartree": -1.16,
            "gradient_hartree_per_bohr": [[0.0, 0.0, 0.001], [0.0, 0.0, -0.001]],
            "s2": None,
            "s2_deviation": None,
            "wall_time_seconds": 4.0,
            "tolerances": case.tolerances,
            "tolerance_status": case.raw["tolerance_status"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "validation/orca_gpu4pyscf/runs" / case.case_id
            for engine in ("pyscf-cpu", "gpu4pyscf"):
                result = dict(base, engine=engine)
                write_json(run_dir / engine / "result.json", result)
            suite_path = root / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "schema": "crosscode-suite-v1",
                        "suite_id": "pair_test",
                        "cases": [
                            {
                                "case_id": case.case_id,
                                "category": "gpu_smoke",
                                "config": "unused",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "summary.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "summarize_suite.py"),
                    str(suite_path),
                    "--root",
                    str(root),
                    "--left-engine",
                    "gpu4pyscf",
                    "--right-engine",
                    "pyscf-cpu",
                    "--write-comparisons",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], "crosscode-suite-summary-v2")
            self.assertEqual(summary["left_engine"], "gpu4pyscf")
            self.assertEqual(summary["paired_results"], 1)
            self.assertEqual(summary["paired_passed"], 1)
            row = summary["rows"][0]
            self.assertEqual(row["comparison"]["left"]["wall_time_seconds"], 4.0)
            comparison_path = run_dir / "gpu4pyscf-vs-pyscf-cpu.json"
            self.assertTrue(comparison_path.is_file())
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(comparison["left_engine"], "gpu4pyscf")
            self.assertEqual(comparison["right_engine"], "pyscf-cpu")


if __name__ == "__main__":
    unittest.main()
