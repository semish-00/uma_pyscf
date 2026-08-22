"""The `uma-pyscf` entry point and its subcommand registry."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import platform
import tempfile
import unittest

import uma_pyscf
from uma_pyscf.cli.main import SUBCOMMANDS, build_parser, main
from uma_pyscf.core.io import write_json_atomic
from uma_pyscf.schemas.label_record import (
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)
from uma_pyscf.schemas.split_manifest import SplitManifest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "sampling" / "example_bond_scan_v1.yaml"
EXAMPLE_XYZ = REPO_ROOT / "configs" / "sampling" / "structures" / "sih4_seed_example.xyz"
EXAMPLE_SPLIT_CONFIG = REPO_ROOT / "configs" / "datasets" / "example_parent_split_v1.yaml"
EXAMPLE_CANDIDATES = "example_bond_scan_v1_candidates.json"
MULTIPLICITY_SPLIT_CONFIG = (
    "schema_version: 1\n"
    "split_id: example_multiplicity_split_v1\n"
    "axis: multiplicity\n"
    "seed: 20260822\n"
    "partitions:\n"
    "  train: 0.5\n"
    "  holdout: 0.5\n"
)


def run(argv: list[str]) -> tuple[int, str]:
    """Run the CLI with `argv` and capture its exit code and stdout."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = main(argv)
    return code, stream.getvalue()


def run_capturing_stderr(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI with `argv` and capture its exit code, stdout, and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def h2_record_dict() -> dict[str, object]:
    """Return a valid minimal H2 label record in its on-disk form."""
    return LabelRecord(
        record_id="h2_neutral_singlet",
        structure=Structure(
            atomic_numbers=(1, 1),
            positions_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74144)),
        ),
        state=ElectronicState(charge=0, multiplicity=1, spin_2s=0),
        method=Method(
            functional="wb97m-v",
            basis="def2-tzvpd",
            ecp=None,
            aux_basis=None,
            grid_level=3,
            nlc_grid_level=1,
            grid_response=True,
            density_fit=False,
            scf_conv_tol=1e-10,
            scf_max_cycle=200,
        ),
        engine=Engine(name="gpu4pyscf", versions={"pyscf": "2.6.2"}),
        results=Results(
            energy_hartree=-1.1730407,
            gradient_hartree_per_bohr=((0.0, 0.0, -0.0123456), (0.0, 0.0, 0.0123456)),
            converged=True,
        ),
        raw=RawArtifact(),
        qc=QcState(status="pending"),
    ).to_dict()


class InfoCommandTests(unittest.TestCase):
    def test_info_succeeds_and_reports_the_package_version(self) -> None:
        code, output = run(["info"])
        self.assertEqual(code, 0)
        self.assertIn(f"uma_pyscf_version={uma_pyscf.__version__}", output)

    def test_info_reports_interpreter_and_platform(self) -> None:
        _, output = run(["info"])
        self.assertIn(f"python_version={platform.python_version()}", output)
        self.assertIn("platform=", output)

    def test_info_takes_no_arguments(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                main(["info", "--unexpected"])
        self.assertEqual(caught.exception.code, 2)


class TopLevelTests(unittest.TestCase):
    def test_no_subcommand_prints_help_and_returns_two(self) -> None:
        code, output = run([])
        self.assertEqual(code, 2)
        self.assertIn("usage: uma-pyscf", output)
        self.assertIn("info", output)

    def test_version_flag_exits_zero_with_the_package_version(self) -> None:
        stream = io.StringIO()
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stdout(stream):
                main(["--version"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn(uma_pyscf.__version__, stream.getvalue())

    def test_unknown_subcommand_exits_two(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                main(["nonexistent"])
        self.assertEqual(caught.exception.code, 2)


class ValidateRecordCommandTests(unittest.TestCase):
    def test_valid_records_report_ok_and_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json_atomic(first, h2_record_dict())
            write_json_atomic(second, h2_record_dict() | {"record_id": "h2_stretched"})
            code, output = run(["validate-record", str(first), str(second)])
        self.assertEqual(code, 0)
        self.assertEqual(output.splitlines(), [f"{first}: ok", f"{second}: ok"])

    def test_one_broken_record_fails_the_run_and_every_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.json"
            bad = Path(directory) / "bad.json"
            write_json_atomic(good, h2_record_dict())
            write_json_atomic(bad, h2_record_dict() | {"schema": "uma-pyscf-label-record-v0"})
            code, output = run(["validate-record", str(good), str(bad)])
        lines = output.splitlines()
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], f"{good}: ok")
        self.assertTrue(lines[1].startswith(f"{bad}: ERROR "))
        self.assertIn("uma-pyscf-label-record-v1", lines[1])

    def test_a_forbidden_force_field_is_reported_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            record = h2_record_dict()
            results = dict(record["results"])  # type: ignore[arg-type]
            results["forces_ev_per_angstrom"] = [[0.0, 0.0, 0.63], [0.0, 0.0, -0.63]]
            record["results"] = results
            write_json_atomic(path, record)
            code, output = run(["validate-record", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("forces_ev_per_angstrom", output)
        self.assertIn("export layer", output)

    def test_unreadable_and_missing_files_are_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            malformed = Path(directory) / "malformed.json"
            malformed.write_text("{not json", encoding="utf-8")
            code, output = run(["validate-record", str(missing), str(malformed)])
        lines = output.splitlines()
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(": ERROR " in line for line in lines))

    def test_written_records_stay_readable_as_plain_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json_atomic(path, h2_record_dict())
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["units"]["gradient"],
                "hartree/bohr",
            )

    def test_at_least_one_path_is_required(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                main(["validate-record"])
        self.assertEqual(caught.exception.code, 2)


class SampleCommandTests(unittest.TestCase):
    def test_the_reference_config_generates_both_files_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, output = run(["sample", str(EXAMPLE_CONFIG), "--output-dir", directory])
            manifest = Path(directory) / "example_bond_scan_v1_candidates.json"
            report = Path(directory) / "example_bond_scan_v1_geometry_qc.json"
            self.assertEqual(code, 0)
            self.assertTrue(manifest.is_file())
            self.assertTrue(report.is_file())
            self.assertEqual(
                output.splitlines()[-1],
                f"accepted=7 rejected=0 manifest={manifest} qc={report}",
            )
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["schema"],
                "uma-pyscf-candidate-manifest-v1",
            )

    def test_a_rejected_candidate_is_printed_and_still_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sih4.xyz").write_text(
                EXAMPLE_XYZ.read_text(encoding="utf-8"), encoding="utf-8"
            )
            config = root / "collapsed.yaml"
            config.write_text(
                "schema_version: 1\n"
                "sampling_id: cli_reject_v1\n"
                "structures:\n"
                "  - id: sih4_seed\n"
                "    xyz_path: sih4.xyz\n"
                "operations:\n"
                "  - kind: bond_scan\n"
                "    structure: sih4_seed\n"
                "    charge: 0\n"
                "    multiplicity: 1\n"
                "    anchor_index: 0\n"
                "    moved_index: 1\n"
                "    factors: [0.5]\n",
                encoding="utf-8",
            )
            code, output = run(["sample", str(config), "--output-dir", str(root / "out")])
        lines = output.splitlines()
        self.assertEqual(code, 0)
        self.assertTrue(lines[0].startswith("rejected cli_reject_v1_sih4_seed_bond01_x0p5: "))
        self.assertIn("minimum distance", lines[0])
        self.assertTrue(lines[-1].startswith("accepted=0 rejected=1 "))

    def test_a_broken_config_reports_an_error_and_exits_one(self) -> None:
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.yaml"
            with contextlib.redirect_stderr(stream):
                code, output = run(["sample", str(missing), "--output-dir", directory])
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("ERROR", stream.getvalue())

    def test_the_output_directory_is_required(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                main(["sample", str(EXAMPLE_CONFIG)])
        self.assertEqual(caught.exception.code, 2)


class SplitCommandTests(unittest.TestCase):
    """The `split` subcommand, run against the committed P2.2 example candidates.

    Both outcomes are exercised on purpose. The committed parent split config
    is *refused* on this candidate set, because its seven records all descend
    from one seed structure and therefore form one indivisible parent group;
    that refusal is the leakage guarantee working, not a bug to route around.
    A multiplicity split of the same records succeeds, and separates the
    charge/spin siblings deliberately, which is what a spin holdout is for.
    """

    def candidates(self, directory: Path) -> Path:
        """Generate the P2.2 example candidates into `directory`."""
        code, _ = run(["sample", str(EXAMPLE_CONFIG), "--output-dir", str(directory)])
        self.assertEqual(code, 0)
        return directory / EXAMPLE_CANDIDATES

    def test_the_example_parent_split_is_refused_because_there_is_one_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = self.candidates(root)
            code, output, errors = run_capturing_stderr(
                [
                    "split",
                    "--config",
                    str(EXAMPLE_SPLIT_CONFIG),
                    "--candidates",
                    str(candidates),
                    "--output-dir",
                    str(root / "splits"),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("1 distinct group", errors)
        self.assertIn("'sih4_seed'", errors)
        self.assertIn("3 partitions", errors)
        self.assertIn("more distinct groups", errors)
        self.assertFalse((Path(directory) / "splits").exists())

    def test_a_multiplicity_split_of_the_same_candidates_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = self.candidates(root)
            config = root / "multiplicity_split.yaml"
            config.write_text(MULTIPLICITY_SPLIT_CONFIG, encoding="utf-8")
            code, output = run(
                [
                    "split",
                    "--config",
                    str(config),
                    "--candidates",
                    str(candidates),
                    "--output-dir",
                    str(root / "splits"),
                ]
            )
            written = root / "splits" / "example_multiplicity_split_v1.json"
            self.assertEqual(code, 0)
            self.assertTrue(written.is_file())
            split = SplitManifest.from_dict(json.loads(written.read_text(encoding="utf-8")))
        lines = output.splitlines()
        self.assertEqual(lines[-1], f"split={written}")
        self.assertIn("axis=multiplicity groups=2 records=7", lines)
        self.assertTrue(any(line.startswith("partition=train ") for line in lines))
        self.assertTrue(any(line.startswith("partition=holdout ") for line in lines))

        assigned = sorted(record for ids in split.record_assignments.values() for record in ids)
        self.assertEqual(len(assigned), 7)
        self.assertEqual(len(set(assigned)), 7)
        self.assertEqual(sorted(split.group_assignments), ["1", "2"])

        # The six singlets -- the scan, the displacements, the neutral state --
        # travel together; the cation doublet is the deliberate holdout.
        singlets = split.record_assignments[split.group_assignments["1"]]
        doublets = split.record_assignments[split.group_assignments["2"]]
        self.assertEqual(len(singlets), 6)
        self.assertEqual(doublets, ("example_bond_scan_v1_sih4_seed_q1m2",))
        self.assertNotEqual(split.group_assignments["1"], split.group_assignments["2"])

    def test_the_split_manifest_is_byte_identical_on_a_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = self.candidates(root)
            config = root / "multiplicity_split.yaml"
            config.write_text(MULTIPLICITY_SPLIT_CONFIG, encoding="utf-8")
            argv = [
                "split",
                "--config",
                str(config),
                "--candidates",
                str(candidates),
                "--output-dir",
                str(root / "splits"),
            ]
            written = root / "splits" / "example_multiplicity_split_v1.json"
            run(argv)
            first = written.read_bytes()
            run(argv)
            second = written.read_bytes()
        self.assertEqual(first, second)
        self.assertNotIn(b"timestamp", first)

    def test_a_missing_candidate_file_is_reported_and_exits_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, output, errors = run_capturing_stderr(
                [
                    "split",
                    "--config",
                    str(EXAMPLE_SPLIT_CONFIG),
                    "--candidates",
                    str(root / "absent.json"),
                    "--output-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("ERROR", errors)

    def test_every_argument_is_required(self) -> None:
        for argv in (
            ["split", "--candidates", "c.json", "--output-dir", "out"],
            ["split", "--config", "s.yaml", "--output-dir", "out"],
            ["split", "--config", "s.yaml", "--candidates", "c.json"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as caught:
                    with contextlib.redirect_stderr(io.StringIO()):
                        main(argv)
                self.assertEqual(caught.exception.code, 2)


class RegistryTests(unittest.TestCase):
    def test_every_registered_subcommand_is_reachable_and_documented(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        for command in SUBCOMMANDS:
            with self.subTest(subcommand=command.name):
                self.assertIn(command.name, help_text)
                self.assertTrue(command.help.strip())

    def test_subcommand_names_are_unique(self) -> None:
        names = [command.name for command in SUBCOMMANDS]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
