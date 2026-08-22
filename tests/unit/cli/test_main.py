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


def run(argv: list[str]) -> tuple[int, str]:
    """Run the CLI with `argv` and capture its exit code and stdout."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = main(argv)
    return code, stream.getvalue()


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
