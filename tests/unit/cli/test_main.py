"""The `uma-pyscf` entry point and its subcommand registry."""

from __future__ import annotations

import contextlib
import io
import platform
import unittest

import uma_pyscf
from uma_pyscf.cli.main import SUBCOMMANDS, build_parser, main


def run(argv: list[str]) -> tuple[int, str]:
    """Run the CLI with `argv` and capture its exit code and stdout."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = main(argv)
    return code, stream.getvalue()


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
