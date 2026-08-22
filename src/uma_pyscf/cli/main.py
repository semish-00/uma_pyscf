#!/usr/bin/env python3
"""The ``uma-pyscf`` console entry point.

The CLI stays thin on purpose: it parses arguments, and the work belongs to the
module that owns it. Subcommands are declared once in ``SUBCOMMANDS``, so a
later milestone adds ``label``, ``qc``, ``dataset``, and the rest by appending
an entry and pointing it at that module's handler.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import platform
import sys

from .. import __version__
from ..datasets.cli import configure_split, run_split
from ..sampling.cli import configure_sample, run_sample
from ..schemas.cli import configure_validate_records, run_validate_records

__all__ = ["SUBCOMMANDS", "Subcommand", "build_parser", "main", "run_info"]

Handler = Callable[[argparse.Namespace], int]
Configurer = Callable[[argparse.ArgumentParser], None]

NO_SUBCOMMAND_EXIT_CODE = 2


@dataclass(frozen=True)
class Subcommand:
    """One registered ``uma-pyscf`` subcommand.

    ``configure`` receives the subparser and adds the subcommand's arguments;
    ``handler`` receives the parsed namespace and returns the process exit code.
    """

    name: str
    help: str
    handler: Handler
    configure: Configurer | None = None


def run_info(args: argparse.Namespace) -> int:
    """Print the installed package version and the interpreter it runs on."""
    del args  # `info` takes no arguments.
    print(f"uma_pyscf_version={__version__}")
    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    return 0


SUBCOMMANDS: tuple[Subcommand, ...] = (
    Subcommand(
        name="info",
        help="Print the package version, Python version, and platform.",
        handler=run_info,
    ),
    Subcommand(
        name="validate-record",
        help="Validate canonical label record JSON files against the current schema.",
        handler=run_validate_records,
        configure=configure_validate_records,
    ),
    Subcommand(
        name="sample",
        help="Generate structure candidates from a sampling config and QC their geometry.",
        handler=run_sample,
        configure=configure_sample,
    ),
    Subcommand(
        name="split",
        help="Assign a candidate manifest's records to partitions by whole groups.",
        handler=run_split,
        configure=configure_split,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with every registered subcommand attached."""
    parser = argparse.ArgumentParser(
        prog="uma-pyscf",
        description="Tooling for UMA fine-tuning on GPU4PySCF labels.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"uma-pyscf {__version__}",
        help="Print the package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")
    for command in SUBCOMMANDS:
        child = subparsers.add_parser(command.name, help=command.help, description=command.help)
        if command.configure is not None:
            command.configure(child)
        child.set_defaults(handler=command.handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code; never raises on bad input."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return NO_SUBCOMMAND_EXIT_CODE
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
