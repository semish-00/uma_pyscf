"""The ``uma-pyscf validate-record`` subcommand.

The CLI package only registers this; the checking itself belongs to the schema
that defines what a valid record is, which is why the handler lives here.
"""

from __future__ import annotations

import argparse

from ..core.errors import UmaPyscfError
from ..core.io import read_json
from .label_record import LabelRecord

__all__ = ["configure_validate_records", "run_validate_records"]


def configure_validate_records(parser: argparse.ArgumentParser) -> None:
    """Add the file arguments of ``validate-record`` to its subparser."""
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="<path>",
        help="Label record JSON file to validate. Repeatable.",
    )


def run_validate_records(args: argparse.Namespace) -> int:
    """Validate every named file, reporting one line each.

    Every file is reported even after one fails, so a batch check names all the
    bad records in a single run. The exit code is 1 if any file was rejected.
    """
    failures = 0
    for path in args.paths:
        try:
            LabelRecord.from_dict(read_json(path))
        except (UmaPyscfError, OSError, ValueError) as exc:
            failures += 1
            print(f"{path}: ERROR {exc}")
        else:
            print(f"{path}: ok")
    return 1 if failures else 0
