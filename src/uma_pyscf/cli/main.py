#!/usr/bin/env python3
"""The ``uma-pyscf`` console entry point.

The CLI stays thin on purpose: it parses arguments, and the work belongs to the
module that owns it. Subcommands are declared once in ``SUBCOMMANDS``, so a
later milestone adds ``label``, ``dataset``, and the rest by appending an entry
and pointing it at that module's handler.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import platform
import sys

from .. import __version__
from ..calculators.cli import configure_label, run_label
from ..datasets.ase_lmdb_cli import (
    configure_dataset,
    configure_verify_dataset,
    run_dataset,
    run_verify_dataset,
)
from ..datasets.baseline_cli import configure_fit_baseline, run_fit_baseline
from ..datasets.cli import configure_split, run_split
from ..inference.cli import (
    configure_evaluate_uma,
    configure_predict_uma,
    run_evaluate_uma,
    run_predict_uma,
)
from ..qc.cli import configure_qc, run_qc
from ..sampling.cli import configure_sample, run_sample
from ..sampling.portfolio_cli import configure_assemble_portfolio, run_assemble_portfolio
from ..sampling.selection_cli import configure_select, run_select
from ..sampling.trajectory_cli import configure_import_trajectory, run_import_trajectory
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
        name="assemble-portfolio",
        help="Assemble score-independent source quotas into one candidate portfolio.",
        handler=run_assemble_portfolio,
        configure=configure_assemble_portfolio,
    ),
    Subcommand(
        name="select",
        help="Select scored candidates with deterministic policies and parent quotas.",
        handler=run_select,
        configure=configure_select,
    ),
    Subcommand(
        name="import-trajectory",
        help="Import deterministically thinned ASE trajectory frames as unlabeled candidates.",
        handler=run_import_trajectory,
        configure=configure_import_trajectory,
    ),
    Subcommand(
        name="label",
        help="Label a candidate manifest with the frozen GPU4PySCF protocol.",
        handler=run_label,
        configure=configure_label,
    ),
    Subcommand(
        name="qc",
        help="Judge pending label records against a QC config and write the verdicts.",
        handler=run_qc,
        configure=configure_qc,
    ),
    Subcommand(
        name="split",
        help="Assign a candidate manifest's records to partitions by whole groups.",
        handler=run_split,
        configure=configure_split,
    ),
    Subcommand(
        name="dataset",
        help="Export accepted records to checksummed, load-back-verified ASE-LMDB shards.",
        handler=run_dataset,
        configure=configure_dataset,
    ),
    Subcommand(
        name="verify-dataset",
        help="Recheck an ASE-LMDB manifest, source records, shard hashes, and loaded rows.",
        handler=run_verify_dataset,
        configure=configure_verify_dataset,
    ),
    Subcommand(
        name="fit-baseline",
        help="Fit a train-only atomic composition energy baseline.",
        handler=run_fit_baseline,
        configure=configure_fit_baseline,
    ),
    Subcommand(
        name="evaluate-uma",
        help="Evaluate a base UMA model on verified ASE-LMDB partitions.",
        handler=run_evaluate_uma,
        configure=configure_evaluate_uma,
    ),
    Subcommand(
        name="predict-uma",
        help="Predict an unlabeled candidate manifest with a pinned UMA model.",
        handler=run_predict_uma,
        configure=configure_predict_uma,
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
