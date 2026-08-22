"""The ``uma-pyscf sample`` subcommand.

The CLI package registers this; the generation itself belongs to the module that
owns what a candidate is, which is why the handler lives here next to it.

Rejections are not failures. A run that generated its candidates and rejected
some of them did its job and exits 0, with one line per rejection so the reason
is visible without opening the QC report; the report has the same information
and more. Exit code 1 means the run could not be trusted -- a bad config, a
missing structure file, an unwritable output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..core.errors import UmaPyscfError
from .generate import generate_candidates, write_outputs

__all__ = ["configure_sample", "run_sample"]


def configure_sample(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of ``sample`` to its subparser."""
    parser.add_argument(
        "config",
        metavar="<config>",
        help="Sampling config (YAML or JSON) describing the structures and operations.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="<dir>",
        help="Directory to write the candidate manifest and the geometry QC report into.",
    )


def run_sample(args: argparse.Namespace) -> int:
    """Generate candidates from a config and write both records."""
    try:
        manifest, report = generate_candidates(Path(args.config))
        manifest_path, report_path = write_outputs(manifest, report, Path(args.output_dir))
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    for entry in report.entries:
        if entry["status"] == "rejected":
            print(f"rejected {entry['record_id']}: {entry['reason']}")
    print(
        f"accepted={report.count('accepted')} rejected={report.count('rejected')} "
        f"manifest={manifest_path} qc={report_path}"
    )
    return 0
