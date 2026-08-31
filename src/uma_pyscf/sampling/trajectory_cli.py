"""CLI for importing ASE trajectory frames into a candidate manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..core.errors import UmaPyscfError
from .trajectory_import import import_trajectory_candidates, write_trajectory_outputs

__all__ = ["configure_import_trajectory", "run_import_trajectory"]


def configure_import_trajectory(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", metavar="<config>")
    parser.add_argument("--source-root", required=True, metavar="<dir>")
    parser.add_argument("--output-dir", required=True, metavar="<dir>")


def run_import_trajectory(args: argparse.Namespace) -> int:
    try:
        manifest, report = import_trajectory_candidates(Path(args.config), Path(args.source_root))
        manifest_path, report_path = write_trajectory_outputs(
            manifest, report, Path(args.output_dir)
        )
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"accepted={report.count('accepted')} rejected={report.count('rejected')} "
        f"manifest={manifest_path} qc={report_path}"
    )
    return 0
