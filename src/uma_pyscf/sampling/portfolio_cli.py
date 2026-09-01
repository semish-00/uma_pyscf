"""CLI for score-independent candidate-portfolio assembly."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..core.errors import UmaPyscfError
from .portfolio import assemble_portfolio, write_portfolio_outputs

__all__ = ["configure_assemble_portfolio", "run_assemble_portfolio"]


def configure_assemble_portfolio(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", metavar="<config>")
    parser.add_argument("--source-root", required=True, metavar="<dir>")
    parser.add_argument("--output-dir", required=True, metavar="<dir>")


def run_assemble_portfolio(args: argparse.Namespace) -> int:
    try:
        manifest, report = assemble_portfolio(Path(args.config), Path(args.source_root))
        manifest_path, report_path = write_portfolio_outputs(
            manifest, report, Path(args.output_dir)
        )
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"sources={report['counts']['source_manifests']} "
        f"available={report['counts']['available']} selected={report['counts']['selected']} "
        f"manifest={manifest_path} report={report_path}"
    )
    return 0
