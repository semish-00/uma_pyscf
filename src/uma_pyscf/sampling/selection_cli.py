"""The ``uma-pyscf select`` subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..core.errors import UmaPyscfError
from .selection import run_selection

__all__ = ["configure_select", "run_select"]


def configure_select(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scores", required=True, metavar="<scores.json>")
    parser.add_argument("--config", required=True, metavar="<selection.yaml>")
    parser.add_argument("--output", required=True, metavar="<selection.json>")


def run_select(args: argparse.Namespace) -> int:
    try:
        manifest = run_selection(Path(args.scores), Path(args.config), Path(args.output))
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"select: ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"policies={len(manifest.policy_selections)} "
        f"union={len(manifest.union_record_ids)} output={args.output}"
    )
    return 0
