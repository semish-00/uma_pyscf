"""The ``uma-pyscf label`` production batch command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..core.errors import UmaPyscfError
from ..core.io import read_json
from ..schemas.candidate import CandidateManifest
from ..states.registry import load_state_registry
from .config import load_dft_config
from .runner import build_label_plan, run_label_batch
from .sharding import shard_candidate_manifest
from .subprocess_adapter import SubprocessGpu4PyscfAdapter

__all__ = ["configure_label", "run_label"]


def configure_label(parser: argparse.ArgumentParser) -> None:
    """Add label-pipeline arguments to the top-level parser."""
    parser.add_argument(
        "--config", required=True, metavar="<config>", help="Versioned DFT protocol YAML."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        metavar="<manifest>",
        help="Candidate manifest produced by `uma-pyscf sample`.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="<dir>",
        help="Run root for raw attempts, canonical records, ledger, and summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate protocol and scope, then print the deterministic execution plan.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry records already terminally failed in an existing ledger.",
    )
    parser.add_argument(
        "--state-registry",
        metavar="<registry>",
        help="Versioned state registry required when the manifest has non-default states.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        metavar="<count>",
        help="Split execution deterministically across this many independent GPU workers.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        metavar="<index>",
        help="Zero-based execution shard handled by this process.",
    )


def run_label(args: argparse.Namespace) -> int:
    """Validate, plan, or execute a resumable GPU4PySCF label batch."""
    try:
        config = load_dft_config(Path(args.config))
        manifest = CandidateManifest.from_dict(read_json(Path(args.manifest)))
        if args.shard_count != 1 or args.shard_index != 0:
            manifest = shard_candidate_manifest(
                manifest,
                shard_index=int(args.shard_index),
                shard_count=int(args.shard_count),
            )
        state_registry = (
            load_state_registry(Path(args.state_registry)) if args.state_registry else None
        )
        if args.dry_run:
            plan = build_label_plan(manifest, config, state_registry=state_registry)
            print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
            return 1 if plan["counts"]["blocked"] else 0
        summary = run_label_batch(
            manifest,
            config,
            Path(args.output_dir),
            SubprocessGpu4PyscfAdapter(),
            retry_failed=bool(args.retry_failed),
            state_registry=state_registry,
        )
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.manifest}: ERROR {exc}", file=sys.stderr)
        return 1
    counts = summary["counts"]
    print(
        f"completed={counts['completed']} skipped={counts['skipped']} "
        f"failed={counts['failed']} blocked={counts['blocked']} "
        f"summary={Path(args.output_dir) / 'summary.json'}"
    )
    return 1 if counts["failed"] or counts["blocked"] else 0
