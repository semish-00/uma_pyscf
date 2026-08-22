"""The ``uma-pyscf split`` subcommand and the split config it reads.

The CLI package only registers this; deciding what a split *is* belongs to the
module that owns splits, which is why the handler and the config parser live
here next to :mod:`uma_pyscf.datasets.splits`.

The config parser is written out here rather than shared with the sampling
milestone on purpose. The two schemas have nothing in common beyond
``schema_version`` and a YAML file extension, and importing across sibling
modules would put a dependency edge where the repository structure plan forbids
one (``sampling`` and ``datasets`` are peers; both may depend on ``schemas``,
neither on the other). The cost is a dozen lines of key checking; the benefit is
that changing the sampling config format cannot silently change what a split
config means.

Exit code 1 means the run could not be trusted: an unreadable or unknown config
key, a candidate manifest that does not validate, or -- the interesting one --
a dataset with fewer distinct groups on the requested axis than the config has
partitions. That last refusal is not a defect to route around. It is the
leakage guarantee reporting that the requested holdout cannot be cut from this
data without dividing a group.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

from ..core.errors import UmaPyscfError, ValidationError
from ..core.ids import canonical_json_fingerprint, validate_record_id
from ..core.io import read_json, write_json_atomic
from ..schemas._fields import (
    reject_unknown_keys,
    require_int,
    require_key,
    require_str,
    validated_json_object,
)
from ..schemas.candidate import CandidateManifest
from ..schemas.split_manifest import SplitManifest, validate_axis, validate_partitions
from .splits import generate_split, split_items_from_candidate_manifest

__all__ = [
    "SPLIT_CONFIG_SCHEMA_VERSION",
    "configure_split",
    "load_split_config",
    "run_split",
    "split_from_candidates",
    "write_split",
]

SPLIT_CONFIG_SCHEMA_VERSION = 1

_TOP_LEVEL_KEYS = (
    "schema_version",
    "split_id",
    "created",
    "derived_from",
    "description",
    "axis",
    "seed",
    "partitions",
)
_REQUIRED_TOP_LEVEL_KEYS = ("schema_version", "split_id", "axis", "seed", "partitions")


def load_split_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a split config, returning it verbatim.

    The file is read with ``yaml.safe_load``, which also accepts JSON, and the
    result is checked to be JSON-safe: a bare YAML date such as
    ``created: 2026-08-22`` parses into a ``date`` object rather than a string,
    so it is refused here with its key named (quote it, and it is a string).

    Nothing is filled in and nothing is normalized. Every key is checked, and an
    unknown one stops the run rather than being ignored -- a misspelled
    ``partitions`` would otherwise produce a split with different fractions from
    the ones its author wrote down.

    The declaration order of ``partitions`` is preserved, because YAML mappings
    load in document order and that order breaks deficit ties during assignment.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Split config {source} cannot be read: {exc}.") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"Split config {source} is not valid YAML: {exc}.") from exc
    config = validated_json_object(loaded, "config")
    reject_unknown_keys(config, _TOP_LEVEL_KEYS, "config")
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        require_key(config, key, "config")

    version = require_int(config["schema_version"], "config.schema_version")
    if version != SPLIT_CONFIG_SCHEMA_VERSION:
        raise ValidationError(
            f"config.schema_version must be {SPLIT_CONFIG_SCHEMA_VERSION}; got {version}."
        )
    validate_record_id(require_str(config["split_id"], "config.split_id"))
    validate_axis(config["axis"], "config.axis")
    require_int(config["seed"], "config.seed")
    validate_partitions(config["partitions"], "config.partitions")
    return config


def split_from_candidates(config: dict[str, Any], candidates_path: str | Path) -> SplitManifest:
    """Build the split a config asks for from a candidate manifest file.

    The manifest is read through :meth:`CandidateManifest.from_dict`, so it is
    validated in full before anything is split: a candidate set that is not a
    valid record is not a dataset a split may describe.

    ``source.sha256`` fingerprints the manifest's *content* through
    :func:`~uma_pyscf.core.ids.canonical_json_fingerprint` rather than the file
    bytes, which makes the digest independent of the file's formatting and
    recomputable by anyone holding the manifest.
    """
    manifest = CandidateManifest.from_dict(read_json(candidates_path))
    return generate_split(
        split_items_from_candidate_manifest(manifest),
        split_id=str(config["split_id"]),
        axis=str(config["axis"]),
        partitions=config["partitions"],
        seed=int(config["seed"]),
        source_id=manifest.sampling_id,
        source_sha256=canonical_json_fingerprint(manifest.to_dict()),
    )


def write_split(split: SplitManifest, output_dir: str | Path) -> Path:
    """Write the split manifest atomically and return its path.

    The file is named after the split id, so one directory holds several splits
    of the same dataset. Production split ids follow the naming rule of the
    repository structure plan -- ``split_<dataset_id>_<axis>_v<N>`` -- which
    makes that file name the one the plan specifies.
    """
    path = Path(output_dir) / f"{split.split_id}.json"
    write_json_atomic(path, split.to_dict())
    return path


def configure_split(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of ``split`` to its subparser."""
    parser.add_argument(
        "--config",
        required=True,
        metavar="<config>",
        help="Split config (YAML or JSON) naming the axis, seed, and partitions.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        metavar="<path>",
        help="Candidate manifest JSON file holding the records to split.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="<dir>",
        help="Directory to write the split manifest into.",
    )


def run_split(args: argparse.Namespace) -> int:
    """Split a candidate manifest according to a config and write the result."""
    try:
        config = load_split_config(Path(args.config))
        split = split_from_candidates(config, Path(args.candidates))
        path = write_split(split, Path(args.output_dir))
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    for name in split.partitions:
        print(
            f"partition={name} records={len(split.record_assignments[name])} "
            f"groups={split.groups_in(name)}"
        )
    print(f"axis={split.axis} groups={len(split.group_assignments)} records={split.record_count}")
    print(f"split={path}")
    return 0
