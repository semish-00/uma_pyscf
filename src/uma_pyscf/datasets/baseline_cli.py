"""The ``uma-pyscf fit-baseline`` command."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

from ..core.errors import UmaPyscfError, ValidationError
from ..core.ids import sha256_of_file, validate_record_id
from ..core.io import read_json, write_json_atomic
from ..schemas._fields import (
    reject_unknown_keys,
    require_int,
    require_key,
    require_str,
    validated_json_object,
)
from ..schemas.composition_baseline import COMPOSITION_BASELINE_METHOD, CompositionBaseline
from ..schemas.label_record import LabelRecord
from ..schemas.split_manifest import SplitManifest
from .baseline import fit_atomic_composition_baseline

__all__ = [
    "BASELINE_CONFIG_SCHEMA_VERSION",
    "configure_fit_baseline",
    "load_baseline_config",
    "run_fit_baseline",
]

BASELINE_CONFIG_SCHEMA_VERSION = 1
_CONFIG_KEYS = (
    "schema_version",
    "baseline_id",
    "created",
    "derived_from",
    "description",
    "method",
    "fit_partition",
)
_REQUIRED_KEYS = ("schema_version", "baseline_id", "method", "fit_partition")


def load_baseline_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"Baseline config {source} cannot be read: {exc}.") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"Baseline config {source} is not valid YAML: {exc}.") from exc
    config = validated_json_object(loaded, "config")
    reject_unknown_keys(config, _CONFIG_KEYS, "config")
    for key in _REQUIRED_KEYS:
        require_key(config, key, "config")
    version = require_int(config["schema_version"], "config.schema_version")
    if version != BASELINE_CONFIG_SCHEMA_VERSION:
        raise ValidationError(
            f"config.schema_version must be {BASELINE_CONFIG_SCHEMA_VERSION}; got {version}."
        )
    validate_record_id(require_str(config["baseline_id"], "config.baseline_id"))
    if require_str(config["method"], "config.method") != COMPOSITION_BASELINE_METHOD:
        raise ValidationError(
            f"config.method must be {COMPOSITION_BASELINE_METHOD!r}."
        )
    validate_record_id(require_str(config["fit_partition"], "config.fit_partition"))
    return config


def _resolve_record_paths(raw_paths: list[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.glob("*.json"), key=lambda entry: entry.name)
            if not found:
                raise ValidationError(f"{path} contains no *.json label records.")
            paths.extend(found)
        else:
            paths.append(path)
    if not paths:
        raise ValidationError("--records named no files.")
    return tuple(paths)


def _fit_from_paths(
    config: dict[str, Any], records_paths: tuple[Path, ...], split_path: Path
) -> CompositionBaseline:
    records: list[LabelRecord] = []
    checksums: dict[str, str] = {}
    for path in records_paths:
        try:
            record = LabelRecord.from_dict(read_json(path))
        except (UmaPyscfError, OSError, ValueError) as exc:
            raise ValidationError(f"{path}: {exc}") from exc
        if record.record_id in checksums:
            raise ValidationError(f"Duplicate record id {record.record_id!r} in --records.")
        records.append(record)
        checksums[record.record_id] = sha256_of_file(path)
    split = SplitManifest.from_dict(read_json(split_path))
    return fit_atomic_composition_baseline(
        records,
        split,
        baseline_id=str(config["baseline_id"]),
        fit_partition=str(config["fit_partition"]),
        record_checksums_sha256=checksums,
    )


def configure_fit_baseline(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, metavar="<config>")
    parser.add_argument("--split", required=True, metavar="<split>")
    parser.add_argument(
        "--records",
        required=True,
        nargs="+",
        metavar="<path>",
        help="Accepted label record file or directory. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True, metavar="<dir>")


def run_fit_baseline(args: argparse.Namespace) -> int:
    try:
        config = load_baseline_config(Path(args.config))
        baseline = _fit_from_paths(
            config,
            _resolve_record_paths(args.records),
            Path(args.split),
        )
        output = Path(args.output_dir) / f"{baseline.baseline_id}.json"
        write_json_atomic(output, baseline.to_dict())
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"baseline={baseline.baseline_id} fit_partition={baseline.split['fit_partition']} "
        f"rank={baseline.design_rank} elements={','.join(baseline.elements)}"
    )
    for partition, metrics in baseline.metrics_by_partition.items():
        print(
            f"partition={partition} records={metrics['records']} "
            f"compositions={metrics['compositions']} rmse_hartree={metrics['rmse_hartree']:.12g} "
            f"max_abs_hartree={metrics['max_abs_error_hartree']:.12g}"
        )
    print(f"artifact={output}")
    return 0
