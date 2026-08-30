"""The ``uma-pyscf dataset`` command for verified ASE-LMDB export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

from ..core.errors import UmaPyscfError, ValidationError
from ..core.ids import sha256_of_file, validate_record_id
from ..core.io import read_json
from ..schemas._fields import (
    reject_unknown_keys,
    require_int,
    require_key,
    require_str,
    validated_json_object,
)
from ..schemas.dataset_manifest import AseDatasetManifest
from ..schemas.label_record import LabelRecord
from ..schemas.split_manifest import SplitManifest
from .ase_lmdb import export_ase_lmdb_dataset, verify_ase_lmdb_dataset

__all__ = [
    "ASE_DATASET_CONFIG_SCHEMA_VERSION",
    "configure_dataset",
    "configure_verify_dataset",
    "load_dataset_config",
    "run_dataset",
    "run_verify_dataset",
]

ASE_DATASET_CONFIG_SCHEMA_VERSION = 1
_CONFIG_KEYS = (
    "schema_version",
    "dataset_id",
    "created",
    "derived_from",
    "description",
    "task",
    "regression_tasks",
    "shard_size",
    "fairchem_core_version",
)
_REQUIRED_KEYS = (
    "schema_version",
    "dataset_id",
    "task",
    "regression_tasks",
    "shard_size",
    "fairchem_core_version",
)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"Dataset config {source} cannot be read: {exc}.") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"Dataset config {source} is not valid YAML: {exc}.") from exc
    config = validated_json_object(loaded, "config")
    reject_unknown_keys(config, _CONFIG_KEYS, "config")
    for key in _REQUIRED_KEYS:
        require_key(config, key, "config")
    version_value = require_int(config["schema_version"], "config.schema_version")
    if version_value != ASE_DATASET_CONFIG_SCHEMA_VERSION:
        raise ValidationError(
            f"config.schema_version must be {ASE_DATASET_CONFIG_SCHEMA_VERSION}; "
            f"got {version_value}."
        )
    validate_record_id(require_str(config["dataset_id"], "config.dataset_id"))
    if require_str(config["task"], "config.task") != "omol":
        raise ValidationError("config.task must be 'omol'.")
    if require_str(config["regression_tasks"], "config.regression_tasks") != "ef":
        raise ValidationError("config.regression_tasks must be 'ef' (energy plus forces).")
    shard_size = require_int(config["shard_size"], "config.shard_size")
    if shard_size < 1:
        raise ValidationError("config.shard_size must be at least 1.")
    require_str(config["fairchem_core_version"], "config.fairchem_core_version")
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


def _load_records(paths: tuple[Path, ...]) -> tuple[tuple[LabelRecord, ...], dict[str, str]]:
    records: list[LabelRecord] = []
    checksums: dict[str, str] = {}
    for path in paths:
        try:
            record = LabelRecord.from_dict(read_json(path))
        except (UmaPyscfError, OSError, ValueError) as exc:
            raise ValidationError(f"{path}: {exc}") from exc
        if record.record_id in checksums:
            raise ValidationError(f"Duplicate record id {record.record_id!r} in --records.")
        records.append(record)
        checksums[record.record_id] = sha256_of_file(path)
    return tuple(records), checksums


def configure_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, metavar="<config>")
    parser.add_argument("--split", required=True, metavar="<split>")
    parser.add_argument(
        "--records",
        required=True,
        nargs="+",
        metavar="<path>",
        help="Accepted canonical label record file or directory. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="<dir>",
        help="New versioned directory to publish after complete load-back verification.",
    )


def configure_verify_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, metavar="<manifest>")
    parser.add_argument(
        "--records",
        required=True,
        nargs="+",
        metavar="<path>",
        help="Canonical label record file or directory. Repeatable.",
    )
    parser.add_argument("--dataset-dir", required=True, metavar="<dir>")


def run_dataset(args: argparse.Namespace) -> int:
    try:
        config = load_dataset_config(Path(args.config))
        records, checksums = _load_records(_resolve_record_paths(args.records))
        split_path = Path(args.split)
        split = SplitManifest.from_dict(read_json(split_path))
        manifest, path = export_ase_lmdb_dataset(
            records,
            split,
            dataset_id=str(config["dataset_id"]),
            shard_size=int(config["shard_size"]),
            fairchem_core_version=str(config["fairchem_core_version"]),
            record_checksums_sha256=checksums,
            split_sha256=sha256_of_file(split_path),
            output_dir=Path(args.output_dir),
        )
        verify_ase_lmdb_dataset(
            manifest,
            records,
            record_checksums_sha256=checksums,
            output_dir=Path(args.output_dir),
        )
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"dataset={manifest.dataset_id} format={manifest.format} task={manifest.task} "
        f"records={manifest.record_count}"
    )
    for partition, block in manifest.partitions.items():
        print(
            f"partition={partition} records={block['record_count']} shards={len(block['shards'])}"
        )
    print(f"manifest={path}")
    return 0


def run_verify_dataset(args: argparse.Namespace) -> int:
    try:
        records, checksums = _load_records(_resolve_record_paths(args.records))
        manifest = AseDatasetManifest.from_dict(read_json(Path(args.manifest)))
        verify_ase_lmdb_dataset(
            manifest,
            records,
            record_checksums_sha256=checksums,
            output_dir=Path(args.dataset_dir),
        )
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.manifest}: ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"dataset={manifest.dataset_id} verified records={manifest.record_count} "
        f"partitions={len(manifest.partitions)}"
    )
    return 0
