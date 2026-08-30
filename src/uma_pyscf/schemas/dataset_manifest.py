"""Manifest for one verified ASE-LMDB teaching dataset.

The database files are generated artifacts and remain outside Git.  This
manifest is their portable integrity record: it binds every canonical label
file and split manifest to every LMDB shard, records the unit/sign conversion,
and states the fairchem compatibility contract that consumes charge and spin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ._fields import (
    reject_unknown_keys,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
)

__all__ = ["ASE_DATASET_MANIFEST_SCHEMA", "AseDatasetManifest"]

ASE_DATASET_MANIFEST_SCHEMA = "uma-pyscf-ase-dataset-manifest-v1"
ASE_DATASET_FORMAT = "ase-lmdb"
ASE_DATASET_TASK = "omol"
ASE_DATASET_REGRESSION_TASKS = "ef"
ASE_DATASET_UNITS = {
    "energy": "eV",
    "forces": "eV/angstrom",
    "positions": "angstrom",
}
ASE_FORCE_CONVENTION = "forces=-gradient"

_MANIFEST_KEYS = (
    "schema",
    "dataset_id",
    "format",
    "task",
    "regression_tasks",
    "units",
    "force_convention",
    "compatibility",
    "split",
    "record_checksums_sha256",
    "partitions",
)
_COMPATIBILITY_KEYS = (
    "ase_version",
    "ase_db_backends_version",
    "fairchem_core_version",
    "fairchem_a2g_data_keys",
)
_SPLIT_KEYS = ("id", "sha256")
_PARTITION_KEYS = ("record_count", "record_ids", "shards")
_SHARD_KEYS = ("path", "sha256", "record_count", "record_ids")
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, path: str) -> str:
    digest = require_str(value, path).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValidationError(f"{path} must be 64 hexadecimal characters; got {value!r}.")
    return digest


def _record_ids(value: object, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    ids = tuple(
        validate_record_id(require_str(raw, f"{path}[{index}]"))
        for index, raw in enumerate(require_sequence(value, path))
    )
    if not allow_empty and not ids:
        raise ValidationError(f"{path} must not be empty.")
    if tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
        raise ValidationError(f"{path} must be sorted and contain no duplicates.")
    return ids


def _compatibility(value: object) -> dict[str, Any]:
    raw = require_mapping(value, "dataset.compatibility")
    reject_unknown_keys(raw, _COMPATIBILITY_KEYS, "dataset.compatibility")
    for key in _COMPATIBILITY_KEYS:
        require_key(raw, key, "dataset.compatibility")
    data_keys = tuple(
        require_str(item, f"dataset.compatibility.fairchem_a2g_data_keys[{index}]")
        for index, item in enumerate(
            require_sequence(
                raw["fairchem_a2g_data_keys"],
                "dataset.compatibility.fairchem_a2g_data_keys",
            )
        )
    )
    if data_keys != ("charge", "spin"):
        raise ValidationError(
            "dataset.compatibility.fairchem_a2g_data_keys must be exactly "
            "['charge', 'spin']; without these keys fairchem defaults OMol state inputs."
        )
    return {
        "ase_version": require_str(raw["ase_version"], "dataset.compatibility.ase_version"),
        "ase_db_backends_version": require_str(
            raw["ase_db_backends_version"],
            "dataset.compatibility.ase_db_backends_version",
        ),
        "fairchem_core_version": require_str(
            raw["fairchem_core_version"], "dataset.compatibility.fairchem_core_version"
        ),
        "fairchem_a2g_data_keys": list(data_keys),
    }


def _split(value: object) -> dict[str, str]:
    raw = require_mapping(value, "dataset.split")
    reject_unknown_keys(raw, _SPLIT_KEYS, "dataset.split")
    for key in _SPLIT_KEYS:
        require_key(raw, key, "dataset.split")
    return {
        "id": validate_record_id(require_str(raw["id"], "dataset.split.id")),
        "sha256": _require_sha256(raw["sha256"], "dataset.split.sha256"),
    }


def _partitions(value: object) -> dict[str, dict[str, Any]]:
    raw = require_mapping(value, "dataset.partitions")
    if len(raw) < 2:
        raise ValidationError("dataset.partitions must contain at least two partitions.")
    checked: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for partition in sorted(raw):
        validate_record_id(partition)
        path = f"dataset.partitions.{partition}"
        block = require_mapping(raw[partition], path)
        reject_unknown_keys(block, _PARTITION_KEYS, path)
        for key in _PARTITION_KEYS:
            require_key(block, key, path)
        ids = _record_ids(block["record_ids"], f"{path}.record_ids", allow_empty=True)
        count = require_int(block["record_count"], f"{path}.record_count")
        if count != len(ids):
            raise ValidationError(
                f"{path}.record_count is {count} but record_ids contains {len(ids)} records."
            )
        overlap = sorted(seen.intersection(ids))
        if overlap:
            raise ValidationError(
                f"{path}.record_ids repeats records from another partition: {overlap!r}."
            )
        seen.update(ids)

        shards: list[dict[str, Any]] = []
        shard_ids: list[str] = []
        for index, item in enumerate(require_sequence(block["shards"], f"{path}.shards")):
            shard_path = f"{path}.shards[{index}]"
            shard = require_mapping(item, shard_path)
            reject_unknown_keys(shard, _SHARD_KEYS, shard_path)
            for key in _SHARD_KEYS:
                require_key(shard, key, shard_path)
            ids_in_shard = _record_ids(shard["record_ids"], f"{shard_path}.record_ids")
            shard_count = require_int(shard["record_count"], f"{shard_path}.record_count")
            if shard_count != len(ids_in_shard):
                raise ValidationError(
                    f"{shard_path}.record_count is {shard_count} but record_ids contains "
                    f"{len(ids_in_shard)} records."
                )
            relative = require_str(shard["path"], f"{shard_path}.path")
            if relative.startswith("/") or ".." in relative.split("/"):
                raise ValidationError(f"{shard_path}.path must be a safe relative path.")
            shards.append(
                {
                    "path": relative,
                    "sha256": _require_sha256(shard["sha256"], f"{shard_path}.sha256"),
                    "record_count": shard_count,
                    "record_ids": list(ids_in_shard),
                }
            )
            shard_ids.extend(ids_in_shard)
        if tuple(shard_ids) != ids:
            raise ValidationError(
                f"{path}.shards must list each partition record exactly once and in "
                "record_ids order."
            )
        checked[partition] = {
            "record_count": count,
            "record_ids": list(ids),
            "shards": shards,
        }
    if not seen:
        raise ValidationError("dataset.partitions contains no records.")
    return checked


@dataclass(frozen=True, kw_only=True)
class AseDatasetManifest:
    """Integrity and compatibility record for a complete ASE-LMDB export."""

    dataset_id: str
    compatibility: dict[str, Any]
    split: dict[str, str]
    record_checksums_sha256: dict[str, str]
    partitions: dict[str, dict[str, Any]]
    schema: str = ASE_DATASET_MANIFEST_SCHEMA
    format: str = ASE_DATASET_FORMAT
    task: str = ASE_DATASET_TASK
    regression_tasks: str = ASE_DATASET_REGRESSION_TASKS
    units: dict[str, str] | None = None
    force_convention: str = ASE_FORCE_CONVENTION

    def __post_init__(self) -> None:
        if self.schema != ASE_DATASET_MANIFEST_SCHEMA:
            raise ValidationError(
                f"dataset.schema must be {ASE_DATASET_MANIFEST_SCHEMA!r}; got {self.schema!r}."
            )
        for path, actual, expected in (
            ("dataset.format", self.format, ASE_DATASET_FORMAT),
            ("dataset.task", self.task, ASE_DATASET_TASK),
            ("dataset.regression_tasks", self.regression_tasks, ASE_DATASET_REGRESSION_TASKS),
            ("dataset.force_convention", self.force_convention, ASE_FORCE_CONVENTION),
        ):
            if actual != expected:
                raise ValidationError(f"{path} must be {expected!r}; got {actual!r}.")
        units = (
            dict(ASE_DATASET_UNITS)
            if self.units is None
            else require_mapping(self.units, "dataset.units")
        )
        if units != ASE_DATASET_UNITS:
            raise ValidationError(
                f"dataset.units must be exactly {ASE_DATASET_UNITS!r}; got {units!r}."
            )
        checksums_raw = require_mapping(
            self.record_checksums_sha256, "dataset.record_checksums_sha256"
        )
        checksums = {
            validate_record_id(record_id): _require_sha256(
                digest, f"dataset.record_checksums_sha256.{record_id}"
            )
            for record_id, digest in checksums_raw.items()
        }
        partitions = _partitions(self.partitions)
        assigned = {
            record_id for block in partitions.values() for record_id in block["record_ids"]
        }
        if assigned != set(checksums):
            raise ValidationError(
                "dataset.record_checksums_sha256 keys must exactly match records in partitions."
            )
        object.__setattr__(self, "dataset_id", validate_record_id(self.dataset_id))
        object.__setattr__(self, "compatibility", _compatibility(self.compatibility))
        object.__setattr__(self, "split", _split(self.split))
        object.__setattr__(self, "record_checksums_sha256", dict(sorted(checksums.items())))
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "units", dict(units))

    @property
    def record_count(self) -> int:
        return sum(block["record_count"] for block in self.partitions.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "format": self.format,
            "task": self.task,
            "regression_tasks": self.regression_tasks,
            "units": dict(self.units or {}),
            "force_convention": self.force_convention,
            "compatibility": dict(self.compatibility),
            "split": dict(self.split),
            "record_checksums_sha256": dict(self.record_checksums_sha256),
            "partitions": {
                name: {
                    "record_count": block["record_count"],
                    "record_ids": list(block["record_ids"]),
                    "shards": [dict(shard) for shard in block["shards"]],
                }
                for name, block in self.partitions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> AseDatasetManifest:
        raw = require_mapping(data, "dataset")
        reject_unknown_keys(raw, _MANIFEST_KEYS, "dataset")
        for key in _MANIFEST_KEYS:
            require_key(raw, key, "dataset")
        return cls(
            schema=raw["schema"],
            dataset_id=raw["dataset_id"],
            format=raw["format"],
            task=raw["task"],
            regression_tasks=raw["regression_tasks"],
            units=raw["units"],
            force_convention=raw["force_convention"],
            compatibility=raw["compatibility"],
            split=raw["split"],
            record_checksums_sha256=raw["record_checksums_sha256"],
            partitions=raw["partitions"],
        )
