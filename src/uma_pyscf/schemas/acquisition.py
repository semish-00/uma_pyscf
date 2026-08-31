"""Versioned records for acquisition scores and deterministic selections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, validate_record_id
from ._fields import (
    optional_int,
    optional_str,
    reject_unknown_keys,
    require_finite_float,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    validated_json_object,
)

__all__ = [
    "ACQUISITION_SCORE_SCHEMA",
    "SELECTION_MANIFEST_SCHEMA",
    "AcquisitionScoreManifest",
    "AcquisitionScoreRecord",
    "SelectionManifest",
]

ACQUISITION_SCORE_SCHEMA = "uma-pyscf-acquisition-scores-v1"
SELECTION_MANIFEST_SCHEMA = "uma-pyscf-selection-manifest-v1"

_HEX_DIGITS = frozenset("0123456789abcdef")
_SCORE_RECORD_KEYS = (
    "record_id",
    "parent_id",
    "trajectory_id",
    "frame_index",
    "scores",
    "provenance",
)
_SCORE_MANIFEST_KEYS = ("schema", "score_id", "source", "records")
_SELECTION_MANIFEST_KEYS = (
    "schema",
    "selection_id",
    "source",
    "config_sha256",
    "config",
    "policy_selections",
    "union_record_ids",
)
_SOURCE_KEYS = ("id", "sha256")


def _require_sha256(value: object, path: str) -> str:
    digest = require_str(value, path).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValidationError(f"{path} must be 64 hexadecimal characters; got {value!r}.")
    return digest


def _validated_source(value: object, path: str) -> dict[str, str]:
    mapping = require_mapping(value, path)
    reject_unknown_keys(mapping, _SOURCE_KEYS, path)
    source_id = validate_record_id(require_str(require_key(mapping, "id", path), f"{path}.id"))
    digest = _require_sha256(require_key(mapping, "sha256", path), f"{path}.sha256")
    return {"id": source_id, "sha256": digest}


def _validated_scores(value: object, path: str) -> dict[str, float]:
    mapping = require_mapping(value, path)
    if not mapping:
        raise ValidationError(f"{path} must contain at least one acquisition score.")
    scores: dict[str, float] = {}
    for name, raw in mapping.items():
        validate_record_id(name)
        scores[name] = require_finite_float(raw, f"{path}.{name}")
    return scores


@dataclass(frozen=True, kw_only=True)
class AcquisitionScoreRecord:
    """Scores for one candidate, grouped by its indivisible parent/trajectory."""

    record_id: str
    parent_id: str
    trajectory_id: str | None = None
    frame_index: int | None = None
    scores: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record_id = validate_record_id(self.record_id)
        parent_id = validate_record_id(self.parent_id)
        trajectory_id = optional_str(self.trajectory_id, "score_record.trajectory_id")
        if trajectory_id is not None:
            trajectory_id = validate_record_id(trajectory_id)
        frame_index = optional_int(self.frame_index, "score_record.frame_index")
        if frame_index is not None and frame_index < 0:
            raise ValidationError("score_record.frame_index must not be negative.")
        if frame_index is not None and trajectory_id is None:
            raise ValidationError(
                "score_record.trajectory_id is required when frame_index is present."
            )
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "scores", _validated_scores(self.scores, "score_record.scores"))
        object.__setattr__(
            self,
            "provenance",
            validated_json_object(self.provenance, "score_record.provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "parent_id": self.parent_id,
            "trajectory_id": self.trajectory_id,
            "frame_index": self.frame_index,
            "scores": dict(self.scores),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Any) -> AcquisitionScoreRecord:
        mapping = require_mapping(data, "score_record")
        reject_unknown_keys(mapping, _SCORE_RECORD_KEYS, "score_record")
        return cls(
            record_id=require_key(mapping, "record_id", "score_record"),
            parent_id=require_key(mapping, "parent_id", "score_record"),
            trajectory_id=mapping.get("trajectory_id"),
            frame_index=mapping.get("frame_index"),
            scores=require_key(mapping, "scores", "score_record"),
            provenance=mapping.get("provenance") or {},
        )


@dataclass(frozen=True, kw_only=True)
class AcquisitionScoreManifest:
    """A complete, uniquely keyed pool of acquisition scores."""

    schema: str = ACQUISITION_SCORE_SCHEMA
    score_id: str
    source: dict[str, str]
    records: tuple[AcquisitionScoreRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != ACQUISITION_SCORE_SCHEMA:
            raise ValidationError(
                f"scores.schema must be {ACQUISITION_SCORE_SCHEMA!r}; got {self.schema!r}."
            )
        records: list[AcquisitionScoreRecord] = []
        seen: set[str] = set()
        for index, record in enumerate(require_sequence(self.records, "scores.records")):
            if not isinstance(record, AcquisitionScoreRecord):
                raise ValidationError(
                    f"scores.records[{index}] must be an AcquisitionScoreRecord."
                )
            if record.record_id in seen:
                raise ValidationError(f"scores.records[{index}] repeats {record.record_id!r}.")
            seen.add(record.record_id)
            records.append(record)
        if not records:
            raise ValidationError("scores.records must contain at least one candidate.")
        object.__setattr__(self, "score_id", validate_record_id(self.score_id))
        object.__setattr__(self, "source", _validated_source(self.source, "scores.source"))
        object.__setattr__(self, "records", tuple(records))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "score_id": self.score_id,
            "source": dict(self.source),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Any) -> AcquisitionScoreManifest:
        mapping = require_mapping(data, "scores")
        reject_unknown_keys(mapping, _SCORE_MANIFEST_KEYS, "scores")
        return cls(
            schema=require_key(mapping, "schema", "scores"),
            score_id=require_key(mapping, "score_id", "scores"),
            source=require_key(mapping, "source", "scores"),
            records=tuple(
                AcquisitionScoreRecord.from_dict(item)
                for item in require_sequence(
                    require_key(mapping, "records", "scores"), "scores.records"
                )
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SelectionManifest:
    """Deterministic policy selections tied to one immutable score file."""

    schema: str = SELECTION_MANIFEST_SCHEMA
    selection_id: str
    source: dict[str, str]
    config_sha256: str
    config: dict[str, Any]
    policy_selections: dict[str, tuple[str, ...]]
    union_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != SELECTION_MANIFEST_SCHEMA:
            raise ValidationError(
                f"selection.schema must be {SELECTION_MANIFEST_SCHEMA!r}; got {self.schema!r}."
            )
        config = validated_json_object(self.config, "selection.config")
        digest = _require_sha256(self.config_sha256, "selection.config_sha256")
        if digest != canonical_json_fingerprint(config):
            raise ValidationError("selection.config_sha256 does not match selection.config.")
        raw_policies = require_mapping(self.policy_selections, "selection.policy_selections")
        if not raw_policies:
            raise ValidationError("selection.policy_selections must not be empty.")
        policies: dict[str, tuple[str, ...]] = {}
        derived_union: set[str] = set()
        for name, raw_ids in raw_policies.items():
            validate_record_id(name)
            ids = tuple(
                validate_record_id(require_str(item, f"selection.policy_selections.{name}"))
                for item in require_sequence(raw_ids, f"selection.policy_selections.{name}")
            )
            if len(ids) != len(set(ids)):
                raise ValidationError(f"selection policy {name!r} repeats a record id.")
            if not ids:
                raise ValidationError(f"selection policy {name!r} selected no records.")
            policies[name] = ids
            derived_union.update(ids)
        union_ids = tuple(
            validate_record_id(require_str(item, "selection.union_record_ids"))
            for item in require_sequence(self.union_record_ids, "selection.union_record_ids")
        )
        expected_union = tuple(sorted(derived_union))
        if union_ids != expected_union:
            raise ValidationError(
                "selection.union_record_ids must be the sorted union of all policy selections."
            )
        object.__setattr__(self, "selection_id", validate_record_id(self.selection_id))
        object.__setattr__(self, "source", _validated_source(self.source, "selection.source"))
        object.__setattr__(self, "config_sha256", digest)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "policy_selections", policies)
        object.__setattr__(self, "union_record_ids", union_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selection_id": self.selection_id,
            "source": dict(self.source),
            "config_sha256": self.config_sha256,
            "config": dict(self.config),
            "policy_selections": {
                name: list(record_ids) for name, record_ids in self.policy_selections.items()
            },
            "union_record_ids": list(self.union_record_ids),
        }

    @classmethod
    def from_dict(cls, data: Any) -> SelectionManifest:
        mapping = require_mapping(data, "selection")
        reject_unknown_keys(mapping, _SELECTION_MANIFEST_KEYS, "selection")
        raw_policies = require_mapping(
            require_key(mapping, "policy_selections", "selection"),
            "selection.policy_selections",
        )
        return cls(
            schema=require_key(mapping, "schema", "selection"),
            selection_id=require_key(mapping, "selection_id", "selection"),
            source=require_key(mapping, "source", "selection"),
            config_sha256=require_key(mapping, "config_sha256", "selection"),
            config=require_key(mapping, "config", "selection"),
            policy_selections={
                name: require_sequence(value, f"selection.policy_selections.{name}")
                for name, value in raw_policies.items()
            },
            union_record_ids=require_sequence(
                require_key(mapping, "union_record_ids", "selection"),
                "selection.union_record_ids",
            ),
        )
