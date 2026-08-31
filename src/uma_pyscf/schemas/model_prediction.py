"""Unlabeled model predictions on a versioned candidate manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ._fields import (
    reject_unknown_keys,
    require_finite_float,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    require_vector_rows,
    validated_json_object,
)
from .label_record import ElectronicState, Structure

__all__ = [
    "MODEL_PREDICTION_SCHEMA",
    "PREDICTION_UNITS",
    "ModelPredictionManifest",
    "ModelPredictionRecord",
]

MODEL_PREDICTION_SCHEMA = "uma-pyscf-model-predictions-v1"
PREDICTION_UNITS: Mapping[str, str] = MappingProxyType(
    {"energy": "eV", "forces": "eV/angstrom", "positions": "angstrom"}
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_RECORD_KEYS = ("record_id", "structure", "state", "results")
_RESULT_KEYS = ("energy_ev", "forces_ev_per_angstrom")
_MANIFEST_KEYS = (
    "schema",
    "prediction_id",
    "source",
    "model",
    "runtime",
    "units",
    "records",
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
    return {
        "id": source_id,
        "sha256": _require_sha256(require_key(mapping, "sha256", path), f"{path}.sha256"),
    }


@dataclass(frozen=True, kw_only=True)
class ModelPredictionRecord:
    """One energy/force prediction retaining candidate geometry and state."""

    record_id: str
    structure: Structure
    state: ElectronicState
    energy_ev: float
    forces_ev_per_angstrom: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.structure, Structure):
            raise ValidationError("prediction.structure must be a Structure.")
        if not isinstance(self.state, ElectronicState):
            raise ValidationError("prediction.state must be an ElectronicState.")
        forces = require_vector_rows(
            self.forces_ev_per_angstrom, "prediction.results.forces_ev_per_angstrom"
        )
        if len(forces) != self.structure.atom_count:
            raise ValidationError(
                "prediction force rows must match prediction.structure atom count."
            )
        object.__setattr__(self, "record_id", validate_record_id(self.record_id))
        object.__setattr__(
            self,
            "energy_ev",
            require_finite_float(self.energy_ev, "prediction.results.energy_ev"),
        )
        object.__setattr__(self, "forces_ev_per_angstrom", forces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "structure": self.structure.to_dict(),
            "state": self.state.to_dict(),
            "results": {
                "energy_ev": self.energy_ev,
                "forces_ev_per_angstrom": [list(row) for row in self.forces_ev_per_angstrom],
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> ModelPredictionRecord:
        mapping = require_mapping(data, "prediction")
        reject_unknown_keys(mapping, _RECORD_KEYS, "prediction")
        results = require_mapping(
            require_key(mapping, "results", "prediction"), "prediction.results"
        )
        reject_unknown_keys(results, _RESULT_KEYS, "prediction.results")
        return cls(
            record_id=require_key(mapping, "record_id", "prediction"),
            structure=Structure.from_dict(require_key(mapping, "structure", "prediction")),
            state=ElectronicState.from_dict(require_key(mapping, "state", "prediction")),
            energy_ev=require_key(results, "energy_ev", "prediction.results"),
            forces_ev_per_angstrom=require_key(
                results, "forces_ev_per_angstrom", "prediction.results"
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ModelPredictionManifest:
    """Predictions from one pinned model over one immutable candidate pool."""

    schema: str = MODEL_PREDICTION_SCHEMA
    prediction_id: str
    source: dict[str, str]
    model: dict[str, Any]
    runtime: dict[str, Any]
    units: dict[str, str] = field(default_factory=lambda: dict(PREDICTION_UNITS))
    records: tuple[ModelPredictionRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != MODEL_PREDICTION_SCHEMA:
            raise ValidationError(
                f"predictions.schema must be {MODEL_PREDICTION_SCHEMA!r}; got {self.schema!r}."
            )
        units = require_mapping(self.units, "predictions.units")
        if units != dict(PREDICTION_UNITS):
            raise ValidationError(
                f"predictions.units must be {dict(PREDICTION_UNITS)!r}; got {units!r}."
            )
        records: list[ModelPredictionRecord] = []
        seen: set[str] = set()
        for index, record in enumerate(require_sequence(self.records, "predictions.records")):
            if not isinstance(record, ModelPredictionRecord):
                raise ValidationError(
                    f"predictions.records[{index}] must be a ModelPredictionRecord."
                )
            if record.record_id in seen:
                raise ValidationError(
                    f"predictions.records[{index}] repeats {record.record_id!r}."
                )
            seen.add(record.record_id)
            records.append(record)
        if not records:
            raise ValidationError("predictions.records must not be empty.")
        object.__setattr__(self, "prediction_id", validate_record_id(self.prediction_id))
        object.__setattr__(self, "source", _validated_source(self.source, "predictions.source"))
        object.__setattr__(self, "model", validated_json_object(self.model, "predictions.model"))
        object.__setattr__(
            self, "runtime", validated_json_object(self.runtime, "predictions.runtime")
        )
        object.__setattr__(self, "units", dict(PREDICTION_UNITS))
        object.__setattr__(self, "records", tuple(records))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prediction_id": self.prediction_id,
            "source": dict(self.source),
            "model": dict(self.model),
            "runtime": dict(self.runtime),
            "units": dict(self.units),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ModelPredictionManifest:
        mapping = require_mapping(data, "predictions")
        reject_unknown_keys(mapping, _MANIFEST_KEYS, "predictions")
        return cls(
            schema=require_key(mapping, "schema", "predictions"),
            prediction_id=require_key(mapping, "prediction_id", "predictions"),
            source=require_key(mapping, "source", "predictions"),
            model=require_key(mapping, "model", "predictions"),
            runtime=require_key(mapping, "runtime", "predictions"),
            units=require_key(mapping, "units", "predictions"),
            records=tuple(
                ModelPredictionRecord.from_dict(item)
                for item in require_sequence(
                    require_key(mapping, "records", "predictions"),
                    "predictions.records",
                )
            ),
        )
