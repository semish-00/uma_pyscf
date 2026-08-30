"""A train-only atomic composition energy baseline.

Schema ``uma-pyscf-composition-baseline-v1`` records the fitted per-element
reference energies and enough provenance to prove which split and which label
files produced them.  The baseline is deliberately fitted to one named split
partition only; metrics for the other partitions are evaluation results, not
additional fitting data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.elements import atomic_number
from ..core.errors import ValidationError
from ..core.ids import validate_record_id
from ._fields import (
    reject_unknown_keys,
    require_finite_float,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
)

__all__ = [
    "COMPOSITION_BASELINE_METHOD",
    "COMPOSITION_BASELINE_SCHEMA",
    "CompositionBaseline",
]

COMPOSITION_BASELINE_SCHEMA = "uma-pyscf-composition-baseline-v1"
COMPOSITION_BASELINE_METHOD = "per_element_least_squares_v1"

_BASELINE_KEYS = (
    "schema",
    "baseline_id",
    "method",
    "energy_unit",
    "split",
    "elements",
    "design_rank",
    "atomic_reference_energy_hartree",
    "record_checksums_sha256",
    "fit_record_ids",
    "metrics_by_partition",
)
_SPLIT_KEYS = ("id", "sha256", "fit_partition")
_METRIC_KEYS = (
    "records",
    "compositions",
    "mean_error_hartree",
    "mae_hartree",
    "rmse_hartree",
    "max_abs_error_hartree",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, path: str) -> str:
    digest = require_str(value, path).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise ValidationError(f"{path} must be 64 hexadecimal characters; got {value!r}.")
    return digest


def _validated_split(value: object) -> dict[str, str]:
    split = require_mapping(value, "baseline.split")
    reject_unknown_keys(split, _SPLIT_KEYS, "baseline.split")
    for key in _SPLIT_KEYS:
        require_key(split, key, "baseline.split")
    return {
        "id": validate_record_id(require_str(split["id"], "baseline.split.id")),
        "sha256": _require_sha256(split["sha256"], "baseline.split.sha256"),
        "fit_partition": validate_record_id(
            require_str(split["fit_partition"], "baseline.split.fit_partition")
        ),
    }


def _validated_metrics(value: object) -> dict[str, dict[str, int | float]]:
    raw = require_mapping(value, "baseline.metrics_by_partition")
    if not raw:
        raise ValidationError("baseline.metrics_by_partition must name at least one partition.")
    checked: dict[str, dict[str, int | float]] = {}
    for partition, metric_value in raw.items():
        validate_record_id(partition)
        path = f"baseline.metrics_by_partition.{partition}"
        metrics = require_mapping(metric_value, path)
        reject_unknown_keys(metrics, _METRIC_KEYS, path)
        for key in _METRIC_KEYS:
            require_key(metrics, key, path)
        records = require_int(metrics["records"], f"{path}.records")
        compositions = require_int(metrics["compositions"], f"{path}.compositions")
        if records < 1 or compositions < 1 or compositions > records:
            raise ValidationError(
                f"{path} must have records >= compositions >= 1; got "
                f"records={records}, compositions={compositions}."
            )
        values = {
            key: require_finite_float(metrics[key], f"{path}.{key}")
            for key in _METRIC_KEYS[2:]
        }
        for key in ("mae_hartree", "rmse_hartree", "max_abs_error_hartree"):
            if values[key] < 0.0:
                raise ValidationError(f"{path}.{key} must not be negative.")
        checked[partition] = {"records": records, "compositions": compositions, **values}
    return dict(sorted(checked.items()))


@dataclass(frozen=True, kw_only=True)
class CompositionBaseline:
    """Per-element reference energies fitted exclusively to one split partition."""

    schema: str = COMPOSITION_BASELINE_SCHEMA
    baseline_id: str
    method: str
    energy_unit: str
    split: dict[str, str]
    elements: tuple[str, ...]
    design_rank: int
    atomic_reference_energy_hartree: dict[str, float]
    record_checksums_sha256: dict[str, str]
    fit_record_ids: tuple[str, ...]
    metrics_by_partition: dict[str, dict[str, int | float]]

    def __post_init__(self) -> None:
        if self.schema != COMPOSITION_BASELINE_SCHEMA:
            raise ValidationError(
                f"baseline.schema must be {COMPOSITION_BASELINE_SCHEMA!r}; "
                f"got {self.schema!r}."
            )
        if self.method != COMPOSITION_BASELINE_METHOD:
            raise ValidationError(
                f"baseline.method must be {COMPOSITION_BASELINE_METHOD!r}; got {self.method!r}."
            )
        if self.energy_unit != "hartree":
            raise ValidationError(
                f"baseline.energy_unit must be 'hartree'; got {self.energy_unit!r}."
            )
        elements = tuple(
            require_str(value, f"baseline.elements[{index}]")
            for index, value in enumerate(require_sequence(self.elements, "baseline.elements"))
        )
        if (
            not elements
            or tuple(sorted(elements)) != elements
            or len(set(elements)) != len(elements)
        ):
            raise ValidationError(
                "baseline.elements must be a non-empty, alphabetically sorted list without "
                f"duplicates; got {elements!r}."
            )
        for symbol in elements:
            atomic_number(symbol)

        references_raw = require_mapping(
            self.atomic_reference_energy_hartree,
            "baseline.atomic_reference_energy_hartree",
        )
        if set(references_raw) != set(elements):
            raise ValidationError(
                "baseline.atomic_reference_energy_hartree keys must exactly match "
                f"baseline.elements; got {sorted(references_raw)!r} and {list(elements)!r}."
            )
        references = {
            symbol: require_finite_float(
                references_raw[symbol],
                f"baseline.atomic_reference_energy_hartree.{symbol}",
            )
            for symbol in elements
        }

        checksums_raw = require_mapping(
            self.record_checksums_sha256, "baseline.record_checksums_sha256"
        )
        if not checksums_raw:
            raise ValidationError("baseline.record_checksums_sha256 must not be empty.")
        checksums: dict[str, str] = {}
        for record_id, digest in checksums_raw.items():
            checked_id = validate_record_id(record_id)
            checksums[checked_id] = _require_sha256(
                digest, f"baseline.record_checksums_sha256.{record_id}"
            )

        fit_ids = tuple(
            validate_record_id(require_str(value, f"baseline.fit_record_ids[{index}]"))
            for index, value in enumerate(
                require_sequence(self.fit_record_ids, "baseline.fit_record_ids")
            )
        )
        if not fit_ids or tuple(sorted(fit_ids)) != fit_ids or len(set(fit_ids)) != len(fit_ids):
            raise ValidationError(
                "baseline.fit_record_ids must be a non-empty, sorted list without duplicates."
            )
        missing = sorted(set(fit_ids) - set(checksums))
        if missing:
            raise ValidationError(
                f"baseline.fit_record_ids names records without checksums: {missing!r}."
            )

        split = _validated_split(self.split)
        metrics = _validated_metrics(self.metrics_by_partition)
        fit_partition = split["fit_partition"]
        if fit_partition not in metrics:
            raise ValidationError(
                f"baseline.metrics_by_partition does not include fit partition "
                f"{fit_partition!r}."
            )
        if metrics[fit_partition]["records"] != len(fit_ids):
            raise ValidationError(
                f"baseline fit partition reports {metrics[fit_partition]['records']} records "
                f"but baseline.fit_record_ids lists {len(fit_ids)}."
            )

        rank = require_int(self.design_rank, "baseline.design_rank")
        if rank != len(elements):
            raise ValidationError(
                f"baseline.design_rank must equal the number of elements for an identifiable "
                f"fit; got rank={rank}, elements={len(elements)}."
            )
        object.__setattr__(self, "baseline_id", validate_record_id(self.baseline_id))
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "design_rank", rank)
        object.__setattr__(self, "atomic_reference_energy_hartree", references)
        object.__setattr__(self, "record_checksums_sha256", dict(sorted(checksums.items())))
        object.__setattr__(self, "fit_record_ids", fit_ids)
        object.__setattr__(self, "metrics_by_partition", metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "baseline_id": self.baseline_id,
            "method": self.method,
            "energy_unit": self.energy_unit,
            "split": dict(self.split),
            "elements": list(self.elements),
            "design_rank": self.design_rank,
            "atomic_reference_energy_hartree": dict(
                self.atomic_reference_energy_hartree
            ),
            "record_checksums_sha256": dict(self.record_checksums_sha256),
            "fit_record_ids": list(self.fit_record_ids),
            "metrics_by_partition": {
                name: dict(metrics) for name, metrics in self.metrics_by_partition.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> CompositionBaseline:
        mapping = require_mapping(data, "baseline")
        schema = mapping.get("schema")
        if schema != COMPOSITION_BASELINE_SCHEMA:
            raise ValidationError(
                f"baseline.schema must be {COMPOSITION_BASELINE_SCHEMA!r}; got {schema!r}."
            )
        reject_unknown_keys(mapping, _BASELINE_KEYS, "baseline")
        return cls(
            schema=schema,
            baseline_id=require_key(mapping, "baseline_id", "baseline"),
            method=require_key(mapping, "method", "baseline"),
            energy_unit=require_key(mapping, "energy_unit", "baseline"),
            split=require_key(mapping, "split", "baseline"),
            elements=tuple(
                require_sequence(require_key(mapping, "elements", "baseline"), "baseline.elements")
            ),
            design_rank=require_key(mapping, "design_rank", "baseline"),
            atomic_reference_energy_hartree=require_key(
                mapping, "atomic_reference_energy_hartree", "baseline"
            ),
            record_checksums_sha256=require_key(
                mapping, "record_checksums_sha256", "baseline"
            ),
            fit_record_ids=tuple(
                require_sequence(
                    require_key(mapping, "fit_record_ids", "baseline"),
                    "baseline.fit_record_ids",
                )
            ),
            metrics_by_partition=require_key(mapping, "metrics_by_partition", "baseline"),
        )
