"""Fit and evaluate a leakage-safe atomic composition energy baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum, sqrt

from ..core.elements import PERIODIC_SYMBOLS
from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint
from ..schemas.composition_baseline import (
    COMPOSITION_BASELINE_METHOD,
    CompositionBaseline,
)
from ..schemas.label_record import LabelRecord
from ..schemas.split_manifest import SplitManifest
from .splits import composition_formula

__all__ = [
    "atomic_counts",
    "fit_atomic_composition_baseline",
    "predict_baseline_energy",
]


def atomic_counts(record: LabelRecord, elements: Sequence[str]) -> tuple[float, ...]:
    """Return the element-count design row for ``record`` in ``elements`` order."""
    counts: dict[str, int] = {}
    for number in record.structure.atomic_numbers:
        symbol = PERIODIC_SYMBOLS[number]
        counts[symbol] = counts.get(symbol, 0) + 1
    return tuple(float(counts.get(symbol, 0)) for symbol in elements)


def _solve_full_rank(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve one square system by partial-pivot Gaussian elimination.

    The system is the small normal equation of an integer element-count matrix.
    Rank deficiency is a scientific provenance error -- the requested atomic
    references are not identifiable from the training compositions -- so it is
    detected and reported rather than regularized into an arbitrary answer.
    """
    size = len(matrix)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    tolerance = max(1.0, scale) * 1e-12
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValidationError(
                "The training composition design is rank deficient: per-element reference "
                "energies are not identifiable. Add an independent training composition; "
                "do not fit on a holdout partition or regularize away the missing rank."
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]
    return [augmented[index][-1] for index in range(size)]


def _least_squares(rows: Sequence[Sequence[float]], targets: Sequence[float]) -> list[float]:
    if not rows:
        raise ValidationError("The fit partition contains no records.")
    width = len(rows[0])
    if width == 0 or len(rows) < width:
        raise ValidationError(
            f"The fit has {len(rows)} records for {width} element references; at least "
            "one full-rank row per element is required."
        )
    normal = [[0.0 for _ in range(width)] for _ in range(width)]
    projected = [0.0 for _ in range(width)]
    for row, target in zip(rows, targets, strict=True):
        if len(row) != width:
            raise ValidationError("Composition design rows have inconsistent widths.")
        for left in range(width):
            projected[left] += row[left] * target
            for right in range(width):
                normal[left][right] += row[left] * row[right]
    return _solve_full_rank(normal, projected)


def predict_baseline_energy(
    record: LabelRecord, references: Mapping[str, float]
) -> float:
    """Return ``sum_Z n_Z * epsilon_Z`` for one label record."""
    counts: dict[str, int] = {}
    for number in record.structure.atomic_numbers:
        symbol = PERIODIC_SYMBOLS[number]
        counts[symbol] = counts.get(symbol, 0) + 1
    unknown = sorted(set(counts) - set(references))
    if unknown:
        raise ValidationError(
            f"Record {record.record_id!r} contains element(s) absent from the baseline: "
            f"{unknown!r}."
        )
    return fsum(count * float(references[symbol]) for symbol, count in counts.items())


def _metrics(
    records: Sequence[LabelRecord], references: Mapping[str, float]
) -> dict[str, int | float]:
    residuals = [
        record.results.energy_hartree - predict_baseline_energy(record, references)
        for record in records
    ]
    if not residuals:
        raise ValidationError("A declared split partition contains no records.")
    return {
        "records": len(records),
        "compositions": len(
            {composition_formula(record.structure.atomic_numbers) for record in records}
        ),
        "mean_error_hartree": fsum(residuals) / len(residuals),
        "mae_hartree": fsum(abs(value) for value in residuals) / len(residuals),
        "rmse_hartree": sqrt(fsum(value * value for value in residuals) / len(residuals)),
        "max_abs_error_hartree": max(abs(value) for value in residuals),
    }


def fit_atomic_composition_baseline(
    records: Sequence[LabelRecord],
    split: SplitManifest,
    *,
    baseline_id: str,
    fit_partition: str,
    record_checksums_sha256: Mapping[str, str],
) -> CompositionBaseline:
    """Fit atomic references on one partition and evaluate every partition.

    Every record named by the split must be present exactly once and already
    accepted by QC.  This deliberately refuses a partial accepted subset: if a
    source candidate was rejected, the split must be regenerated for the actual
    released record set instead of silently changing its membership here.
    """
    if not isinstance(split, SplitManifest):
        raise ValidationError(f"split must be SplitManifest; got {type(split).__name__}.")
    if fit_partition not in split.record_assignments:
        raise ValidationError(
            f"Fit partition {fit_partition!r} is not declared by split {split.split_id!r}."
        )
    by_id: dict[str, LabelRecord] = {}
    for index, record in enumerate(records):
        if not isinstance(record, LabelRecord):
            raise ValidationError(
                f"records[{index}] must be a LabelRecord; got {type(record).__name__}."
            )
        if record.record_id in by_id:
            raise ValidationError(f"Record id {record.record_id!r} appears more than once.")
        if record.qc.status != "accepted":
            raise ValidationError(
                f"Record {record.record_id!r} has qc.status {record.qc.status!r}; a baseline "
                "may be fitted and evaluated only on accepted records."
            )
        by_id[record.record_id] = record

    assigned = {
        record_id
        for partition_ids in split.record_assignments.values()
        for record_id in partition_ids
    }
    if set(by_id) != assigned:
        missing = sorted(assigned - set(by_id))
        extra = sorted(set(by_id) - assigned)
        raise ValidationError(
            f"Records do not exactly match split {split.split_id!r}; missing={missing!r}, "
            f"extra={extra!r}."
        )
    if set(record_checksums_sha256) != set(by_id):
        raise ValidationError(
            "record_checksums_sha256 must name exactly the records used by the baseline."
        )

    fit_ids = split.record_assignments[fit_partition]
    fit_records = tuple(by_id[record_id] for record_id in fit_ids)
    elements = tuple(
        sorted(
            {
                PERIODIC_SYMBOLS[number]
                for record in fit_records
                for number in record.structure.atomic_numbers
            }
        )
    )
    all_elements = {
        PERIODIC_SYMBOLS[number]
        for record in records
        for number in record.structure.atomic_numbers
    }
    if set(elements) != all_elements:
        absent = sorted(all_elements - set(elements))
        raise ValidationError(
            f"Fit partition {fit_partition!r} does not contain every dataset element; "
            f"missing {absent!r}. Atomic references cannot be extrapolated to them."
        )
    rows = [atomic_counts(record, elements) for record in fit_records]
    targets = [record.results.energy_hartree for record in fit_records]
    solution = _least_squares(rows, targets)
    references = dict(zip(elements, solution, strict=True))
    metrics = {
        partition: _metrics(tuple(by_id[record_id] for record_id in ids), references)
        for partition, ids in split.record_assignments.items()
    }
    return CompositionBaseline(
        baseline_id=baseline_id,
        method=COMPOSITION_BASELINE_METHOD,
        energy_unit="hartree",
        split={
            "id": split.split_id,
            "sha256": canonical_json_fingerprint(split.to_dict()),
            "fit_partition": fit_partition,
        },
        elements=elements,
        design_rank=len(elements),
        atomic_reference_energy_hartree=references,
        record_checksums_sha256=dict(record_checksums_sha256),
        fit_record_ids=tuple(sorted(fit_ids)),
        metrics_by_partition=metrics,
    )
