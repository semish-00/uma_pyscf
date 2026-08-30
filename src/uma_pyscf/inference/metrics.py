"""Deterministic energy/force metrics for UMA predictions."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt
from typing import Any

from ..core.errors import ValidationError

__all__ = ["PredictionRecord", "summarize_predictions"]

Vector = tuple[float, float, float]


@dataclass(frozen=True, kw_only=True)
class PredictionRecord:
    """One model prediction paired with its immutable ASE-LMDB reference."""

    partition: str
    record_id: str
    atomic_numbers: tuple[int, ...]
    charge: int
    multiplicity: int
    reference_energy_ev: float
    predicted_energy_ev: float
    reference_forces_ev_per_angstrom: tuple[Vector, ...]
    predicted_forces_ev_per_angstrom: tuple[Vector, ...]

    def __post_init__(self) -> None:
        natoms = len(self.atomic_numbers)
        if natoms < 1:
            raise ValidationError("A prediction record must contain at least one atom.")
        if len(self.reference_forces_ev_per_angstrom) != natoms:
            raise ValidationError("Reference force rows do not match the atom count.")
        if len(self.predicted_forces_ev_per_angstrom) != natoms:
            raise ValidationError("Predicted force rows do not match the atom count.")

    @property
    def energy_error_ev(self) -> float:
        return self.predicted_energy_ev - self.reference_energy_ev

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "record_id": self.record_id,
            "atomic_numbers": list(self.atomic_numbers),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "reference_energy_ev": self.reference_energy_ev,
            "predicted_energy_ev": self.predicted_energy_ev,
            "energy_error_ev": self.energy_error_ev,
            "reference_forces_ev_per_angstrom": [
                list(row) for row in self.reference_forces_ev_per_angstrom
            ],
            "predicted_forces_ev_per_angstrom": [
                list(row) for row in self.predicted_forces_ev_per_angstrom
            ],
        }


def summarize_predictions(records: tuple[PredictionRecord, ...]) -> dict[str, int | float]:
    """Return absolute and same-composition-relative errors for one partition."""
    if not records:
        raise ValidationError("Cannot summarize an empty prediction partition.")
    energy_errors = [record.energy_error_ev for record in records]
    force_errors = [
        predicted - reference
        for record in records
        for predicted_row, reference_row in zip(
            record.predicted_forces_ev_per_angstrom,
            record.reference_forces_ev_per_angstrom,
            strict=True,
        )
        for predicted, reference in zip(predicted_row, reference_row, strict=True)
    ]
    if not force_errors:
        raise ValidationError("Prediction records contain no force components.")
    errors_by_composition: dict[tuple[tuple[int, int], ...], list[float]] = {}
    for record in records:
        counts: dict[int, int] = {}
        for number in record.atomic_numbers:
            counts[number] = counts.get(number, 0) + 1
        composition = tuple(sorted(counts.items()))
        errors_by_composition.setdefault(composition, []).append(record.energy_error_ev)
    composition_centered_errors = [
        value - fsum(values) / len(values)
        for values in errors_by_composition.values()
        for value in values
    ]
    return {
        "records": len(records),
        "atoms": sum(len(record.atomic_numbers) for record in records),
        "compositions": len(errors_by_composition),
        "energy_mean_error_ev": fsum(energy_errors) / len(energy_errors),
        "energy_mae_ev": fsum(abs(value) for value in energy_errors) / len(energy_errors),
        "energy_rmse_ev": sqrt(
            fsum(value * value for value in energy_errors) / len(energy_errors)
        ),
        "energy_mae_ev_per_atom": fsum(
            abs(record.energy_error_ev) / len(record.atomic_numbers) for record in records
        )
        / len(records),
        "energy_same_composition_centered_mae_ev": fsum(
            abs(value) for value in composition_centered_errors
        )
        / len(composition_centered_errors),
        "energy_same_composition_centered_rmse_ev": sqrt(
            fsum(value * value for value in composition_centered_errors)
            / len(composition_centered_errors)
        ),
        "force_component_mae_ev_per_angstrom": fsum(abs(value) for value in force_errors)
        / len(force_errors),
        "force_component_rmse_ev_per_angstrom": sqrt(
            fsum(value * value for value in force_errors) / len(force_errors)
        ),
        "force_component_max_abs_error_ev_per_angstrom": max(
            abs(value) for value in force_errors
        ),
    }
