#!/usr/bin/env python
"""Compare PFP single points with canonical GPU4PySCF label records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from math import fsum, sqrt
from pathlib import Path
import sys
from typing import Any

HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.422067476325886
POSITION_TOLERANCE_ANGSTROM = 1.0e-12


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path} contains no JSON records")
    return files


def _load_by_id(path: Path, schema: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for source in _record_files(path):
        record = _read_json(source)
        if record.get("schema") != schema:
            raise ValueError(f"{source} has schema {record.get('schema')!r}, expected {schema!r}")
        record_id = str(record["record_id"])
        if record_id in records:
            raise ValueError(f"duplicate record_id {record_id!r}")
        records[record_id] = record
    return records


def _partition_metrics(
    pfp_records: dict[str, dict[str, Any]],
    reference_records: dict[str, dict[str, Any]],
    split: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if split.get("schema") != "uma-pyscf-split-manifest-v1":
        raise ValueError("--split is not an uma-pyscf-split-manifest-v1 record")
    assignments = split.get("record_assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("--split has no record_assignments")
    expected = set(pfp_records) & set(reference_records)
    assigned: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}
    for partition, raw_ids in assignments.items():
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"split partition {partition!r} has no record ids")
        record_ids = [str(value) for value in raw_ids]
        missing = set(record_ids) - expected
        if missing:
            raise ValueError(
                f"split partition {partition!r} names missing records: {sorted(missing)}"
            )
        assigned.extend(record_ids)
        _, partition_summary = compare_records(
            {record_id: pfp_records[record_id] for record_id in record_ids},
            {record_id: reference_records[record_id] for record_id in record_ids},
        )
        metrics[str(partition)] = partition_summary
    if len(assigned) != len(set(assigned)):
        raise ValueError("--split assigns at least one record to multiple partitions")
    if set(assigned) != expected:
        raise ValueError(
            "--split record assignments do not exactly cover the compared record ids"
        )
    return metrics


def _composition(numbers: list[int]) -> str:
    return ";".join(f"{number}:{count}" for number, count in sorted(Counter(numbers).items()))


def compare_records(
    pfp_records: dict[str, dict[str, Any]],
    reference_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common = sorted(set(pfp_records) & set(reference_records))
    if not common:
        raise ValueError("PFP and reference inputs have no record_id in common")
    rows: list[dict[str, Any]] = []
    force_errors: list[float] = []
    errors_by_composition: dict[str, list[float]] = defaultdict(list)
    maximum_position_delta = 0.0
    for record_id in common:
        pfp = pfp_records[record_id]
        reference = reference_records[record_id]
        pfp_input = pfp["input"]
        ref_structure = reference["structure"]
        if pfp_input["atomic_numbers"] != ref_structure["atomic_numbers"]:
            raise ValueError(f"atomic numbers differ for {record_id}")
        pfp_positions = pfp_input["positions_angstrom"]
        reference_positions = ref_structure["positions_angstrom"]
        if len(pfp_positions) != len(reference_positions):
            raise ValueError(f"position row count differs for {record_id}")
        position_delta = max(
            abs(float(pfp_value) - float(reference_value))
            for pfp_row, reference_row in zip(
                pfp_positions, reference_positions, strict=True
            )
            for pfp_value, reference_value in zip(pfp_row, reference_row, strict=True)
        )
        maximum_position_delta = max(maximum_position_delta, position_delta)
        if position_delta > POSITION_TOLERANCE_ANGSTROM:
            raise ValueError(
                f"positions differ for {record_id}: max delta={position_delta:.3e} angstrom"
            )
        if int(pfp_input["charge"]) != int(reference["state"]["charge"]):
            raise ValueError(f"charge differs for {record_id}")
        if int(pfp_input["multiplicity"]) != int(reference["state"]["multiplicity"]):
            raise ValueError(f"multiplicity differs for {record_id}")
        pfp_energy = float(pfp["results"]["energy_ev"])
        reference_energy = float(reference["results"]["energy_hartree"]) * HARTREE_TO_EV
        energy_error = pfp_energy - reference_energy
        composition = _composition(pfp_input["atomic_numbers"])
        errors_by_composition[composition].append(energy_error)
        pfp_forces = pfp["results"]["forces_ev_per_angstrom"]
        reference_gradients = reference["results"]["gradient_hartree_per_bohr"]
        if len(pfp_forces) != len(reference_gradients):
            raise ValueError(f"force row count differs for {record_id}")
        record_force_errors: list[float] = []
        for predicted_row, gradient_row in zip(pfp_forces, reference_gradients, strict=True):
            for predicted, gradient in zip(predicted_row, gradient_row, strict=True):
                reference_force = -float(gradient) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
                error = float(predicted) - reference_force
                force_errors.append(error)
                record_force_errors.append(error)
        rows.append(
            {
                "record_id": record_id,
                "composition": composition,
                "atoms": len(pfp_input["atomic_numbers"]),
                "pfp_energy_ev": pfp_energy,
                "reference_energy_ev": reference_energy,
                "energy_error_ev": energy_error,
                "force_component_mae_ev_per_angstrom": fsum(abs(v) for v in record_force_errors)
                / len(record_force_errors),
                "force_component_max_abs_error_ev_per_angstrom": max(
                    abs(v) for v in record_force_errors
                ),
            }
        )
    means = {key: fsum(values) / len(values) for key, values in errors_by_composition.items()}
    centered = []
    for row in rows:
        value = float(row["energy_error_ev"]) - means[str(row["composition"])]
        row["energy_same_composition_centered_error_ev"] = value
        centered.append(value)
    summary = {
        "schema": "uma-pyscf-pfp-comparison-v1",
        "records": len(rows),
        "atoms": sum(int(row["atoms"]) for row in rows),
        "compositions": len(errors_by_composition),
        "geometry_max_abs_delta_angstrom": maximum_position_delta,
        "energy_same_composition_centered_mae_ev": fsum(abs(value) for value in centered)
        / len(centered),
        "energy_same_composition_centered_rmse_ev": sqrt(
            fsum(value * value for value in centered) / len(centered)
        ),
        "force_component_mae_ev_per_angstrom": fsum(abs(value) for value in force_errors)
        / len(force_errors),
        "force_component_rmse_ev_per_angstrom": sqrt(
            fsum(value * value for value in force_errors) / len(force_errors)
        ),
        "force_component_max_abs_error_ev_per_angstrom": max(abs(value) for value in force_errors),
        "composition_energy_offsets_ev": means,
        "pfp_only_record_ids": sorted(set(pfp_records) - set(reference_records)),
        "reference_only_record_ids": sorted(set(reference_records) - set(pfp_records)),
    }
    return rows, summary


def _write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfp-records", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    args = parser.parse_args()
    try:
        pfp = _load_by_id(args.pfp_records, "uma-pyscf-pfp-single-point-v1")
        reference = _load_by_id(args.reference_records, "uma-pyscf-label-record-v1")
        rows, summary = compare_records(pfp, reference)
        if args.split is not None:
            split = _read_json(args.split)
            summary["split"] = {
                "split_id": split.get("split_id"),
                "sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
            }
            summary["metrics_by_partition"] = _partition_metrics(pfp, reference, split)
        _write_outputs(rows, summary, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
