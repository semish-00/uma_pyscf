#!/usr/bin/env python3
"""Measure density-fitting relative-energy and gradient errors for C3 sentinels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from common import load_result, write_json

HARTREE_TO_KCAL_MOL = 627.5094740631
IDENTITY_KEYS = (
    "atoms",
    "charge",
    "multiplicity",
    "pyscf_spin_2s",
    "electron_count",
    "functional",
    "basis",
)


def _gradient_error(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float]:
    a = left["gradient_hartree_per_bohr"]
    b = right["gradient_hartree_per_bohr"]
    if len(a) != len(b):
        raise ValueError("Gradient atom counts differ.")
    differences = [
        float(y) - float(x)
        for row_a, row_b in zip(a, b)
        for x, y in zip(row_a, row_b)
    ]
    return (
        math.sqrt(sum(value * value for value in differences) / len(differences)),
        max(abs(value) for value in differences),
    )


def _load(root: Path, case_id: str) -> dict[str, Any]:
    result = load_result(root / "runs" / case_id / "pyscf-cpu" / "result.json")
    if result.get("engine") != "pyscf-cpu" or result.get("converged") is not True:
        raise ValueError(f"{case_id} must be a converged pyscf-cpu result.")
    return result


def _require_density_pair(
    direct: dict[str, Any], density: dict[str, Any], label: str
) -> None:
    direct_case = direct.get("case")
    density_case = density.get("case")
    if not isinstance(direct_case, dict) or not isinstance(density_case, dict):
        raise ValueError(f"{label} results must contain case metadata.")
    for key in IDENTITY_KEYS:
        if direct_case.get(key) != density_case.get(key):
            raise ValueError(f"{label} density fitting changes case.{key}.")
    left = direct.get("settings")
    right = density.get("settings")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError(f"{label} results must contain settings metadata.")
    changed = [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    if changed != ["density_fit"] or left.get("density_fit") is not False or right.get("density_fit") is not True:
        raise ValueError(
            f"{label} must change only density_fit false -> true; changed={changed}."
        )


def _require_same_state(structure: dict[str, Any], reference: dict[str, Any]) -> None:
    left = structure["case"]
    right = reference["case"]
    for key in IDENTITY_KEYS[1:]:
        if left.get(key) != right.get(key):
            raise ValueError(f"Relative-energy pair differs in case.{key}.")
    left_elements = [atom.get("element") for atom in left["atoms"]]
    right_elements = [atom.get("element") for atom in right["atoms"]]
    if left_elements != right_elements:
        raise ValueError("Relative-energy pair differs in element order.")


def analyze(suite: dict[str, Any], root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for entry in suite["cases"]:
        structure_id = str(entry["structure_case_id"])
        reference_id = str(entry["reference_case_id"])
        density_id = str(entry["case_id"])
        reference_density_id = str(entry["reference_density_fit_case_id"])
        direct = _load(root, structure_id)
        reference = _load(root, reference_id)
        density = _load(root, density_id)
        reference_density = _load(root, reference_density_id)
        _require_density_pair(direct, density, structure_id)
        _require_density_pair(reference, reference_density, reference_id)
        _require_same_state(direct, reference)
        direct_relative = float(direct["energy_hartree"]) - float(reference["energy_hartree"])
        density_relative = float(density["energy_hartree"]) - float(
            reference_density["energy_hartree"]
        )
        error = density_relative - direct_relative
        gradient_rmse, gradient_max = _gradient_error(direct, density)
        rows.append(
            {
                "case_id": density_id,
                "structure_case_id": structure_id,
                "reference_case_id": reference_id,
                "direct_relative_energy_hartree": direct_relative,
                "density_fit_relative_energy_hartree": density_relative,
                "relative_energy_signed_error_hartree": error,
                "relative_energy_absolute_error_hartree": abs(error),
                "relative_energy_absolute_error_kcal_mol": (
                    abs(error) * HARTREE_TO_KCAL_MOL
                ),
                "gradient_component_rmse_hartree_per_bohr": gradient_rmse,
                "gradient_max_absolute_difference_hartree_per_bohr": gradient_max,
                "direct_wall_time_seconds": float(direct["wall_time_seconds"]),
                "density_fit_wall_time_seconds": float(density["wall_time_seconds"]),
                "speedup_vs_direct": (
                    float(direct["wall_time_seconds"])
                    / float(density["wall_time_seconds"])
                ),
            }
        )
    worst = max(rows, key=lambda row: row["relative_energy_absolute_error_hartree"])
    return {
        "schema": "c3-density-fit-relative-analysis-v1",
        "suite_id": suite["suite_id"],
        "selection_status": "diagnostic_not_production_frozen",
        "case_count": len(rows),
        "worst_relative_energy_absolute_error_hartree": worst[
            "relative_energy_absolute_error_hartree"
        ],
        "worst_relative_energy_absolute_error_kcal_mol": worst[
            "relative_energy_absolute_error_kcal_mol"
        ],
        "worst_relative_energy_case_id": worst["case_id"],
        "worst_gradient_component_rmse_hartree_per_bohr": max(
            row["gradient_component_rmse_hartree_per_bohr"] for row in rows
        ),
        "worst_gradient_max_absolute_difference_hartree_per_bohr": max(
            row["gradient_max_absolute_difference_hartree_per_bohr"] for row in rows
        ),
        "aggregate_speedup_vs_direct": sum(
            row["direct_wall_time_seconds"] for row in rows
        )
        / sum(row["density_fit_wall_time_seconds"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    report = analyze(suite, args.root.resolve())
    write_json(args.output, report)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    print(
        f"cases={report['case_count']} "
        f"worst_relative_error={report['worst_relative_energy_absolute_error_hartree']:.6e} Eh "
        f"({report['worst_relative_energy_absolute_error_kcal_mol']:.6e} kcal/mol) "
        f"speedup={report['aggregate_speedup_vs_direct']:.3f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
