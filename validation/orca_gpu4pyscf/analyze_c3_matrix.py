#!/usr/bin/env python3
"""Compare C3 GPU setting variants with their direct grid-5/VV10-5 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from common import write_json
from generate_c3_matrix import VARIANTS

IDENTITY_KEYS = (
    "atoms",
    "charge",
    "multiplicity",
    "pyscf_spin_2s",
    "electron_count",
    "functional",
    "basis",
)
PROVISIONAL_THRESHOLDS = {
    "energy_abs_hartree": 5e-6,
    "gradient_rmse_hartree_per_bohr": 2e-5,
    "gradient_max_hartree_per_bohr": 1e-4,
}


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _gradient(result: dict[str, Any]) -> list[list[float]]:
    rows = result.get("gradient_hartree_per_bohr")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Result has no gradient_hartree_per_bohr array.")
    gradient: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Every gradient row must contain three values.")
        gradient.append([float(value) for value in row])
    return gradient


def _require_same_scientific_case(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    left = baseline.get("case")
    right = candidate.get("case")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("Both results must contain case metadata.")
    for key in IDENTITY_KEYS:
        if left.get(key) != right.get(key):
            raise ValueError(f"C3 variant changes scientific case field {key}.")
    if baseline.get("engine") != "gpu4pyscf" or candidate.get("engine") != "gpu4pyscf":
        raise ValueError("C3 analysis requires two GPU4PySCF results.")
    if baseline.get("converged") is not True or candidate.get("converged") is not True:
        raise ValueError("C3 analysis requires converged baseline and candidate results.")


def _require_one_axis_change(
    baseline: dict[str, Any], candidate: dict[str, Any], key: str, expected: Any
) -> None:
    left = baseline.get("settings")
    right = candidate.get("settings")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("Both results must contain settings metadata.")
    changed = [name for name in sorted(set(left) | set(right)) if left.get(name) != right.get(name)]
    if changed != [key]:
        raise ValueError(f"Expected only settings.{key} to change; changed={changed}.")
    if right.get(key) != expected:
        raise ValueError(f"settings.{key}={right.get(key)!r}, expected {expected!r}.")


def compare_variant(
    baseline: dict[str, Any], candidate: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    key = str(row["setting_key"])
    _require_same_scientific_case(baseline, candidate)
    _require_one_axis_change(baseline, candidate, key, row["candidate_value"])
    left_gradient = _gradient(baseline)
    right_gradient = _gradient(candidate)
    if len(left_gradient) != len(right_gradient):
        raise ValueError("Baseline and candidate gradient shapes differ.")
    differences = [
        right_gradient[atom][axis] - left_gradient[atom][axis]
        for atom in range(len(left_gradient))
        for axis in range(3)
    ]
    rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
    max_abs = max(abs(value) for value in differences)
    energy_difference = float(candidate["energy_hartree"]) - float(
        baseline["energy_hartree"]
    )
    baseline_wall = float(baseline["wall_time_seconds"])
    candidate_wall = float(candidate["wall_time_seconds"])
    metrics = {
        "base_case_id": row["base_case_id"],
        "case_id": row["case_id"],
        "axis": row["axis"],
        "setting_key": key,
        "baseline_value": row["baseline_value"],
        "candidate_value": row["candidate_value"],
        "energy_signed_difference_hartree": energy_difference,
        "energy_absolute_difference_hartree": abs(energy_difference),
        "gradient_component_rmse_hartree_per_bohr": rmse,
        "gradient_max_absolute_difference_hartree_per_bohr": max_abs,
        "s2_absolute_difference": (
            abs(float(candidate["s2"]) - float(baseline["s2"]))
            if baseline.get("s2") is not None and candidate.get("s2") is not None
            else None
        ),
        "baseline_wall_time_seconds": baseline_wall,
        "candidate_wall_time_seconds": candidate_wall,
        "speedup_vs_baseline": baseline_wall / candidate_wall,
    }
    metrics["within_provisional_cpu_gpu_thresholds"] = (
        metrics["energy_absolute_difference_hartree"]
        <= PROVISIONAL_THRESHOLDS["energy_abs_hartree"]
        and rmse <= PROVISIONAL_THRESHOLDS["gradient_rmse_hartree_per_bohr"]
        and max_abs <= PROVISIONAL_THRESHOLDS["gradient_max_hartree_per_bohr"]
    )
    return metrics


def analyze(suite_path: Path, root: Path) -> dict[str, Any]:
    suite = _load_json(suite_path)
    rows: list[dict[str, Any]] = []
    for entry in suite.get("cases", []):
        if not isinstance(entry, dict):
            raise ValueError("Suite case entries must be objects.")
        base_case_id = str(entry["base_case_id"])
        case_id = str(entry["case_id"])
        baseline = _load_json(root / "runs" / base_case_id / "gpu4pyscf" / "result.json")
        candidate = _load_json(root / "runs" / case_id / "gpu4pyscf" / "result.json")
        rows.append(compare_variant(baseline, candidate, entry))

    groups: list[dict[str, Any]] = []
    for suffix, axis, key, value in VARIANTS:
        selected = [
            row
            for row in rows
            if row["axis"] == axis and row["candidate_value"] == value
        ]
        groups.append(
            {
                "variant": suffix,
                "axis": axis,
                "setting_key": key,
                "candidate_value": value,
                "case_count": len(selected),
                "all_within_provisional_cpu_gpu_thresholds": all(
                    row["within_provisional_cpu_gpu_thresholds"] for row in selected
                ),
                "worst_energy_absolute_difference_hartree": max(
                    row["energy_absolute_difference_hartree"] for row in selected
                ),
                "worst_gradient_component_rmse_hartree_per_bohr": max(
                    row["gradient_component_rmse_hartree_per_bohr"] for row in selected
                ),
                "worst_gradient_max_absolute_difference_hartree_per_bohr": max(
                    row["gradient_max_absolute_difference_hartree_per_bohr"]
                    for row in selected
                ),
                "aggregate_speedup_vs_baseline": sum(
                    row["baseline_wall_time_seconds"] for row in selected
                )
                / sum(row["candidate_wall_time_seconds"] for row in selected),
            }
        )
    return {
        "schema": "gpu-c3-settings-analysis-v1",
        "suite_id": suite.get("suite_id"),
        "selection_status": "diagnostic_not_production_frozen",
        "provisional_thresholds_for_context_only": dict(PROVISIONAL_THRESHOLDS),
        "rows": rows,
        "groups": groups,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    report = analyze(args.suite.resolve(), args.root.resolve())
    write_json(args.output, report)
    if args.csv_output:
        write_csv(args.csv_output, report["rows"])
    for group in report["groups"]:
        print(
            f"{group['variant']}: dE={group['worst_energy_absolute_difference_hartree']:.6e} "
            f"grad_rmse={group['worst_gradient_component_rmse_hartree_per_bohr']:.6e} "
            f"grad_max={group['worst_gradient_max_absolute_difference_hartree_per_bohr']:.6e} "
            f"speedup={group['aggregate_speedup_vs_baseline']:.3f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
