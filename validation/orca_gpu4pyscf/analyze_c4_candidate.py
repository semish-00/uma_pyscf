#!/usr/bin/env python3
"""Compare the C4 GPU density-fitting candidate with direct CPU/ORCA results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from common import load_result, write_json

IDENTITY_KEYS = (
    "atoms",
    "charge",
    "multiplicity",
    "pyscf_spin_2s",
    "electron_count",
    "functional",
    "basis",
)
PAIR_SPECS = (
    ("gpu4pyscf-density-fit", "pyscf-cpu-direct"),
    ("gpu4pyscf-density-fit", "orca-direct"),
)


def _require_same_case(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_case = left.get("case")
    right_case = right.get("case")
    if not isinstance(left_case, dict) or not isinstance(right_case, dict):
        raise ValueError("Results must contain case metadata.")
    for key in IDENTITY_KEYS:
        if left_case.get(key) != right_case.get(key):
            raise ValueError(f"C4 comparison differs in case.{key}.")
    if left.get("converged") is not True or right.get("converged") is not True:
        raise ValueError("C4 comparison requires converged results.")


def _pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    _require_same_case(left, right)
    left_gradient = left["gradient_hartree_per_bohr"]
    right_gradient = right["gradient_hartree_per_bohr"]
    if len(left_gradient) != len(right_gradient):
        raise ValueError("Gradient atom counts differ.")
    differences = [
        float(a) - float(b)
        for left_row, right_row in zip(left_gradient, right_gradient)
        for a, b in zip(left_row, right_row)
    ]
    energy_difference = float(left["energy_hartree"]) - float(right["energy_hartree"])
    if left.get("tolerances") != right.get("tolerances") or not isinstance(
        left.get("tolerances"), dict
    ):
        raise ValueError("C4 comparison requires identical tolerance mappings.")
    tolerances = left["tolerances"]
    gradient_rmse = math.sqrt(
        sum(value * value for value in differences) / len(differences)
    )
    gradient_max = max(abs(value) for value in differences)
    return {
        "energy_signed_difference_hartree": energy_difference,
        "energy_absolute_difference_hartree": abs(energy_difference),
        "gradient_component_rmse_hartree_per_bohr": gradient_rmse,
        "gradient_max_absolute_difference_hartree_per_bohr": gradient_max,
        "s2_absolute_difference": (
            abs(float(left["s2"]) - float(right["s2"]))
            if left.get("s2") is not None and right.get("s2") is not None
            else None
        ),
        "within_manifest_energy_tolerance": abs(energy_difference)
        <= float(tolerances["energy_abs_hartree"]),
        "within_manifest_gradient_rmse_tolerance": gradient_rmse
        <= float(tolerances["gradient_rms_hartree_per_bohr"]),
        "within_manifest_gradient_max_tolerance": gradient_max
        <= float(tolerances["gradient_max_hartree_per_bohr"]),
    }


def _require_candidate_change(candidate: dict[str, Any], cpu: dict[str, Any]) -> None:
    left = cpu.get("settings")
    right = candidate.get("settings")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("C4 CPU and GPU results must contain settings metadata.")
    changed = [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    allowed_changes = {
        ("density_fit",),
        (
            "density_fit",
            "init_guess",
            "initial_density_generated_before_device_conversion",
        ),
    }
    if tuple(changed) not in allowed_changes or left.get("density_fit") is not False or right.get("density_fit") is not True:
        raise ValueError(
            "C4 candidate must change density_fit false -> true, optionally "
            "with an explicit shared initial density; "
            f"changed={changed}."
        )


def _load(root: Path, case_id: str, engine: str) -> dict[str, Any]:
    return load_result(root / "runs" / case_id / engine / "result.json")


def analyze(suite: dict[str, Any], root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for entry in suite["cases"]:
        candidate_id = str(entry["case_id"])
        base_id = str(entry["base_case_id"])
        candidate = _load(root, candidate_id, "gpu4pyscf")
        cpu = _load(root, base_id, "pyscf-cpu")
        orca = _load(root, base_id, "orca")
        _require_candidate_change(candidate, cpu)
        for right_name, right in (("pyscf-cpu-direct", cpu), ("orca-direct", orca)):
            rows.append(
                {
                    "case_id": candidate_id,
                    "base_case_id": base_id,
                    "category": entry.get("category", ""),
                    "left_engine": "gpu4pyscf-density-fit",
                    "right_engine": right_name,
                    **_pair_metrics(candidate, right),
                }
            )
        performance_rows.append(
            {
                "case_id": candidate_id,
                "base_case_id": base_id,
                "gpu_wall_time_seconds": float(candidate["wall_time_seconds"]),
                "cpu_direct_wall_time_seconds": float(cpu["wall_time_seconds"]),
                "speedup_vs_cpu_direct": (
                    float(cpu["wall_time_seconds"])
                    / float(candidate["wall_time_seconds"])
                ),
            }
        )

    pairs: list[dict[str, Any]] = []
    for left_engine, right_engine in PAIR_SPECS:
        selected = [
            row
            for row in rows
            if row["left_engine"] == left_engine and row["right_engine"] == right_engine
        ]
        worst_energy = max(selected, key=lambda row: row["energy_absolute_difference_hartree"])
        worst_rmse = max(
            selected, key=lambda row: row["gradient_component_rmse_hartree_per_bohr"]
        )
        worst_max = max(
            selected,
            key=lambda row: row["gradient_max_absolute_difference_hartree_per_bohr"],
        )
        pairs.append(
            {
                "left_engine": left_engine,
                "right_engine": right_engine,
                "case_count": len(selected),
                "within_manifest_energy_tolerance_count": sum(
                    row["within_manifest_energy_tolerance"] for row in selected
                ),
                "within_manifest_gradient_rmse_tolerance_count": sum(
                    row["within_manifest_gradient_rmse_tolerance"] for row in selected
                ),
                "within_manifest_gradient_max_tolerance_count": sum(
                    row["within_manifest_gradient_max_tolerance"] for row in selected
                ),
                "worst_energy_absolute_difference_hartree": {
                    "value": worst_energy["energy_absolute_difference_hartree"],
                    "case_id": worst_energy["case_id"],
                },
                "worst_gradient_component_rmse_hartree_per_bohr": {
                    "value": worst_rmse["gradient_component_rmse_hartree_per_bohr"],
                    "case_id": worst_rmse["case_id"],
                },
                "worst_gradient_max_absolute_difference_hartree_per_bohr": {
                    "value": worst_max[
                        "gradient_max_absolute_difference_hartree_per_bohr"
                    ],
                    "case_id": worst_max["case_id"],
                },
            }
        )
    return {
        "schema": "gpu-c4-density-fit-analysis-v1",
        "suite_id": suite["suite_id"],
        "selection_status": "conditional_candidate_not_production_frozen",
        "case_count": len(performance_rows),
        "aggregate_speedup_vs_cpu_direct": sum(
            row["cpu_direct_wall_time_seconds"] for row in performance_rows
        )
        / sum(row["gpu_wall_time_seconds"] for row in performance_rows),
        "pairs": pairs,
        "rows": rows,
        "performance_rows": performance_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--performance-csv-output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    report = analyze(suite, args.root.resolve())
    write_json(args.output, report)
    _write_csv(args.csv_output, report["rows"])
    _write_csv(args.performance_csv_output, report["performance_rows"])
    print(
        f"cases={report['case_count']} "
        f"speedup_vs_cpu_direct={report['aggregate_speedup_vs_cpu_direct']:.3f}x"
    )
    for pair in report["pairs"]:
        print(
            f"{pair['left_engine']} - {pair['right_engine']}: "
            f"worst_dE={pair['worst_energy_absolute_difference_hartree']['value']:.6e} "
            f"worst_grad_rmse={pair['worst_gradient_component_rmse_hartree_per_bohr']['value']:.6e} "
            f"worst_grad_max={pair['worst_gradient_max_absolute_difference_hartree_per_bohr']['value']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
