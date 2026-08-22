#!/usr/bin/env python3
"""Compare two normalized cross-code energy-and-gradient results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from common import load_result, write_json


def _gradient(data: dict[str, Any]) -> list[list[float]]:
    raw = data.get("gradient_hartree_per_bohr")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Result has no non-empty gradient_hartree_per_bohr array.")
    gradient: list[list[float]] = []
    for atom_index, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"Gradient row {atom_index} must contain three components.")
        gradient.append([float(value) for value in row])
    return gradient


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_case = left.get("case")
    right_case = right.get("case")
    if not isinstance(left_case, dict) or not isinstance(right_case, dict):
        raise ValueError("Both results must contain case metadata.")
    for key in (
        "case_id",
        "input_fingerprint_sha256",
        "charge",
        "multiplicity",
        "functional",
        "basis",
    ):
        if left_case.get(key) != right_case.get(key):
            raise ValueError(
                f"Cannot compare results with different case.{key}: "
                f"{left_case.get(key)!r} != {right_case.get(key)!r}."
            )
    if left.get("converged") is not True or right.get("converged") is not True:
        raise ValueError("Both results must explicitly report converged=true.")

    left_gradient = _gradient(left)
    right_gradient = _gradient(right)
    if len(left_gradient) != len(right_gradient):
        raise ValueError("Gradient atom counts differ.")

    component_differences: list[tuple[float, int, int, float]] = []
    squared_sum = 0.0
    component_count = 0
    for atom_index, (left_row, right_row) in enumerate(
        zip(left_gradient, right_gradient)
    ):
        if len(left_row) != len(right_row):
            raise ValueError(f"Gradient component counts differ for atom {atom_index}.")
        for axis, (left_value, right_value) in enumerate(zip(left_row, right_row)):
            difference = left_value - right_value
            component_differences.append((abs(difference), atom_index, axis, difference))
            squared_sum += difference * difference
            component_count += 1
    max_abs, max_atom, max_axis, signed_at_max = max(component_differences)
    gradient_rms = math.sqrt(squared_sum / component_count)

    energy_difference = float(left["energy_hartree"]) - float(right["energy_hartree"])
    left_tolerances = left.get("tolerances")
    right_tolerances = right.get("tolerances")
    if left_tolerances != right_tolerances or not isinstance(left_tolerances, dict):
        raise ValueError("Results must carry identical tolerance mappings.")
    tolerances = {key: float(value) for key, value in left_tolerances.items()}
    checks = {
        "energy_abs_hartree": abs(energy_difference)
        <= tolerances["energy_abs_hartree"],
        "gradient_rms_hartree_per_bohr": gradient_rms
        <= tolerances["gradient_rms_hartree_per_bohr"],
        "gradient_max_hartree_per_bohr": max_abs
        <= tolerances["gradient_max_hartree_per_bohr"],
    }
    return {
        "schema": "crosscode-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": left_case["case_id"],
        "input_fingerprint_sha256": left_case["input_fingerprint_sha256"],
        "left_engine": left["engine"],
        "right_engine": right["engine"],
        "energy_difference_hartree_left_minus_right": energy_difference,
        "energy_absolute_difference_hartree": abs(energy_difference),
        "gradient_rms_difference_hartree_per_bohr": gradient_rms,
        "gradient_max_absolute_difference_hartree_per_bohr": max_abs,
        "gradient_max_location": {
            "atom_index_zero_based": max_atom,
            "axis": ("x", "y", "z")[max_axis],
            "signed_left_minus_right": signed_at_max,
        },
        "tolerances": tolerances,
        "tolerance_status": left.get("tolerance_status"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First normalized result JSON.")
    parser.add_argument("right", help="Second normalized result JSON.")
    parser.add_argument("--output", required=True, help="Comparison JSON path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = compare_results(load_result(args.left), load_result(args.right))
    write_json(args.output, report)
    print(
        f"{report['left_engine']} vs {report['right_engine']}: "
        f"{'PASS' if report['passed'] else 'FAIL'}"
    )
    print(f"Wrote {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
