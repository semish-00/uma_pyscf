#!/usr/bin/env python3
"""Aggregate the Gate 1 metric table for a suite across the three engine pairs.

Implements the metric set and the provisional CPU-GPU numeric gate of
`docs/plans/01_gpu4pyscf_validation_plan.md` section 7. This is a report, not a
gate enforcement tool: it exits 0 whenever aggregation succeeds, so a partially
populated runs tree can still be inspected.

    python gate1_metrics.py suites/si_ge_h_cl_ladder_v1.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from common import load_result, write_json

ENGINES = ("orca", "pyscf-cpu", "gpu4pyscf")
# Fixed order and orientation; every metric is left minus right.
ENGINE_PAIRS = (
    ("gpu4pyscf", "pyscf-cpu"),
    ("pyscf-cpu", "orca"),
    ("gpu4pyscf", "orca"),
)
CPU_GPU_PAIR = ENGINE_PAIRS[0]
CPU_ORCA_PAIR = ENGINE_PAIRS[1]
SUMMARY_SCHEMA = "gate1-metrics-summary-v1"
TOLERANCE_STATUS = "provisional_not_scientifically_frozen"
PROVISIONAL_THRESHOLDS = {
    "energy_abs_hartree": 5e-6,
    "gradient_rms_hartree_per_bohr": 2e-5,
    "gradient_max_hartree_per_bohr": 1e-4,
}
AXES = ("x", "y", "z")

CASE_METRIC_HEADERS = [
    "case_id",
    "category",
    "left_engine",
    "right_engine",
    "energy_signed_difference_hartree",
    "energy_absolute_difference_hartree",
    "gradient_component_rmse_hartree_per_bohr",
    "gradient_component_mae_hartree_per_bohr",
    "gradient_max_absolute_difference_hartree_per_bohr",
    "gradient_max_atom_index_zero_based",
    "gradient_max_axis",
]
PERFORMANCE_HEADERS = [
    "case_id",
    "category",
    "engine",
    "converged",
    "energy_hartree",
    "s2",
    "s2_target",
    "s2_deviation",
    "wall_time_seconds",
    "scf_wall_time_seconds",
    "gradient_wall_time_seconds",
]
# Older results and every ORCA result predate the split timers; those cells stay
# blank instead of being invented.
OPTIONAL_PERFORMANCE_FIELDS = (
    "s2",
    "s2_target",
    "s2_deviation",
    "wall_time_seconds",
    "scf_wall_time_seconds",
    "gradient_wall_time_seconds",
)


def gradient_matrix(result: dict[str, Any]) -> list[list[float]]:
    raw = result.get("gradient_hartree_per_bohr")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"{result.get('engine')!r} result has no non-empty "
            "gradient_hartree_per_bohr array."
        )
    gradient: list[list[float]] = []
    for atom_index, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"Gradient row {atom_index} must contain three components.")
        gradient.append([float(value) for value in row])
    return gradient


def require_comparable(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Refuse to compare results that do not describe the same converged input.

    A silent comparison across different fingerprints would report a numeric
    difference that is really an input difference, so this raises instead.
    """
    left_case = left.get("case")
    right_case = right.get("case")
    if not isinstance(left_case, dict) or not isinstance(right_case, dict):
        raise ValueError("Both results must contain case metadata.")
    for key in ("case_id", "input_fingerprint_sha256"):
        if left_case.get(key) != right_case.get(key):
            raise ValueError(
                f"Cannot compare {left.get('engine')!r} and {right.get('engine')!r} "
                f"results with different case.{key}: "
                f"{left_case.get(key)!r} != {right_case.get(key)!r}."
            )
    if left.get("converged") is not True or right.get("converged") is not True:
        raise ValueError(
            f"Both results for case {left_case.get('case_id')!r} must report "
            f"converged=true ({left.get('engine')}={left.get('converged')!r}, "
            f"{right.get('engine')}={right.get('converged')!r})."
        )


def pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    require_comparable(left, right)
    left_gradient = gradient_matrix(left)
    right_gradient = gradient_matrix(right)
    if len(left_gradient) != len(right_gradient):
        raise ValueError(
            f"Gradient atom counts differ for case {left['case']['case_id']!r}."
        )

    absolute_differences: list[tuple[float, int, int]] = []
    squared_sum = 0.0
    absolute_sum = 0.0
    for atom_index, (left_row, right_row) in enumerate(zip(left_gradient, right_gradient)):
        for axis, (left_value, right_value) in enumerate(zip(left_row, right_row)):
            difference = left_value - right_value
            absolute_differences.append((abs(difference), atom_index, axis))
            squared_sum += difference * difference
            absolute_sum += abs(difference)
    component_count = len(absolute_differences)
    max_abs, max_atom, max_axis = max(absolute_differences)

    energy_difference = float(left["energy_hartree"]) - float(right["energy_hartree"])
    return {
        "energy_signed_difference_hartree": energy_difference,
        "energy_absolute_difference_hartree": abs(energy_difference),
        "gradient_component_rmse_hartree_per_bohr": math.sqrt(squared_sum / component_count),
        "gradient_component_mae_hartree_per_bohr": absolute_sum / component_count,
        "gradient_max_absolute_difference_hartree_per_bohr": max_abs,
        "gradient_max_atom_index_zero_based": max_atom,
        "gradient_max_axis": AXES[max_axis],
    }


def load_available_results(runs: Path, case_id: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for engine in ENGINES:
        path = runs / case_id / engine / "result.json"
        if path.is_file():
            results[engine] = load_result(path)
    return results


def performance_row(
    case_id: str, category: str, engine: str, result: dict[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "engine": engine,
        "converged": result.get("converged"),
        "energy_hartree": result.get("energy_hartree"),
    }
    for field in OPTIONAL_PERFORMANCE_FIELDS:
        row[field] = result.get(field)
    return row


def worst_of(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {"value": None, "case_id": None}
    worst = max(rows, key=lambda row: row[key])
    return {"value": worst[key], "case_id": worst["case_id"]}


def pair_rows(case_rows: list[dict[str, Any]], pair: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        row
        for row in case_rows
        if (row["left_engine"], row["right_engine"]) == pair
    ]


def evaluate_gate(
    case_rows: list[dict[str, Any]], case_ids: list[str], missing_gpu: list[str]
) -> dict[str, Any]:
    """Evaluate the provisional CPU-GPU numeric gate of plan section 7."""
    rows = pair_rows(case_rows, CPU_GPU_PAIR)
    cases: list[dict[str, Any]] = []
    for row in rows:
        failed_metrics: list[str] = []
        if (
            row["energy_absolute_difference_hartree"]
            > PROVISIONAL_THRESHOLDS["energy_abs_hartree"]
        ):
            failed_metrics.append("energy_abs_hartree")
        if (
            row["gradient_component_rmse_hartree_per_bohr"]
            > PROVISIONAL_THRESHOLDS["gradient_rms_hartree_per_bohr"]
        ):
            failed_metrics.append("gradient_rms_hartree_per_bohr")
        if (
            row["gradient_max_absolute_difference_hartree_per_bohr"]
            > PROVISIONAL_THRESHOLDS["gradient_max_hartree_per_bohr"]
        ):
            failed_metrics.append("gradient_max_hartree_per_bohr")
        cases.append(
            {
                "case_id": row["case_id"],
                "passed": not failed_metrics,
                "failed_metrics": failed_metrics,
            }
        )
    passed_case_count = sum(1 for case in cases if case["passed"])
    final_calculation_success = not missing_gpu
    # The gate is a statement about every case in the suite, so a suite whose
    # CPU counterparts are absent must not pass vacuously with zero pairs.
    full_pairing = len(cases) == len(case_ids)
    return {
        "left_engine": CPU_GPU_PAIR[0],
        "right_engine": CPU_GPU_PAIR[1],
        "thresholds": dict(PROVISIONAL_THRESHOLDS),
        "tolerance_status": TOLERANCE_STATUS,
        "evaluated_case_count": len(cases),
        "passed_case_count": passed_case_count,
        "failed_case_count": len(cases) - passed_case_count,
        "pairing_coverage": {
            "required_case_count": len(case_ids),
            "paired_case_count": len(cases),
            "passed": full_pairing,
        },
        "final_calculation_success": {
            "required_case_count": len(case_ids),
            "gpu4pyscf_result_count": len(case_ids) - len(missing_gpu),
            "passed": final_calculation_success,
        },
        "cases": cases,
        "passed": full_pairing
        and passed_case_count == len(cases)
        and final_calculation_success,
    }


def evaluate_relative_condition(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Plan section 7: the CPU-GPU difference should stay below the ORCA-CPU one."""
    cpu_orca = {row["case_id"]: row for row in pair_rows(case_rows, CPU_ORCA_PAIR)}
    compared: list[str] = []
    energy_violations: list[str] = []
    gradient_violations: list[str] = []
    for row in pair_rows(case_rows, CPU_GPU_PAIR):
        reference = cpu_orca.get(row["case_id"])
        if reference is None:
            continue
        compared.append(row["case_id"])
        if (
            row["energy_absolute_difference_hartree"]
            > reference["energy_absolute_difference_hartree"]
        ):
            energy_violations.append(row["case_id"])
        if (
            row["gradient_component_rmse_hartree_per_bohr"]
            > reference["gradient_component_rmse_hartree_per_bohr"]
        ):
            gradient_violations.append(row["case_id"])
    return {
        "description": (
            "The CPU-GPU difference must stay at or below the corresponding "
            "CPU-ORCA difference for the same case."
        ),
        "compared_case_count": len(compared),
        "compared_case_ids": compared,
        "energy_absolute_difference_violations": energy_violations,
        "gradient_component_rmse_violations": gradient_violations,
        "passed": not energy_violations and not gradient_violations,
    }


def build_report(suite: dict[str, Any], runs: Path) -> dict[str, Any]:
    case_ids: list[str] = []
    case_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {engine: [] for engine in ENGINES}

    for entry in suite["cases"]:
        case_id = str(entry["case_id"])
        category = str(entry.get("category", ""))
        case_ids.append(case_id)
        results = load_available_results(runs, case_id)
        for engine in ENGINES:
            if engine in results:
                performance_rows.append(
                    performance_row(case_id, category, engine, results[engine])
                )
            else:
                missing[engine].append(case_id)
        for left_engine, right_engine in ENGINE_PAIRS:
            if left_engine not in results or right_engine not in results:
                continue
            case_rows.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "left_engine": left_engine,
                    "right_engine": right_engine,
                    **pair_metrics(results[left_engine], results[right_engine]),
                }
            )

    pairs = []
    for left_engine, right_engine in ENGINE_PAIRS:
        rows = pair_rows(case_rows, (left_engine, right_engine))
        pairs.append(
            {
                "left_engine": left_engine,
                "right_engine": right_engine,
                "paired_case_count": len(rows),
                "worst_energy_absolute_difference_hartree": worst_of(
                    rows, "energy_absolute_difference_hartree"
                ),
                "worst_gradient_component_rmse_hartree_per_bohr": worst_of(
                    rows, "gradient_component_rmse_hartree_per_bohr"
                ),
                "worst_gradient_max_absolute_difference_hartree_per_bohr": worst_of(
                    rows, "gradient_max_absolute_difference_hartree_per_bohr"
                ),
            }
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite_id": str(suite["suite_id"]),
        "case_count": len(case_ids),
        "engine_result_counts": {
            engine: len(case_ids) - len(missing[engine]) for engine in ENGINES
        },
        "pairs": pairs,
        "provisional_cpu_gpu_gate": evaluate_gate(
            case_rows, case_ids, missing[CPU_GPU_PAIR[0]]
        ),
        "relative_condition": evaluate_relative_condition(case_rows),
        "missing": missing,
    }
    return {
        "summary": summary,
        "case_rows": case_rows,
        "performance_rows": performance_rows,
    }


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                ["" if row.get(header) is None else row[header] for header in headers]
            )


def format_worst(entry: dict[str, Any]) -> str:
    if entry["value"] is None:
        return "n/a"
    return f"{float(entry['value']):.6e} ({entry['case_id']})"


def print_summary(summary: dict[str, Any]) -> None:
    counts = summary["engine_result_counts"]
    print(
        f"suite={summary['suite_id']} cases={summary['case_count']} "
        + " ".join(f"{engine}={counts[engine]}" for engine in ENGINES)
    )
    if not any(counts.values()):
        print(
            "WARNING: no engine results were found for this suite; the metric "
            "tables are empty."
        )
    for pair in summary["pairs"]:
        print(
            f"{pair['left_engine']} - {pair['right_engine']}: "
            f"paired={pair['paired_case_count']} "
            f"worst_energy_abs={format_worst(pair['worst_energy_absolute_difference_hartree'])} "
            f"worst_gradient_rmse={format_worst(pair['worst_gradient_component_rmse_hartree_per_bohr'])} "
            f"worst_gradient_max={format_worst(pair['worst_gradient_max_absolute_difference_hartree_per_bohr'])}"
        )
    gate = summary["provisional_cpu_gpu_gate"]
    success = gate["final_calculation_success"]
    coverage = gate["pairing_coverage"]
    print(
        f"provisional_cpu_gpu_gate={'PASS' if gate['passed'] else 'FAIL'} "
        f"({TOLERANCE_STATUS}) passed_cases={gate['passed_case_count']}/"
        f"{gate['evaluated_case_count']} failed_cases={gate['failed_case_count']} "
        f"paired={coverage['paired_case_count']}/{coverage['required_case_count']} "
        f"final_calculation_success={success['gpu4pyscf_result_count']}/"
        f"{success['required_case_count']}"
    )
    for case in gate["cases"]:
        if not case["passed"]:
            print(f"  gate FAIL {case['case_id']}: {', '.join(case['failed_metrics'])}")
    relative = summary["relative_condition"]
    print(
        f"relative_condition={'PASS' if relative['passed'] else 'FAIL'} "
        f"compared={relative['compared_case_count']} "
        f"energy_violations={len(relative['energy_absolute_difference_violations'])} "
        f"gradient_violations={len(relative['gradient_component_rmse_violations'])}"
    )
    for engine in ENGINES:
        missing = summary["missing"][engine]
        if missing:
            print(f"missing {engine}: {len(missing)} case(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="Suite manifest JSON.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="validation/orca_gpu4pyscf directory; results are read from <root>/runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated tables (default: <root>/analysis).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "analysis").resolve()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    report = build_report(suite, root / "runs")
    summary = report["summary"]
    suite_id = summary["suite_id"]

    case_metrics_path = output_dir / f"gate1_case_metrics_{suite_id}.csv"
    performance_path = output_dir / f"gate1_performance_{suite_id}.csv"
    summary_path = output_dir / f"gate1_summary_{suite_id}.json"
    write_csv(case_metrics_path, CASE_METRIC_HEADERS, report["case_rows"])
    write_csv(performance_path, PERFORMANCE_HEADERS, report["performance_rows"])
    write_json(summary_path, summary)

    print_summary(summary)
    print(f"case_metrics_csv={case_metrics_path} rows={len(report['case_rows'])}")
    print(f"performance_csv={performance_path} rows={len(report['performance_rows'])}")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
