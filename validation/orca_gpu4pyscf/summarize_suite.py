#!/usr/bin/env python3
"""Summarize available results for a cross-code suite for any engine pair.

Defaults reproduce the historical CPU PySCF vs ORCA summary. For Part I of the
GPU validation plan, the priority pair is:

    python summarize_suite.py suites/gpu_smoke_v1.json \
        --left-engine gpu4pyscf --right-engine pyscf-cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare import compare_results
from common import write_json

ENGINES = ("orca", "pyscf-cpu", "gpu4pyscf")
SUMMARY_SCHEMA = "crosscode-suite-summary-v2"


def engine_extras(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "s2": result.get("s2"),
        "s2_deviation": result.get("s2_deviation"),
        "wall_time_seconds": result.get("wall_time_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--left-engine", choices=ENGINES, default="pyscf-cpu")
    parser.add_argument("--right-engine", choices=ENGINES, default="orca")
    parser.add_argument("--write-comparisons", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.left_engine == args.right_engine:
        raise SystemExit("--left-engine and --right-engine must differ.")
    root = args.root.resolve()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    rows = []
    for entry in suite["cases"]:
        case_id = str(entry["case_id"])
        run_dir = root / "validation/orca_gpu4pyscf/runs" / case_id
        paths = {
            engine: run_dir / engine / "result.json"
            for engine in (args.left_engine, args.right_engine)
        }
        row = {"case_id": case_id, "category": entry["category"],
               "left_available": paths[args.left_engine].is_file(),
               "right_available": paths[args.right_engine].is_file(), "comparison": None}
        if all(path.is_file() for path in paths.values()):
            left = json.loads(paths[args.left_engine].read_text(encoding="utf-8"))
            right = json.loads(paths[args.right_engine].read_text(encoding="utf-8"))
            comparison = compare_results(left, right)
            row["comparison"] = {
                "passed": comparison["passed"],
                "energy_absolute_difference_hartree": comparison["energy_absolute_difference_hartree"],
                "gradient_rms_difference_hartree_per_bohr": comparison["gradient_rms_difference_hartree_per_bohr"],
                "gradient_max_absolute_difference_hartree_per_bohr": comparison["gradient_max_absolute_difference_hartree_per_bohr"],
                "left": engine_extras(left),
                "right": engine_extras(right),
            }
            if args.write_comparisons:
                write_json(
                    run_dir / f"{args.left_engine}-vs-{args.right_engine}.json",
                    comparison,
                )
        rows.append(row)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "suite_id": suite["suite_id"],
        "left_engine": args.left_engine,
        "right_engine": args.right_engine,
        "case_count": len(rows),
        "left_results": sum(row["left_available"] for row in rows),
        "right_results": sum(row["right_available"] for row in rows),
        "paired_results": sum(row["comparison"] is not None for row in rows),
        "paired_passed": sum(bool(row["comparison"] and row["comparison"]["passed"]) for row in rows),
        "rows": rows,
    }
    if args.output:
        write_json(args.output, summary)
    print(json.dumps({key: summary[key] for key in (
        "suite_id", "left_engine", "right_engine", "case_count",
        "left_results", "right_results", "paired_results", "paired_passed")}, indent=2))
    for row in rows:
        if row["comparison"] is not None:
            result = row["comparison"]
            print(f"{row['case_id']}: {'PASS' if result['passed'] else 'FAIL'} dE={result['energy_absolute_difference_hartree']:.6e} grad_rms={result['gradient_rms_difference_hartree_per_bohr']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
