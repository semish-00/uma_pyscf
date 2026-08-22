#!/usr/bin/env python3
"""Summarize available ORCA/PySCF results for a cross-code suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare import compare_results
from common import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-comparisons", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    rows = []
    for entry in suite["cases"]:
        case_id = str(entry["case_id"])
        run_dir = root / "validation/orca_gpu4pyscf/runs" / case_id
        paths = {
            "orca": run_dir / "orca/result.json",
            "pyscf-cpu": run_dir / "pyscf-cpu/result.json",
        }
        row = {"case_id": case_id, "category": entry["category"],
               "orca_available": paths["orca"].is_file(),
               "pyscf_cpu_available": paths["pyscf-cpu"].is_file(), "comparison": None}
        if all(path.is_file() for path in paths.values()):
            orca = json.loads(paths["orca"].read_text(encoding="utf-8"))
            pyscf = json.loads(paths["pyscf-cpu"].read_text(encoding="utf-8"))
            comparison = compare_results(pyscf, orca)
            row["comparison"] = {
                "passed": comparison["passed"],
                "energy_absolute_difference_hartree": comparison["energy_absolute_difference_hartree"],
                "gradient_rms_difference_hartree_per_bohr": comparison["gradient_rms_difference_hartree_per_bohr"],
                "gradient_max_absolute_difference_hartree_per_bohr": comparison["gradient_max_absolute_difference_hartree_per_bohr"],
                "pyscf_s2": pyscf.get("s2"),
                "pyscf_s2_deviation": pyscf.get("s2_deviation"),
            }
            if args.write_comparisons:
                write_json(run_dir / "pyscf-cpu-vs-orca.json", comparison)
        rows.append(row)
    summary = {
        "schema": "crosscode-suite-summary-v1",
        "suite_id": suite["suite_id"],
        "case_count": len(rows),
        "orca_results": sum(row["orca_available"] for row in rows),
        "pyscf_cpu_results": sum(row["pyscf_cpu_available"] for row in rows),
        "paired_results": sum(row["comparison"] is not None for row in rows),
        "paired_passed": sum(bool(row["comparison"] and row["comparison"]["passed"]) for row in rows),
        "rows": rows,
    }
    if args.output:
        write_json(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("suite_id", "case_count", "orca_results", "pyscf_cpu_results", "paired_results", "paired_passed")}, indent=2))
    for row in rows:
        if row["comparison"] is not None:
            result = row["comparison"]
            print(f"{row['case_id']}: {'PASS' if result['passed'] else 'FAIL'} dE={result['energy_absolute_difference_hartree']:.6e} grad_rms={result['gradient_rms_difference_hartree_per_bohr']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
