#!/usr/bin/env python3
"""Export parity-plot source tables as CSV for any engine pair.

Defaults reproduce the historical CPU PySCF vs ORCA tables. For the GPU
validation phase run, for example:

    python export_parity_csv.py --left-engine gpu4pyscf --right-engine pyscf-cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.4220674763
ENGINES = ("orca", "pyscf-cpu", "gpu4pyscf")
REFERENCE_CASES = {
    "H4Si": "sih4_td_seed",
    "GeH4": "geh4_td_seed",
    "Cl4Si": "sicl4_td_seed",
    "Cl4Ge": "gecl4_td_seed",
    "Cl3GeH3Si": "h3si_gecl3_staggered_seed",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def formula(result: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for atom in result["case"]["atoms"]:
        element = str(atom["element"])
        counts[element] = counts.get(element, 0) + 1
    return "".join(
        element + (str(count) if count > 1 else "")
        for element, count in sorted(counts.items())
    )


def flattened_gradient(result: dict[str, Any]) -> list[float]:
    return [float(value) for row in result["gradient_hartree_per_bohr"] for value in row]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="validation/orca_gpu4pyscf directory",
    )
    parser.add_argument("--suite", type=Path, help="Suite manifest JSON (default: 29-case ladder).")
    parser.add_argument("--left-engine", choices=ENGINES, default="pyscf-cpu")
    parser.add_argument("--right-engine", choices=ENGINES, default="orca")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.left_engine == args.right_engine:
        raise SystemExit("--left-engine and --right-engine must differ.")
    root = args.root.resolve()
    left_engine, right_engine = args.left_engine, args.right_engine
    output_dir = (args.output_dir or root / "analysis").resolve()
    suite = load_json(args.suite or root / "suites/si_ge_h_cl_ladder_v1.json")
    runs = root / "runs"

    records: list[dict[str, Any]] = []
    for entry in suite["cases"]:
        case_id = str(entry["case_id"])
        records.append(
            {
                "case_id": case_id,
                "category": str(entry["category"]),
                "left": load_json(runs / case_id / left_engine / "result.json"),
                "right": load_json(runs / case_id / right_engine / "result.json"),
            }
        )

    references: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
    for formula_name, case_id in REFERENCE_CASES.items():
        references[formula_name] = (
            load_json(runs / case_id / left_engine / "result.json"),
            load_json(runs / case_id / right_engine / "result.json"),
            case_id,
        )

    energy_rows: list[list[Any]] = []
    for record in records:
        formula_name = formula(record["right"])
        if formula_name not in references:
            continue
        left_reference, right_reference, reference_case_id = references[formula_name]
        if record["case_id"] == reference_case_id:
            continue
        left_relative = (
            float(record["left"]["energy_hartree"])
            - float(left_reference["energy_hartree"])
        ) * HARTREE_TO_KCAL_MOL
        right_relative = (
            float(record["right"]["energy_hartree"])
            - float(right_reference["energy_hartree"])
        ) * HARTREE_TO_KCAL_MOL
        energy_rows.append(
            [
                record["case_id"], formula_name, record["category"], reference_case_id,
                left_engine, right_engine,
                left_relative, right_relative, left_relative - right_relative,
            ]
        )

    gradient_rows: list[list[Any]] = []
    for record in records:
        left_gradient = flattened_gradient(record["left"])
        right_gradient = flattened_gradient(record["right"])
        if len(left_gradient) != len(right_gradient):
            raise ValueError(f"Gradient size mismatch for {record['case_id']}")
        component_errors = [
            left_value - right_value
            for left_value, right_value in zip(left_gradient, right_gradient)
        ]
        left_rms = rms(left_gradient)
        right_rms = rms(right_gradient)
        error_rms = rms(component_errors)
        gradient_rows.append(
            [
                record["case_id"], record["category"], len(left_gradient),
                left_engine, right_engine,
                left_rms, right_rms, left_rms - right_rms,
                error_rms, error_rms * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            ]
        )

    pair = f"{left_engine}_vs_{right_engine}"
    energy_path = output_dir / f"relative_energy_parity_{pair}.csv"
    gradient_path = output_dir / f"gradient_rms_parity_{pair}.csv"
    write_csv(
        energy_path,
        [
            "case_id", "formula", "category", "reference_case_id",
            "left_engine", "right_engine",
            "left_relative_energy_kcal_mol", "right_relative_energy_kcal_mol",
            "left_minus_right_kcal_mol",
        ],
        energy_rows,
    )
    write_csv(
        gradient_path,
        [
            "case_id", "category", "n_gradient_components",
            "left_engine", "right_engine",
            "left_gradient_component_rms_hartree_per_bohr",
            "right_gradient_component_rms_hartree_per_bohr",
            "left_minus_right_gradient_rms_hartree_per_bohr",
            "component_error_rmse_hartree_per_bohr",
            "component_error_rmse_ev_per_angstrom",
        ],
        gradient_rows,
    )
    print(f"energy_csv={energy_path} rows={len(energy_rows)}")
    print(f"gradient_csv={gradient_path} rows={len(gradient_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
