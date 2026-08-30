#!/usr/bin/env python3
"""Generate the density-fitting CPU-GPU charge/spin parity probe."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from common import load_case

SOURCE_SUITE_ID = "charge_spin_mini_v1"
SUITE_ID = "charge_spin_density_fit_probe_v1"
MINAO_SUITE_ID = "charge_spin_density_fit_minao_probe_v1"
CATEGORY = "charge_spin_density_fit_probe"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def generate(
    root: Path,
    *,
    suite_id: str = SUITE_ID,
    case_suffix: str = "_density_fit_probe",
    explicit_minao: bool = False,
) -> tuple[Path, list[Path]]:
    configs = root / "configs"
    suites = root / "suites"
    source = _load_json(suites / f"{SOURCE_SUITE_ID}.json")
    cases: list[dict[str, Any]] = []
    written: list[Path] = []
    for row in source["cases"]:
        base_case_id = str(row["case_id"])
        baseline_path = root / str(row["config"])
        baseline = _load_json(baseline_path)
        load_case(baseline_path)
        settings = baseline.get("pyscf")
        if not isinstance(settings, dict) or settings.get("density_fit") is not False:
            raise ValueError(f"{base_case_id} must be a direct PySCF baseline.")
        candidate = deepcopy(baseline)
        case_id = f"{base_case_id}{case_suffix}"
        candidate["case_id"] = case_id
        candidate["pyscf"]["density_fit"] = True
        if explicit_minao:
            candidate["pyscf"]["init_guess"] = "minao"
        path = configs / f"{case_id}.json"
        path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        load_case(path)
        written.append(path)
        cases.append(
            {
                "case_id": case_id,
                "category": CATEGORY,
                "base_case_id": base_case_id,
                "state_selection_status": row.get(
                    "state_selection_status", "pending_scientific_review"
                ),
                "config": f"configs/{case_id}.json",
                "resources": deepcopy(row["resources"]),
            }
        )
    suite = {
        "schema": "crosscode-suite-v1",
        "suite_id": suite_id,
        "description": (
            "CPU-GPU numerical parity probe for twelve SiH3/GeH3 charge and "
            "spin states using the conditional density-fitting candidate"
            + (" and an explicit shared MINAO initial density. " if explicit_minao else ". ")
            + "State "
            "selection remains pending scientific review and is not teacher-data approval."
        ),
        "selection_status": "numerical_probe_not_teacher_data_approved",
        "source_suite_id": SOURCE_SUITE_ID,
        "candidate_change": {
            "density_fit": {"from": False, "to": True},
            **(
                {"init_guess": {"from": "implicit_default", "to": "explicit_minao"}}
                if explicit_minao
                else {}
            ),
        },
        "case_count": len(cases),
        "engine_jobs_per_case": ["pyscf-cpu", "gpu4pyscf"],
        "execution_policy": {"sequential": True, "stop_on_first_failure": False},
        "cases": cases,
    }
    path = suites / f"{suite_id}.json"
    path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return path, written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    suite_path, written = generate(args.root.resolve())
    minao_suite_path, minao_written = generate(
        args.root.resolve(),
        suite_id=MINAO_SUITE_ID,
        case_suffix="_density_fit_minao_probe",
        explicit_minao=True,
    )
    print(
        f"Wrote {len(written)} implicit configs and {suite_path}; "
        f"{len(minao_written)} explicit-MINAO configs and {minao_suite_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
