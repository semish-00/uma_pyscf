#!/usr/bin/env python3
"""Generate density-fitting manifests for a small relative-energy sentinel."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from common import load_case

SUITE_ID = "c3_density_fit_relative_v1"
CATEGORY = "c3_density_fit_relative"
SENTINELS = (
    ("sih4_bond1_x0p85", "sih4_td_seed"),
    ("sih4_bond1_x1p30", "sih4_td_seed"),
    ("sih4_random_sigma0p12_s20260814", "sih4_td_seed"),
    ("sicl4_bond1_x0p85", "sicl4_td_seed"),
    ("sicl4_bond1_x1p30", "sicl4_td_seed"),
    ("sicl4_random_sigma0p12_s20261014", "sicl4_td_seed"),
    ("h3si_gecl3_random_sigma0p04_s20261213", "h3si_gecl3_staggered_seed"),
    ("h3si_gecl3_random_sigma0p12_s20261214", "h3si_gecl3_staggered_seed"),
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def generate(root: Path) -> tuple[Path, list[Path]]:
    configs = root / "configs"
    suites = root / "suites"
    suites.mkdir(parents=True, exist_ok=True)
    ladder = _load_json(suites / "si_ge_h_cl_ladder_v1.json")
    resources = {
        str(row["case_id"]): dict(row["resources"])
        for row in ladder.get("cases", [])
        if isinstance(row, dict) and isinstance(row.get("resources"), dict)
    }

    cases: list[dict[str, Any]] = []
    written: list[Path] = []
    for structure_case_id, reference_case_id in SENTINELS:
        baseline_path = configs / f"{structure_case_id}.json"
        baseline = _load_json(baseline_path)
        load_case(baseline_path)
        settings = baseline.get("pyscf")
        if not isinstance(settings, dict):
            raise ValueError(f"{structure_case_id} has no pyscf settings object.")
        expected = {
            "grid_level": 5,
            "nlc_grid_level": 5,
            "density_fit": False,
        }
        actual = {key: settings.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"{structure_case_id} baseline settings {actual} != {expected}.")
        if structure_case_id not in resources:
            raise ValueError(f"Ladder has no resources for {structure_case_id}.")

        candidate = deepcopy(baseline)
        case_id = f"{structure_case_id}_c3_density_fit"
        candidate["case_id"] = case_id
        candidate["pyscf"]["density_fit"] = True
        config_path = configs / f"{case_id}.json"
        config_path.write_text(
            json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
        )
        load_case(config_path)
        written.append(config_path)
        cases.append(
            {
                "case_id": case_id,
                "category": CATEGORY,
                "structure_case_id": structure_case_id,
                "reference_case_id": reference_case_id,
                "reference_density_fit_case_id": (
                    f"{reference_case_id}_c3_density_fit"
                ),
                "config": f"configs/{case_id}.json",
                "resources": dict(resources[structure_case_id]),
            }
        )

    suite = {
        "schema": "crosscode-suite-v1",
        "suite_id": SUITE_ID,
        "description": (
            "CPU PySCF density-fitting sentinel for relative-energy error. "
            "Each distorted structure is compared with a same-composition seed; "
            "the suite is diagnostic and does not freeze a production protocol."
        ),
        "selection_status": "diagnostic_not_production_frozen",
        "case_count": len(cases),
        "engine_jobs_per_case": ["pyscf-cpu"],
        "execution_policy": {"sequential": False, "stop_on_first_failure": False},
        "cases": cases,
    }
    suite_path = suites / f"{SUITE_ID}.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return suite_path, written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    suite_path, written = generate(args.root.resolve())
    print(f"Wrote {len(written)} configs and {suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
