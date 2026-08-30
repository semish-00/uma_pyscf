#!/usr/bin/env python3
"""Generate the one-axis-at-a-time GPU4PySCF C3 setting matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from common import load_case

SUITE_ID = "gpu_c3_settings_matrix_v1"
DENSITY_FIT_SUITE_ID = "c3_density_fit_cpu_gpu_v1"
CATEGORY = "gpu_c3_settings"
BASE_CASE_IDS = (
    "sih4_td_seed",
    "sicl4_td_seed",
    "sih3_doublet_planar_seed",
    "h3si_gecl3_staggered_seed",
)
BASELINE_SETTINGS = {
    "grid_level": 5,
    "nlc_grid_level": 5,
    "density_fit": False,
}
VARIANTS = (
    ("grid4", "ordinary_grid", "grid_level", 4),
    ("grid6", "ordinary_grid", "grid_level", 6),
    ("nlc3", "vv10_grid", "nlc_grid_level", 3),
    ("nlc4", "vv10_grid", "nlc_grid_level", 4),
    ("density_fit", "density_fitting", "density_fit", True),
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _resources_by_case(root: Path) -> dict[str, dict[str, Any]]:
    smoke_suite = _load_json(root / "suites" / "gpu_smoke_v1.json")
    resources = {
        str(row["case_id"]): dict(row["resources"])
        for row in smoke_suite.get("cases", [])
        if isinstance(row, dict) and isinstance(row.get("resources"), dict)
    }
    missing = sorted(set(BASE_CASE_IDS) - set(resources))
    if missing:
        raise ValueError(f"gpu_smoke_v1 is missing resources for {missing}.")
    return resources


def variant_manifest(
    baseline: dict[str, Any], base_case_id: str, suffix: str, key: str, value: Any
) -> dict[str, Any]:
    data = deepcopy(baseline)
    settings = data.get("pyscf")
    if not isinstance(settings, dict):
        raise ValueError(f"{base_case_id} has no pyscf settings object.")
    actual_baseline = {name: settings.get(name) for name in BASELINE_SETTINGS}
    if actual_baseline != BASELINE_SETTINGS:
        raise ValueError(
            f"{base_case_id} baseline settings {actual_baseline} != "
            f"{BASELINE_SETTINGS}."
        )
    data["case_id"] = f"{base_case_id}_c3_{suffix}"
    settings[key] = value
    return data


def generate(root: Path) -> tuple[Path, list[Path]]:
    configs = root / "configs"
    suites = root / "suites"
    suites.mkdir(parents=True, exist_ok=True)
    resources = _resources_by_case(root)

    cases: list[dict[str, Any]] = []
    written: list[Path] = []
    for base_case_id in BASE_CASE_IDS:
        baseline_path = configs / f"{base_case_id}.json"
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline = _load_json(baseline_path)
        load_case(baseline_path)
        for suffix, axis, key, value in VARIANTS:
            data = variant_manifest(baseline, base_case_id, suffix, key, value)
            case_id = str(data["case_id"])
            config_path = configs / f"{case_id}.json"
            config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            load_case(config_path)
            written.append(config_path)
            cases.append(
                {
                    "case_id": case_id,
                    "category": CATEGORY,
                    "base_case_id": base_case_id,
                    "axis": axis,
                    "setting_key": key,
                    "baseline_value": BASELINE_SETTINGS[key],
                    "candidate_value": value,
                    "config": f"configs/{case_id}.json",
                    "resources": dict(resources[base_case_id]),
                }
            )

    suite = {
        "schema": "crosscode-suite-v1",
        "suite_id": SUITE_ID,
        "description": (
            "GPU-only C3 scan on four sentinel molecules. Every manifest changes "
            "exactly one PySCF setting from grid=5, VV10 grid=5, direct SCF. "
            "Candidates are diagnostic and are not a frozen production protocol."
        ),
        "selection_status": "diagnostic_not_production_frozen",
        "baseline_settings": dict(BASELINE_SETTINGS),
        "case_count": len(cases),
        "engine_jobs_per_case": ["gpu4pyscf"],
        "execution_policy": {"sequential": True, "stop_on_first_failure": True},
        "cases": cases,
    }
    suite_path = suites / f"{SUITE_ID}.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")

    density_fit_cases = [
        deepcopy(row) for row in cases if row["setting_key"] == "density_fit"
    ]
    density_fit_suite = {
        "schema": "crosscode-suite-v1",
        "suite_id": DENSITY_FIT_SUITE_ID,
        "description": (
            "CPU-GPU C3 candidate check for density fitting on four sentinel "
            "molecules. This separates the density-fitting approximation from "
            "GPU-port numerical differences; it does not freeze density fitting "
            "as the production protocol."
        ),
        "selection_status": "diagnostic_not_production_frozen",
        "baseline_settings": dict(BASELINE_SETTINGS),
        "case_count": len(density_fit_cases),
        "engine_jobs_per_case": ["pyscf-cpu", "gpu4pyscf"],
        "execution_policy": {"sequential": False, "stop_on_first_failure": False},
        "cases": density_fit_cases,
    }
    density_fit_suite_path = suites / f"{DENSITY_FIT_SUITE_ID}.json"
    density_fit_suite_path.write_text(
        json.dumps(density_fit_suite, indent=2) + "\n", encoding="utf-8"
    )
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
