#!/usr/bin/env python3
"""Generate the SiH3/GeH3 charge-spin mini matrix of validation plan section 3.

Every state reuses an existing seed geometry, so the suite isolates CPU-GPU
numerical parity across charge and spin states instead of mixing in a geometry
change. No structure file is created or modified here.

The chemical states are candidates only: the plan requires a separate
scientific review before any of them is accepted as a training label, and the
suite records that with `state_selection_status`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import Atom, load_case, read_xyz
from generate_ladder_suite import manifest as ladder_manifest

SUITE_ID = "charge_spin_mini_v1"
CATEGORY = "charge_spin_matrix"
BASES = ("sih3", "geh3")
SEED_SUFFIX = "_doublet_planar_seed"
REFERENCE_LABEL = "neutral_doublet_reference"
# The seed radicals are four-atom hydrides; the ladder allocates them 8 CPUs.
NCPUS = 8
RESOURCES = {"ncpus": NCPUS, "memory_gb": 32, "walltime": "06:00:00"}
# (label, charge, multiplicity). Every entry satisfies electron-count/spin
# parity for both SiH3 (17 electrons) and GeH3 (35 electrons); load_case
# re-checks that for the written manifests rather than trusting this table.
STATES = (
    ("neutral_quartet", 0, 4),
    ("cation_singlet", 1, 1),
    ("cation_triplet", 1, 3),
    ("anion_singlet", -1, 1),
    ("anion_triplet", -1, 3),
)
DESCRIPTION = (
    "SiH3 and GeH3 charge/spin mini matrix on the existing neutral doublet seed "
    "geometries. These cases probe CPU-GPU numerical parity across charge and "
    "spin states at a fixed geometry and are NOT yet approved as training-label "
    "states; the scientific state selection is still under review."
)


def state_manifest(
    base: str, case_id: str, charge: int, multiplicity: int, atoms: tuple[Atom, ...]
) -> dict[str, Any]:
    """Copy the ladder manifest conventions and change only the electronic state."""
    data = ladder_manifest(case_id, multiplicity, list(atoms), NCPUS)
    # Both keys already exist, so the emitted key order stays identical to the
    # ladder manifests; only the seed geometry and the charge differ.
    data["structure"] = f"../structures/{base}{SEED_SUFFIX}.xyz"
    data["charge"] = charge
    return data


def generate(root: Path) -> tuple[Path, list[Path]]:
    configs = root / "configs"
    structures = root / "structures"
    suites = root / "suites"
    suites.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    written: list[Path] = []
    for base in BASES:
        seed_case_id = f"{base}{SEED_SUFFIX}"
        seed_config = configs / f"{seed_case_id}.json"
        seed_structure = structures / f"{seed_case_id}.xyz"
        for required in (seed_config, seed_structure):
            if not required.is_file():
                raise FileNotFoundError(
                    f"{required} is required as the reference state of {SUITE_ID}."
                )
        atoms = read_xyz(seed_structure)
        reference = load_case(seed_config)
        cases.append(
            {
                "case_id": seed_case_id,
                "category": CATEGORY,
                "charge": reference.charge,
                "multiplicity": reference.multiplicity,
                "config": f"configs/{seed_case_id}.json",
                "resources": dict(RESOURCES),
                "state": {"base": base, "label": REFERENCE_LABEL},
            }
        )

        for label, charge, multiplicity in STATES:
            case_id = f"{base}_{label}"
            config_path = configs / f"{case_id}.json"
            data = state_manifest(base, case_id, charge, multiplicity, atoms)
            config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            written.append(config_path)
            # Prove electron-count/spin parity for what was actually written.
            case = load_case(config_path)
            if (case.charge, case.multiplicity) != (charge, multiplicity):
                raise ValueError(f"{config_path} does not describe {label}.")
            cases.append(
                {
                    "case_id": case_id,
                    "category": CATEGORY,
                    "charge": charge,
                    "multiplicity": multiplicity,
                    "config": f"configs/{case_id}.json",
                    "resources": dict(RESOURCES),
                    "state": {"base": base, "label": label},
                }
            )

    suite = {
        "schema": "crosscode-suite-v1",
        "suite_id": SUITE_ID,
        "description": DESCRIPTION,
        "state_selection_status": "pending_scientific_review",
        "case_count": len(cases),
        "engine_jobs_per_case": ["pyscf-cpu", "gpu4pyscf"],
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
