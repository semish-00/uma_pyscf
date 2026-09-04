#!/usr/bin/env python3
"""Build blind HCl-elimination interpolation candidates for T0-only parents."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from uma_pyscf.core.ids import canonical_json_fingerprint, sha256_of_file
from uma_pyscf.core.io import write_json_atomic
from uma_pyscf.sampling.filters import minimum_distance_violation
from uma_pyscf.sampling.generate import read_xyz_structure
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord
from uma_pyscf.schemas.label_record import ElectronicState, Structure

FRACTIONS = (0.35, 0.55, 0.75)
HCL_BOND_ANGSTROM = 1.28
FRAGMENT_CL_DISTANCE_ANGSTROM = 4.0


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(value / norm for value in vector)


def _add(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return tuple(factor * value for value in vector)


def _interpolate(
    start: tuple[float, float, float], end: tuple[float, float, float], fraction: float
) -> tuple[float, float, float]:
    return tuple(
        (1.0 - fraction) * left + fraction * right
        for left, right in zip(start, end, strict=True)
    )


def build(config_path: Path) -> tuple[CandidateManifest, dict[str, Any]]:
    structure_dir = config_path.parent / "structures" / "t0_dimer_parents_v1"
    paths = sorted(structure_dir.glob("*.xyz"))
    if len(paths) != 6:
        raise ValueError(f"expected six T0 parent XYZ files, found {len(paths)}")
    config = {
        "builder": "t0_hcl_elimination_linear_interpolation_v1",
        "source_files": [
            {"path": str(path.relative_to(config_path.parent)), "sha256": sha256_of_file(path)}
            for path in paths
        ],
        "fractions": list(FRACTIONS),
        "hcl_bond_angstrom": HCL_BOND_ANGSTROM,
        "fragment_cl_distance_angstrom": FRAGMENT_CL_DISTANCE_ANGSTROM,
        "state": {"charge": 0, "multiplicity": 1},
        "selection": {
            "metal_indices": [0, 1],
            "chlorine_index": "first_Cl_after_metals",
            "hydrogen_index": "first_H_after_selected_Cl",
        },
    }
    records: list[CandidateRecord] = []
    entries: list[dict[str, Any]] = []
    for path in paths:
        parent_id = path.stem
        source = read_xyz_structure(path)
        numbers = source.atomic_numbers
        chlorine_index = next(index for index in range(2, len(numbers)) if numbers[index] == 17)
        hydrogen_index = next(
            index for index in range(chlorine_index + 1, len(numbers)) if numbers[index] == 1
        )
        positions = source.positions_angstrom
        metal = positions[0]
        chlorine = positions[chlorine_index]
        hydrogen = positions[hydrogen_index]
        direction = _unit(tuple(b - a for a, b in zip(metal, chlorine, strict=True)))
        target_chlorine = _add(metal, _scale(direction, FRAGMENT_CL_DISTANCE_ANGSTROM))
        target_hydrogen = _add(
            target_chlorine, _scale(direction, -HCL_BOND_ANGSTROM)
        )
        reaction_id = f"{parent_id}_to_hcl_elimination"
        for frame, fraction in enumerate(FRACTIONS):
            changed = list(positions)
            changed[chlorine_index] = _interpolate(
                chlorine, target_chlorine, fraction
            )
            changed[hydrogen_index] = _interpolate(hydrogen, target_hydrogen, fraction)
            structure = Structure(
                atomic_numbers=numbers,
                positions_angstrom=tuple(changed),
                parent_structure_id=parent_id,
                sampling_method="reaction_interpolation",
                random_seed=None,
            )
            violation = minimum_distance_violation(structure, 0.65)
            record_id = f"t0_hcl_interpolation_v1_{parent_id}_f{frame:02d}"
            if violation is not None:
                entries.append(
                    {
                        "record_id": record_id,
                        "status": "rejected",
                        "reason": "minimum_distance",
                        "details": violation,
                    }
                )
                continue
            records.append(
                CandidateRecord(
                    record_id=record_id,
                    structure=structure,
                    state=ElectronicState(
                        charge=0,
                        multiplicity=1,
                        spin_2s=0,
                        state_provenance="t0_reserved_neutral_singlet",
                    ),
                    generation_parameters={
                        "reaction_id": reaction_id,
                        "path_id": reaction_id,
                        "frame_index": frame,
                        "interpolation_fraction": fraction,
                        "moving_atom_indices": [hydrogen_index, chlorine_index],
                        "selection_mode": "blind_fixed_fraction",
                    },
                )
            )
            entries.append({"record_id": record_id, "status": "accepted", "reason": None})
    manifest = CandidateManifest(
        sampling_id="t0_hcl_interpolation_18_v1",
        config_sha256=canonical_json_fingerprint(config),
        config=config,
        records=tuple(records),
    )
    report = {
        "schema": "uma-pyscf-t0-interpolation-build-report-v1",
        "sampling_id": manifest.sampling_id,
        "counts": {
            "proposed": len(paths) * len(FRACTIONS),
            "accepted": len(records),
            "rejected": len(entries) - len(records),
        },
        "entries": entries,
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    marker = args.config_root / "t0_dimer_parent_seeds_v1.yaml"
    manifest, report = build(marker)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{manifest.sampling_id}_candidates.json"
    report_path = args.output_dir / f"{manifest.sampling_id}_build_report.json"
    write_json_atomic(manifest_path, manifest.to_dict())
    write_json_atomic(report_path, report)
    print(
        f"proposed={report['counts']['proposed']} accepted={report['counts']['accepted']} "
        f"rejected={report['counts']['rejected']}"
    )


if __name__ == "__main__":
    main()
