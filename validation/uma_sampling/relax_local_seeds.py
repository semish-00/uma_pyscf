#!/usr/bin/env python
"""Relax C0 seed candidates with base UMA and emit a displacement config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ase.io import write  # type: ignore[import-not-found]
from ase.optimize import BFGS  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from uma_pyscf.core.ids import sha256_of_file
from uma_pyscf.core.io import read_json, write_json_atomic
from uma_pyscf.inference.uma import _initialize_uma
from uma_pyscf.schemas.candidate import CandidateManifest


def run(manifest_path: Path, output_dir: Path, model_cache_dir: Path) -> None:
    manifest = CandidateManifest.from_dict(read_json(manifest_path))
    context = _initialize_uma(
        model_name="uma-s-1p2",
        checkpoint_path=None,
        model_cache_dir=model_cache_dir,
        task="omol",
        device="cuda",
        inference_settings="default",
        seed=41,
        fairchem_core_version="2.22.0",
    )
    ase = context["ase"]
    calculator = context["calculator"]
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    structures: list[dict[str, str]] = []
    operations: list[dict[str, Any]] = []

    for index, candidate in enumerate(manifest.records):
        structure_id = candidate.structure.parent_structure_id
        if structure_id is None:
            raise ValueError(f"{candidate.record_id} has no parent_structure_id")
        atoms = ase.Atoms(
            numbers=candidate.structure.atomic_numbers,
            positions=candidate.structure.positions_angstrom,
            pbc=False,
        )
        atoms.info.update(charge=candidate.state.charge, spin=candidate.state.multiplicity)
        calculator.reset()
        atoms.calc = calculator
        initial_energy = float(atoms.get_potential_energy())
        initial_positions = np.asarray(atoms.positions, dtype=float).copy()
        initial_max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        optimizer = BFGS(
            atoms,
            trajectory=str(output_dir / f"{structure_id}.traj"),
            logfile=str(output_dir / f"{structure_id}.log"),
        )
        converged = bool(optimizer.run(fmax=0.03, steps=500))
        final_max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        xyz_path = structures_dir / f"{structure_id}.xyz"
        write(xyz_path, atoms, format="xyz")
        summaries.append(
            {
                "record_id": candidate.record_id,
                "structure_id": structure_id,
                "converged": converged,
                "optimizer_steps": optimizer.nsteps,
                "initial_energy_ev": initial_energy,
                "final_energy_ev": float(atoms.get_potential_energy()),
                "initial_max_force_ev_per_angstrom": initial_max_force,
                "final_max_force_ev_per_angstrom": final_max_force,
                "max_displacement_angstrom": float(
                    np.linalg.norm(np.asarray(atoms.positions) - initial_positions, axis=1).max()
                ),
                "xyz_path": f"structures/{structure_id}.xyz",
            }
        )
        structures.append({"id": structure_id, "xyz_path": f"structures/{structure_id}.xyz"})
        for offset, sigma in ((0, 0.02), (4, 0.06)):
            operations.append(
                {
                    "kind": "cartesian_displacement",
                    "structure": structure_id,
                    "charge": 0,
                    "multiplicity": 1,
                    "sigma_angstrom": sigma,
                    "count": 4,
                    "seed": 2026091000 + index * 10 + offset,
                }
            )

    summary = {
        "schema": "uma-pyscf-uma-relaxation-summary-v1",
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_of_file(manifest_path),
        },
        "model": {"name": "uma-s-1p2", "task": "omol", "seed": 41},
        "records": summaries,
    }
    write_json_atomic(output_dir / "relaxation_summary.json", summary)
    if not all(record["converged"] for record in summaries):
        raise RuntimeError("At least one UMA seed relaxation did not converge")
    sampling_config = {
        "schema_version": 1,
        "sampling_id": "calibration_local_displacements_v1",
        "created": "2026-09-01",
        "derived_from": summary["source_manifest"],
        "description": "C0 local displacements around six base-UMA-relaxed parents.",
        "structures": structures,
        "operations": operations,
        "filters": {
            "covalent_factor": 0.65,
            "bond_factor": 1.3,
            "allow_fragments": False,
            "duplicate_decimals": 3,
        },
    }
    write_json_atomic(output_dir / "sampling_config.json", sampling_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.manifest, arguments.output_dir, arguments.model_cache_dir)
