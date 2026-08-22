#!/usr/bin/env python3
"""Generate the deterministic Si/Ge/H/Cl cross-code validation ladder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable


@dataclass(frozen=True)
class Atom:
    symbol: str
    x: float
    y: float
    z: float


TETRAHEDRAL = {
    "sih4": ("Si", "H", 1.480),
    "geh4": ("Ge", "H", 1.525),
    "sicl4": ("Si", "Cl", 2.020),
    "gecl4": ("Ge", "Cl", 2.110),
}
RADICALS = {
    "sih3": ("Si", "H", 1.480),
    "geh3": ("Ge", "H", 1.525),
    "sicl3": ("Si", "Cl", 2.020),
    "gecl3": ("Ge", "Cl", 2.110),
}
COVALENT_RADII = {"H": 0.31, "Si": 1.11, "Ge": 1.20, "Cl": 1.02}
SCAN_FACTORS = (0.85, 1.15, 1.30)


def tetrahedral(center: str, ligand: str, bond: float) -> list[Atom]:
    component = bond / math.sqrt(3.0)
    directions = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
    return [Atom(center, 0.0, 0.0, 0.0)] + [
        Atom(ligand, component * x, component * y, component * z)
        for x, y, z in directions
    ]


def trigonal(center: str, ligand: str, bond: float) -> list[Atom]:
    atoms = [Atom(center, 0.0, 0.0, 0.0)]
    for angle_deg in (0.0, 120.0, 240.0):
        angle = math.radians(angle_deg)
        atoms.append(Atom(ligand, bond * math.cos(angle), bond * math.sin(angle), 0.0))
    return atoms


def mixed_dimer(left_ligand: str, right_ligand: str) -> list[Atom]:
    """Return a staggered L3Si-GeR3 structure with tetrahedral local geometry."""
    si_ge = 2.40
    atoms = [Atom("Si", 0.0, 0.0, -si_ge / 2), Atom("Ge", 0.0, 0.0, si_ge / 2)]
    radial = math.sqrt(8.0 / 9.0)
    bonds = {("Si", "H"): 1.480, ("Si", "Cl"): 2.020,
             ("Ge", "H"): 1.525, ("Ge", "Cl"): 2.110}
    for center, ligand, z0, z_sign, phase in (
        ("Si", left_ligand, -si_ge / 2, -1.0, 0.0),
        ("Ge", right_ligand, si_ge / 2, 1.0, 60.0),
    ):
        bond = bonds[(center, ligand)]
        for angle_deg in (phase, phase + 120.0, phase + 240.0):
            angle = math.radians(angle_deg)
            atoms.append(
                Atom(
                    ligand,
                    bond * radial * math.cos(angle),
                    bond * radial * math.sin(angle),
                    z0 + z_sign * bond / 3.0,
                )
            )
    return atoms


def scan_first_bond(atoms: list[Atom], factor: float) -> list[Atom]:
    first = atoms[1]
    return [atoms[0], Atom(first.symbol, first.x * factor, first.y * factor, first.z * factor), *atoms[2:]]


def displaced(atoms: list[Atom], sigma: float, seed: int) -> list[Atom]:
    rng = random.Random(seed)
    shifts = [[rng.gauss(0.0, sigma) for _ in range(3)] for _ in atoms]
    # Remove rigid translation so the perturbation describes internal geometry.
    mean = [sum(row[axis] for row in shifts) / len(shifts) for axis in range(3)]
    result = [
        Atom(atom.symbol, atom.x + row[0] - mean[0], atom.y + row[1] - mean[1], atom.z + row[2] - mean[2])
        for atom, row in zip(atoms, shifts)
    ]
    for i, left in enumerate(result):
        for right in result[i + 1:]:
            distance = math.sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2)
            cutoff = 0.65 * (COVALENT_RADII[left.symbol] + COVALENT_RADII[right.symbol])
            if distance < cutoff:
                raise ValueError(f"seed={seed} produced {left.symbol}-{right.symbol} distance {distance:.3f} A")
    return result


def write_xyz(path: Path, atoms: Iterable[Atom], comment: str) -> None:
    rows = list(atoms)
    text = [str(len(rows)), comment]
    text.extend(f"{a.symbol:<2s} {a.x: .12f} {a.y: .12f} {a.z: .12f}" for a in rows)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def resources(atoms: list[Atom]) -> dict[str, object]:
    has_cl = any(atom.symbol == "Cl" for atom in atoms)
    is_large = len(atoms) >= 8
    ncpus = 16 if has_cl or is_large else 8
    memory_gb = 64 if ncpus == 16 else 32
    return {"ncpus": ncpus, "memory_gb": memory_gb, "walltime": "12:00:00" if ncpus == 16 else "06:00:00"}


def manifest(case_id: str, multiplicity: int, atoms: list[Atom], ncpus: int) -> dict[str, object]:
    memory_mb = 48000 if ncpus == 16 else 24000
    return {
        "schema_version": 1,
        "case_id": case_id,
        "structure": f"../structures/{case_id}.xyz",
        "charge": 0,
        "multiplicity": multiplicity,
        "calculation": "energy_gradient",
        "method": {"functional": "wb97m-v", "basis": "def2-tzvpd"},
        "scf": {"conv_tol": 1e-10, "max_cycle": 250},
        "pyscf": {
            "verbose": 3,
            "grid_level": 5,
            "nlc_grid_level": 5,
            "grid_response": True,
            "density_fit": False,
            "max_memory_mb": memory_mb,
        },
        "orca": {
            "version": "6.0.0",
            "nprocs": ncpus,
            "maxcore_mb_per_process": 3000,
            "keywords": ["VeryTightSCF", "DEFGRID3", "SCNL", "NORI", "NOCOSX", "NoAutoStart"],
        },
        "tolerances": {
            "energy_abs_hartree": 5e-5,
            "gradient_rms_hartree_per_bohr": 2e-4,
            "gradient_max_hartree_per_bohr": 5e-4,
        },
        "tolerance_status": "provisional_not_scientifically_frozen",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    configs = root / "configs"
    structures = root / "structures"
    suites = root / "suites"
    for directory in (configs, structures, suites):
        directory.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []

    def add(case_id: str, atoms: list[Atom], multiplicity: int, category: str, generation: dict[str, object]) -> None:
        allocation = resources(atoms)
        write_xyz(structures / f"{case_id}.xyz", atoms, json.dumps(generation, sort_keys=True, separators=(",", ":")))
        data = manifest(case_id, multiplicity, atoms, int(allocation["ncpus"]))
        (configs / f"{case_id}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        cases.append({"case_id": case_id, "category": category, "multiplicity": multiplicity,
                      "config": f"configs/{case_id}.json", "resources": allocation, "generation": generation})

    tetra_bases: dict[str, list[Atom]] = {}
    for name, (center, ligand, bond) in TETRAHEDRAL.items():
        base = tetrahedral(center, ligand, bond)
        tetra_bases[name] = base
        for factor in SCAN_FACTORS:
            suffix = f"{factor:.2f}".replace(".", "p")
            add(f"{name}_bond1_x{suffix}", scan_first_bond(base, factor), 1, "bond_scan",
                {"kind": "one_bond_scale", "parent": f"{name}_td_seed", "bond_atom_indices_zero_based": [0, 1], "factor": factor})

    radical_bases: dict[str, list[Atom]] = {}
    for name, (center, ligand, bond) in RADICALS.items():
        base = trigonal(center, ligand, bond)
        radical_bases[name] = base
        add(f"{name}_doublet_planar_seed", base, 2, "radical",
            {"kind": "trigonal_planar_seed", "bond_angstrom": bond})

    mixed_bases = {
        "h3si_geh3": mixed_dimer("H", "H"),
        "h3si_gecl3": mixed_dimer("H", "Cl"),
        "cl3si_geh3": mixed_dimer("Cl", "H"),
    }
    for name, atoms in mixed_bases.items():
        add(f"{name}_staggered_seed", atoms, 1, "mixed_si_ge_h_cl",
            {"kind": "staggered_si_ge_dimer_seed", "si_ge_angstrom": 2.40})

    random_bases = {
        "sih4": tetra_bases["sih4"],
        "geh4": tetra_bases["geh4"],
        "sicl4": tetra_bases["sicl4"],
        "gecl4": tetra_bases["gecl4"],
        "h3si_gecl3": mixed_bases["h3si_gecl3"],
    }
    seed = 20260813
    for base_index, (name, atoms) in enumerate(random_bases.items()):
        for sigma_index, sigma in enumerate((0.04, 0.12)):
            this_seed = seed + 100 * base_index + sigma_index
            suffix = f"{sigma:.2f}".replace(".", "p")
            add(f"{name}_random_sigma{suffix}_s{this_seed}", displaced(atoms, sigma, this_seed), 1,
                "random_displacement", {"kind": "cartesian_gaussian_displacement", "parent": name,
                "sigma_angstrom": sigma, "seed": this_seed, "translation_removed": True,
                "minimum_distance_filter_covalent_radii_factor": 0.65})

    suite = {
        "schema": "crosscode-suite-v1",
        "suite_id": "si_ge_h_cl_ladder_v1",
        "description": "Controlled bond scans, neutral doublet radicals, Si-Ge mixed molecules, and deterministic random displacements.",
        "case_count": len(cases),
        "engine_jobs_per_case": ["orca", "pyscf-cpu"],
        "cases": cases,
    }
    suite_path = suites / "si_ge_h_cl_ladder_v1.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases and {suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
