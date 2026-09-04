#!/usr/bin/env python3
"""Build six neutral closed-shell dimer parents reserved exclusively for T0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ase import Atoms
from ase.io import read, write
import numpy as np

PARENT_SPECS = (
    ("t0_si2h3cl3_split12", "Si", "Si", (1, 2)),
    ("t0_si2h2cl4_split13", "Si", "Si", (1, 3)),
    ("t0_ge2h3cl3_split12", "Ge", "Ge", (1, 2)),
    ("t0_ge2h2cl4_split13", "Ge", "Ge", (1, 3)),
    ("t0_sigeh3cl3_si1_ge2", "Si", "Ge", (1, 2)),
    ("t0_sigeh2cl4_si1_ge3", "Si", "Ge", (1, 3)),
)

CORE_BOND = {("Ge", "Ge"): 2.45, ("Ge", "Si"): 2.40, ("Si", "Si"): 2.34}
TERMINAL_BOND = {
    ("Ge", "Cl"): 2.15,
    ("Ge", "H"): 1.53,
    ("Si", "Cl"): 2.05,
    ("Si", "H"): 1.48,
}


def _terminal_vectors(side: int, phase_degrees: float) -> np.ndarray:
    axial = side / 3.0
    radial = math.sqrt(8.0 / 9.0)
    angles = np.radians(np.asarray([0.0, 120.0, 240.0]) + phase_degrees)
    return np.column_stack(
        (
            np.full(3, axial),
            radial * np.cos(angles),
            radial * np.sin(angles),
        )
    )


def build_parent(
    left_element: str,
    right_element: str,
    chlorine_counts: tuple[int, int],
) -> Atoms:
    key = tuple(sorted((left_element, right_element)))
    core_distance = CORE_BOND[key]
    positions = [np.asarray([-core_distance / 2.0, 0.0, 0.0])]
    positions.append(np.asarray([core_distance / 2.0, 0.0, 0.0]))
    symbols = [left_element, right_element]
    for center, element, chlorine_count, side, phase in (
        (positions[0], left_element, chlorine_counts[0], -1, 0.0),
        (positions[1], right_element, chlorine_counts[1], 1, 60.0),
    ):
        substituents = ["Cl"] * chlorine_count + ["H"] * (3 - chlorine_count)
        for symbol, direction in zip(
            substituents, _terminal_vectors(side, phase), strict=True
        ):
            symbols.append(symbol)
            positions.append(center + TERMINAL_BOND[(element, symbol)] * direction)
    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    if len(atoms) != 8 or atoms.get_chemical_symbols().count("Cl") != sum(chlorine_counts):
        raise AssertionError("unexpected T0 parent composition")
    distances = atoms.get_all_distances(mic=False)
    nonzero = distances[distances > 0.0]
    if float(nonzero.min()) < 1.0:
        raise ValueError(f"constructed atom collision at {float(nonzero.min()):.6f} Angstrom")
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for parent_id, left, right, chlorine_counts in PARENT_SPECS:
        atoms = build_parent(left, right, chlorine_counts)
        path = args.output_dir / f"{parent_id}.xyz"
        write(path, atoms, format="xyz")
        loaded = read(path, index=0)
        if loaded.pbc.any() or loaded.get_chemical_symbols() != atoms.get_chemical_symbols():
            raise ValueError(f"round-trip validation failed for {parent_id}")
        if not np.allclose(loaded.positions, atoms.positions, rtol=0.0, atol=1e-8):
            raise ValueError(f"coordinate round-trip validation failed for {parent_id}")
        summary.append(
            {
                "parent_structure_id": parent_id,
                "formula": atoms.get_chemical_formula(),
                "atom_count": len(atoms),
                "pbc": atoms.pbc.tolist(),
                "minimum_pair_distance_angstrom": float(
                    atoms.get_all_distances(mic=False)[
                        atoms.get_all_distances(mic=False) > 0.0
                    ].min()
                ),
                "xyz_path": str(path),
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
