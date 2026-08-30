#!/usr/bin/env python3
"""Diagnose CPU/GPU SCF-root divergence with a shared explicit initial density."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
from typing import Any

from common import case_record, load_case, target_s2, write_json


def _host_list(value: Any) -> Any:
    if hasattr(value, "get"):
        value = value.get()
    if isinstance(value, (tuple, list)):
        return [_host_list(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run(config_path: Path, device: str, init_guess: str) -> dict[str, Any]:
    case = load_case(config_path)
    from pyscf import dft, gto

    mol = gto.M(
        atom=[(atom.symbol, (atom.x, atom.y, atom.z)) for atom in case.atoms],
        basis=case.basis,
        charge=case.charge,
        spin=case.spin_2s,
        unit="Angstrom",
        verbose=int(case.raw["pyscf"].get("verbose", 3)),
    )
    mf = dft.RKS(mol) if case.spin_2s == 0 else dft.UKS(mol)
    mf.xc = case.functional
    mf.conv_tol = float(case.raw["scf"]["conv_tol"])
    mf.max_cycle = int(case.raw["scf"]["max_cycle"])
    mf.max_memory = float(case.raw["pyscf"]["max_memory_mb"])
    mf.grids.level = int(case.raw["pyscf"]["grid_level"])
    mf.nlcgrids.level = int(case.raw["pyscf"]["nlc_grid_level"])
    if bool(case.raw["pyscf"]["density_fit"]):
        mf = mf.density_fit()

    # Generate the density on the CPU object before to_gpu(), so both engines
    # receive the exact same matrix instead of independently selecting a guess.
    dm0 = mf.get_init_guess(mol, key=init_guess)
    if device == "gpu":
        mf = mf.to_gpu()
    energy = float(mf.kernel(dm0=dm0))
    if not bool(mf.converged):
        raise RuntimeError("SCF did not converge.")
    gradients = mf.nuc_grad_method()
    if hasattr(gradients, "grid_response"):
        gradients.grid_response = bool(case.raw["pyscf"]["grid_response"])
    gradient = gradients.kernel()
    spin = mf.spin_square() if hasattr(mf, "spin_square") else (None, None)
    package_names = (
        "gpu4pyscf-cuda12x",
        "gpu4pyscf-cuda11x",
        "gpu4pyscf-cuda13x",
        "gpu4pyscf",
    )
    gpu_version = None
    for name in package_names:
        try:
            gpu_version = metadata.version(name)
            break
        except metadata.PackageNotFoundError:
            pass
    return {
        "schema": "scf-root-diagnostic-v1",
        "engine": "pyscf-cpu" if device == "cpu" else "gpu4pyscf",
        "case": case_record(case),
        "init_guess": init_guess,
        "initial_density_generated_before_to_gpu": True,
        "converged": True,
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": _host_list(gradient),
        "s2": None if spin[0] is None else float(spin[0]),
        "s2_target": target_s2(case.spin_2s),
        "multiplicity_from_spin_square": None if spin[1] is None else float(spin[1]),
        "mo_energy_hartree": _host_list(mf.mo_energy),
        "mo_occupation": _host_list(mf.mo_occ),
        "gpu4pyscf": gpu_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--init-guess", choices=("minao", "atom", "hcore"), default="minao")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.device, args.init_guess)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("engine", "energy_hartree", "s2")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
