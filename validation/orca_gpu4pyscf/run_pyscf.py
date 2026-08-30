#!/usr/bin/env python3
"""Run a normalized CPU PySCF or GPU4PySCF energy-and-gradient calculation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

from common import RESULT_SCHEMA, case_record, load_case, target_s2, write_json


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def build_plan(config_path: str | Path, device: str) -> dict[str, Any]:
    case = load_case(config_path)
    pyscf_settings = case.raw["pyscf"]
    init_guess = str(pyscf_settings.get("init_guess", "minao"))
    if init_guess not in {"minao", "atom", "hcore"}:
        raise ValueError(
            f"Unsupported pyscf.init_guess {init_guess!r}; use minao, atom, or hcore."
        )
    return {
        "dry_run": True,
        "engine": "pyscf-cpu" if device == "cpu" else "gpu4pyscf",
        "case": case_record(case),
        "reference": "RKS" if case.spin_2s == 0 else "UKS",
        "settings": {
            "scf": case.raw["scf"],
            "verbose": int(pyscf_settings.get("verbose", 3)),
            "grid_level": int(pyscf_settings["grid_level"]),
            "nlc_grid_level": int(pyscf_settings["nlc_grid_level"]),
            "grid_response": bool(pyscf_settings["grid_response"]),
            "density_fit": bool(pyscf_settings["density_fit"]),
            "init_guess": init_guess,
            "initial_density_generated_before_device_conversion": True,
            "max_memory_mb": float(pyscf_settings["max_memory_mb"]),
        },
        "tolerances": case.tolerances,
        "tolerance_status": case.raw.get("tolerance_status"),
    }


def run(config_path: str | Path, device: str) -> dict[str, Any]:
    case = load_case(config_path)
    plan = build_plan(config_path, device)
    try:
        import pyscf
        from pyscf import dft, gto
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is required for execution. Use --dry-run locally, or run this "
            "on the configured CPU/GPU calculation host."
        ) from exc

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
    started = time.perf_counter()
    initial_guess_started = started
    init_guess = str(case.raw["pyscf"].get("init_guess", "minao"))
    # Generate one density matrix on the CPU object and pass it explicitly to
    # both engines. Otherwise GPU4PySCF can generate a different implicit guess
    # after to_gpu() and converge to another open-shell SCF root.
    dm0 = mf.get_init_guess(mol, key=init_guess)
    initial_guess_wall_time = time.perf_counter() - initial_guess_started
    if device == "gpu":
        if not hasattr(mf, "to_gpu"):
            raise RuntimeError("This PySCF installation does not expose to_gpu().")
        mf = mf.to_gpu()
        for attribute, key in (("grids", "grid_level"), ("nlcgrids", "nlc_grid_level")):
            requested_level = int(case.raw["pyscf"][key])
            converted_level = getattr(getattr(mf, attribute), "level", None)
            if converted_level != requested_level:
                raise RuntimeError(
                    f"to_gpu() changed {attribute}.level from {requested_level} to "
                    f"{converted_level!r} for {case.case_id}; refusing to run with "
                    "an unintended grid."
                )

    kernel_started = time.perf_counter()
    energy = float(mf.kernel(dm0=dm0))
    scf_kernel_wall_time = time.perf_counter() - kernel_started
    scf_wall_time = time.perf_counter() - started
    if not bool(mf.converged):
        raise RuntimeError(f"SCF did not converge for {case.case_id}.")

    gradients = mf.nuc_grad_method()
    if hasattr(gradients, "max_memory"):
        gradients.max_memory = mf.max_memory
    requested_grid_response = bool(case.raw["pyscf"]["grid_response"])
    if requested_grid_response and not hasattr(gradients, "grid_response"):
        raise RuntimeError(
            "The selected gradient implementation does not expose grid_response; "
            "refusing to compare an incompletely specified DFT gradient."
        )
    if hasattr(gradients, "grid_response"):
        gradients.grid_response = requested_grid_response
    gradient_started = time.perf_counter()
    gradient = gradients.kernel()
    gradient_wall_time = time.perf_counter() - gradient_started
    wall_time = time.perf_counter() - started
    if hasattr(gradient, "get"):  # CuPy array from the GPU gradient path
        gradient = gradient.get()

    s2 = None
    multiplicity_from_s2 = None
    if hasattr(mf, "spin_square"):
        spin_result = mf.spin_square()
        s2 = float(spin_result[0])
        if len(spin_result) > 1:
            multiplicity_from_s2 = float(spin_result[1])

    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "pyscf": pyscf.__version__,
        "libxc": getattr(dft.libxc, "__version__", None),
        "gpu4pyscf": package_version(
            "gpu4pyscf-cuda12x", "gpu4pyscf-cuda11x", "gpu4pyscf-cuda13x", "gpu4pyscf"
        ),
        "cupy": package_version(
            "cupy-cuda12x", "cupy-cuda11x", "cupy-cuda13x", "cupy"
        ),
        "cuda_device": None,
    }
    if device == "gpu":
        try:
            import cupy

            device_index = int(cupy.cuda.runtime.getDevice())
            runtime["cuda_device"] = device_index
            runtime["cuda_runtime_version"] = int(cupy.cuda.runtime.runtimeGetVersion())
            properties = cupy.cuda.runtime.getDeviceProperties(device_index)
            name = properties.get("name")
            runtime["cuda_device_name"] = (
                name.decode() if isinstance(name, bytes) else name
            )
            runtime["cuda_device_total_memory_bytes"] = int(properties["totalGlobalMem"])
            runtime["cuda_free_memory_bytes_after_run"] = int(
                cupy.cuda.runtime.memGetInfo()[0]
            )
            pool = cupy.get_default_memory_pool()
            runtime["cupy_pool_total_bytes_after_run"] = int(pool.total_bytes())
        except Exception as exc:  # provenance should not invalidate a completed result
            runtime["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"

    do_nlc = bool(mf.do_nlc()) if hasattr(mf, "do_nlc") else None
    result = {
        "schema": RESULT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "pyscf-cpu" if device == "cpu" else "gpu4pyscf",
        "engine_runtime": runtime,
        "case": case_record(case),
        "settings": {
            **plan["settings"],
            "reference": "RKS" if case.spin_2s == 0 else "UKS",
            "nonlocal_correlation_active": do_nlc,
        },
        "converged": True,
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": gradient.tolist(),
        "s2": s2,
        "s2_target": target_s2(case.spin_2s),
        "s2_deviation": None if s2 is None else s2 - target_s2(case.spin_2s),
        "multiplicity_from_spin_square": multiplicity_from_s2,
        "wall_time_seconds": wall_time,
        "scf_wall_time_seconds": scf_wall_time,
        "initial_guess_wall_time_seconds": initial_guess_wall_time,
        "scf_kernel_wall_time_seconds": scf_kernel_wall_time,
        "gradient_wall_time_seconds": gradient_wall_time,
        "tolerances": case.tolerances,
        "tolerance_status": case.raw.get("tolerance_status"),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Cross-code case manifest JSON.")
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", help="Normalized result JSON path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without importing PySCF.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        print(json.dumps(build_plan(args.config, args.device), indent=2, ensure_ascii=False))
        return 0
    if not args.output:
        raise SystemExit("--output is required unless --dry-run is used.")
    result = run(args.config, args.device)
    write_json(args.output, result)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
