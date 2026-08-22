#!/usr/bin/env python3
"""Workstream A2 installation smoke test for a GPU4PySCF host.

Runs an ordered sequence of small checks — CuPy sees the GPU, kernels launch,
PySCF/GPU4PySCF import, and a tiny ωB97M-V/def2-TZVPD RKS and UKS energy plus
analytic gradient run on the GPU — and writes a JSON report. Every check
records passed/failed/skipped with the observed detail, so a broken layer of
the driver/CUDA/CuPy/GPU4PySCF stack is identified instead of guessed.

The script is batch-safe: it reads nothing from stdin and needs no login
shell. Use --device-id to pin a specific GPU.

    python gpu_smoke_check.py --output runs/gpu_smoke_check.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import traceback
from typing import Any, Callable

from common import write_json

SCHEMA = "gpu-smoke-check-v1"
FUNCTIONAL = "wb97m-v"
BASIS = "def2-tzvpd"


def check_cupy_import(state: dict[str, Any]) -> dict[str, Any]:
    import cupy

    state["cupy"] = cupy
    return {"cupy_version": cupy.__version__}


def check_gpu_visible(state: dict[str, Any]) -> dict[str, Any]:
    cupy = state["cupy"]
    count = int(cupy.cuda.runtime.getDeviceCount())
    if count < 1:
        raise RuntimeError("CuPy reports zero CUDA devices.")
    device_id = state["device_id"]
    if device_id >= count:
        raise RuntimeError(f"--device-id {device_id} but only {count} device(s) present.")
    cupy.cuda.Device(device_id).use()
    properties = cupy.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name")
    return {
        "device_count": count,
        "selected_device": device_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": name.decode() if isinstance(name, bytes) else name,
        "memory_total_mib": int(properties["totalGlobalMem"] // (1024 * 1024)),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
    }


def check_cupy_kernel(state: dict[str, Any]) -> dict[str, Any]:
    cupy = state["cupy"]
    matrix = cupy.arange(16.0, dtype=cupy.float64).reshape(4, 4)
    trace = float(cupy.trace(matrix @ matrix).get())
    expected = 1240.0
    if abs(trace - expected) > 1e-9:
        raise RuntimeError(f"GPU matmul trace {trace} != {expected}.")
    return {"matmul_trace": trace}


def check_pyscf_import(state: dict[str, Any]) -> dict[str, Any]:
    import pyscf
    from pyscf import dft

    state["pyscf"] = pyscf
    return {
        "pyscf_version": pyscf.__version__,
        "libxc_version": getattr(dft.libxc, "__version__", None),
    }


def check_gpu4pyscf_import(state: dict[str, Any]) -> dict[str, Any]:
    import gpu4pyscf

    return {"gpu4pyscf_version": getattr(gpu4pyscf, "__version__", None)}


def _gradient_run(state: dict[str, Any], atom: str, charge: int, spin_2s: int) -> dict[str, Any]:
    from pyscf import dft, gto

    mol = gto.M(atom=atom, basis=BASIS, charge=charge, spin=spin_2s, unit="Angstrom", verbose=0)
    mf = dft.RKS(mol) if spin_2s == 0 else dft.UKS(mol)
    mf.xc = FUNCTIONAL
    mf.conv_tol = 1e-9
    mf.max_cycle = 100
    mf.grids.level = int(state["grid_level"])
    mf.nlcgrids.level = int(state["nlc_grid_level"])
    if not hasattr(mf, "to_gpu"):
        raise RuntimeError("This PySCF installation does not expose to_gpu().")
    mf = mf.to_gpu()
    energy = float(mf.kernel())
    if not bool(mf.converged):
        raise RuntimeError("SCF did not converge.")
    do_nlc = bool(mf.do_nlc()) if hasattr(mf, "do_nlc") else None
    if do_nlc is False:
        raise RuntimeError(f"{FUNCTIONAL} ran without its VV10 nonlocal correlation.")
    gradients = mf.nuc_grad_method()
    if hasattr(gradients, "grid_response"):
        gradients.grid_response = True
    gradient = gradients.kernel()
    if hasattr(gradient, "get"):
        gradient = gradient.get()
    flat = [float(value) for row in gradient.tolist() for value in row]
    if not all(value == value and abs(value) < 1e3 for value in flat):
        raise RuntimeError(f"Gradient contains non-finite entries: {flat}.")
    detail: dict[str, Any] = {
        "energy_hartree": energy,
        "gradient_max_abs_hartree_per_bohr": max(abs(value) for value in flat),
        "nonlocal_correlation_active": do_nlc,
        "engine_class": type(mf).__module__ + "." + type(mf).__name__,
    }
    if spin_2s and hasattr(mf, "spin_square"):
        spin_result = mf.spin_square()
        detail["s2"] = float(spin_result[0])
    return detail


def check_rks_gradient(state: dict[str, Any]) -> dict[str, Any]:
    return _gradient_run(state, "H 0 0 0; H 0 0 0.74", charge=0, spin_2s=0)


def check_uks_gradient(state: dict[str, Any]) -> dict[str, Any]:
    detail = _gradient_run(state, "H 0 0 0; H 0 0 1.20; H 0 0 2.90", charge=0, spin_2s=1)
    s2 = detail.get("s2")
    if s2 is not None and abs(s2 - 0.75) > 0.3:
        raise RuntimeError(f"UKS doublet <S^2>={s2} is far from 0.75.")
    return detail


CHECKS: tuple[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
    ("cupy_import", check_cupy_import),
    ("gpu_visible", check_gpu_visible),
    ("cupy_kernel", check_cupy_kernel),
    ("pyscf_import", check_pyscf_import),
    ("gpu4pyscf_import", check_gpu4pyscf_import),
    ("rks_wb97mv_gradient_gpu", check_rks_gradient),
    ("uks_wb97mv_gradient_gpu", check_uks_gradient),
)


def run_checks(device_id: int, grid_level: int, nlc_grid_level: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "device_id": device_id,
        "grid_level": grid_level,
        "nlc_grid_level": nlc_grid_level,
    }
    rows: list[dict[str, Any]] = []
    blocked = False
    for name, check in CHECKS:
        if blocked:
            rows.append({"name": name, "status": "skipped", "detail": None, "error": None})
            continue
        try:
            detail = check(state)
        except Exception as exc:
            rows.append(
                {
                    "name": name,
                    "status": "failed",
                    "detail": None,
                    "error": "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip(),
                }
            )
            blocked = True
        else:
            rows.append({"name": name, "status": "passed", "detail": detail, "error": None})
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node() or None,
        "python": platform.python_version(),
        "functional": FUNCTIONAL,
        "basis": BASIS,
        "grid_level": grid_level,
        "nlc_grid_level": nlc_grid_level,
        "checks": rows,
        "passed": all(row["status"] == "passed" for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--grid-level",
        type=int,
        default=3,
        help="Ordinary grid level for the tiny DFT checks (speed, not production).",
    )
    parser.add_argument("--nlc-grid-level", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Report JSON path.")
    args = parser.parse_args()

    report = run_checks(args.device_id, args.grid_level, args.nlc_grid_level)
    if args.output:
        write_json(args.output, report)
        print(f"Wrote {args.output}")
    for row in report["checks"]:
        line = f"{row['name']}: {row['status']}"
        if row["error"]:
            line += f" ({row['error']})"
        print(line)
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
