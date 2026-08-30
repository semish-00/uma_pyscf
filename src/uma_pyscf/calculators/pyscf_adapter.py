"""GPU4PySCF adapter for the frozen Gate 1 production protocol.

Scientific imports are deliberately local to :meth:`calculate`, so config
validation, dry runs, and all unit tests work on hosts without PySCF or CUDA.
The initial MINAO density is created on the CPU object and passed explicitly to
the GPU kernel; this is the SCF-root reproducibility fix established in Part I.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
import os
from pathlib import Path
import platform
import time
from typing import Any

from ..core.elements import PERIODIC_SYMBOLS
from ..core.errors import ProvenanceError
from ..core.ids import sha256_of_file
from ..core.spin import target_s2
from ..schemas._fields import require_mapping, require_str
from ..schemas.candidate import CandidateRecord
from ..schemas.label_record import Method, Results
from .model import CalculationFailure, CalculationOutput

__all__ = ["Gpu4PyscfAdapter"]


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    raise ProvenanceError(f"None of the required distributions {names!r} is installed.")


def _checksum_from_file(path: Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").split()[0].lower()
    except (OSError, IndexError) as exc:
        raise ProvenanceError(f"Cannot read checksum from {path}: {exc}.") from exc
    if len(first) != 64 or not set(first) <= set("0123456789abcdef"):
        raise ProvenanceError(f"Checksum file {path} does not start with a sha256 digest.")
    return first


def _runtime_provenance(config: Mapping[str, Any], cupy: Any) -> dict[str, str]:
    engine = require_mapping(config.get("engine"), "config.engine")
    accelerators = os.environ.get("CUPY_ACCELERATORS", "")
    if "cutensor" not in {item.strip() for item in accelerators.split(",") if item.strip()}:
        raise ProvenanceError(
            "CUPY_ACCELERATORS must include 'cutensor' for the frozen GPU environment."
        )
    device_index = int(cupy.cuda.runtime.getDevice())
    properties = cupy.cuda.runtime.getDeviceProperties(device_index)
    raw_name = properties.get("name")
    device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    container_sha_file = Path(
        require_str(engine.get("container_sha256_file"), "config.engine.container_sha256_file")
    )
    lock_file = Path(
        require_str(engine.get("python_lock_file"), "config.engine.python_lock_file")
    )
    return {
        "python": platform.python_version(),
        "pyscf": _distribution_version("pyscf"),
        "gpu4pyscf": _distribution_version(
            "gpu4pyscf-cuda12x", "gpu4pyscf-cuda11x", "gpu4pyscf"
        ),
        "cupy": _distribution_version("cupy-cuda12x", "cupy-cuda11x", "cupy"),
        "cutensor": _distribution_version("cutensor-cu12", "cutensor-cu11", "cutensor"),
        "cuda_runtime_version": str(int(cupy.cuda.runtime.runtimeGetVersion())),
        "cuda_driver_version": str(int(cupy.cuda.runtime.driverGetVersion())),
        "cuda_device_index": str(device_index),
        "cuda_device_name": device_name,
        "cuda_device_total_memory_bytes": str(int(properties["totalGlobalMem"])),
        "container_image": require_str(
            engine.get("container_image"), "config.engine.container_image"
        ),
        "container_image_sha256": _checksum_from_file(container_sha_file),
        "python_overlay": require_str(
            engine.get("python_overlay"), "config.engine.python_overlay"
        ),
        "python_lock_file": str(lock_file),
        "python_lock_sha256": sha256_of_file(lock_file),
    }


def _enforce_frozen_runtime(runtime: Mapping[str, str], config: Mapping[str, Any]) -> None:
    """Refuse a version or GPU mismatch before starting an SCF calculation."""
    engine = require_mapping(config.get("engine"), "config.engine")
    required_versions = require_mapping(
        engine.get("required_versions"), "config.engine.required_versions"
    )
    for name, expected in required_versions.items():
        actual = runtime.get(name)
        if actual != expected:
            raise CalculationFailure(
                "runtime_environment",
                f"Runtime {name} version is {actual!r}; protocol requires {expected!r}.",
            )
    required_gpu = require_str(
        engine.get("required_gpu_name"), "config.engine.required_gpu_name"
    )
    if runtime.get("cuda_device_name") != required_gpu:
        raise CalculationFailure(
            "runtime_environment",
            f"Runtime GPU is {runtime.get('cuda_device_name')!r}; protocol requires "
            f"{required_gpu!r}.",
        )


class Gpu4PyscfAdapter:
    """Calculate one energy/gradient label with GPU4PySCF."""

    def calculate(
        self,
        candidate: CandidateRecord,
        method: Method,
        config: Mapping[str, Any],
        *,
        attempt_id: str,
        resource: Mapping[str, Any],
    ) -> CalculationOutput:
        """Run one isolated candidate using explicit CPU-generated MINAO ``dm0``."""
        try:
            import cupy  # type: ignore[import-not-found]
            import pyscf  # type: ignore[import-not-found]
            from pyscf import dft, gto
        except ImportError as exc:
            raise CalculationFailure(
                "runtime_environment", f"Required GPU4PySCF runtime cannot be imported: {exc}"
            ) from exc

        runtime = _runtime_provenance(config, cupy)
        _enforce_frozen_runtime(runtime, config)
        atoms = [
            (
                PERIODIC_SYMBOLS[atomic_number],
                tuple(candidate.structure.positions_angstrom[index]),
            )
            for index, atomic_number in enumerate(candidate.structure.atomic_numbers)
        ]
        max_memory_mb = int(resource["max_memory_mb"])
        mol = gto.M(
            atom=atoms,
            basis=method.basis,
            ecp=method.ecp,
            charge=candidate.state.charge,
            spin=candidate.state.spin_2s,
            unit="Angstrom",
            verbose=3,
            max_memory=max_memory_mb,
        )
        mf = dft.RKS(mol) if candidate.state.spin_2s == 0 else dft.UKS(mol)
        mf.xc = method.functional
        mf.conv_tol = method.scf_conv_tol
        mf.max_cycle = method.scf_max_cycle
        mf.max_memory = max_memory_mb
        mf.grids.level = method.grid_level
        mf.nlcgrids.level = method.nlc_grid_level
        if method.density_fit:
            mf = mf.density_fit(auxbasis=method.aux_basis)

        started = time.perf_counter()
        guess_started = time.perf_counter()
        dm0 = mf.get_init_guess(mol, key="minao")
        guess_seconds = time.perf_counter() - guess_started
        if not hasattr(mf, "to_gpu"):
            raise CalculationFailure("runtime_environment", "PySCF object has no to_gpu().")
        mf = mf.to_gpu()
        for attribute, expected in (
            ("grids", method.grid_level),
            ("nlcgrids", method.nlc_grid_level),
        ):
            actual = getattr(getattr(mf, attribute), "level", None)
            if actual != expected:
                raise CalculationFailure(
                    "runtime_environment",
                    f"to_gpu() changed {attribute}.level from {expected} to {actual!r}.",
                )

        scf_started = time.perf_counter()
        energy = float(mf.kernel(dm0=dm0))
        scf_seconds = time.perf_counter() - scf_started
        if not bool(mf.converged):
            raise CalculationFailure(
                "scf_not_converged",
                f"SCF did not converge in {method.scf_max_cycle} cycles for "
                f"{candidate.record_id}.",
            )

        gradients = mf.nuc_grad_method()
        if hasattr(gradients, "max_memory"):
            gradients.max_memory = max_memory_mb
        if method.grid_response and not hasattr(gradients, "grid_response"):
            raise CalculationFailure(
                "runtime_environment",
                "GPU gradient implementation does not expose grid_response.",
            )
        if hasattr(gradients, "grid_response"):
            gradients.grid_response = method.grid_response
        gradient_started = time.perf_counter()
        gradient = gradients.kernel()
        gradient_seconds = time.perf_counter() - gradient_started
        if hasattr(gradient, "get"):
            gradient = gradient.get()

        s2_value: float | None = None
        if hasattr(mf, "spin_square"):
            s2_value = float(mf.spin_square()[0])
        s2_target = target_s2(candidate.state.spin_2s)
        total_seconds = time.perf_counter() - started
        raw_payload = {
            "attempt_id": attempt_id,
            "reference": "RKS" if candidate.state.spin_2s == 0 else "UKS",
            "initial_density": "minao",
            "initial_density_generated_on": "cpu_before_device_conversion",
            "initial_guess_wall_time_seconds": guess_seconds,
            "scf_kernel_wall_time_seconds": scf_seconds,
            "gradient_wall_time_seconds": gradient_seconds,
            "wall_time_seconds": total_seconds,
            "nonlocal_correlation_active": bool(mf.do_nlc())
            if hasattr(mf, "do_nlc")
            else None,
            "cupy_free_memory_bytes_after_run": int(cupy.cuda.runtime.memGetInfo()[0]),
            "cupy_pool_total_bytes_after_run": int(cupy.get_default_memory_pool().total_bytes()),
            "pyscf_module_version": str(pyscf.__version__),
        }
        cycles_value = getattr(mf, "cycles", None)
        n_iterations = int(cycles_value) if isinstance(cycles_value, int) else None
        gradient_rows: list[tuple[float, float, float]] = []
        for index, row in enumerate(gradient):
            values = tuple(float(value) for value in row)
            if len(values) != 3:
                raise CalculationFailure(
                    "runtime_environment",
                    f"Gradient row {index} has {len(values)} components instead of 3.",
                )
            gradient_rows.append((values[0], values[1], values[2]))
        return CalculationOutput(
            engine_name="gpu4pyscf",
            engine_versions=runtime,
            results=Results(
                energy_hartree=energy,
                gradient_hartree_per_bohr=tuple(gradient_rows),
                converged=True,
                n_iterations=n_iterations,
                s2=s2_value,
                s2_target=s2_target,
                s2_deviation=None if s2_value is None else s2_value - s2_target,
                wall_time_seconds=total_seconds,
                scf_wall_time_seconds=guess_seconds + scf_seconds,
                gradient_wall_time_seconds=gradient_seconds,
            ),
            raw_payload=raw_payload,
        )
