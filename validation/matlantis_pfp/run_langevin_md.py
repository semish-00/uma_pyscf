#!/usr/bin/env python
"""Run reproducible neutral-molecule PFP Langevin MD inside Matlantis.

The output trajectories are candidate sources, never teacher labels. This file
is self-contained because ``pfp_api_client`` is available only in Matlantis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any

PROTOCOL_SCHEMA = "uma-pyscf-pfp-langevin-md-protocol-v1"
RUN_SCHEMA = "uma-pyscf-pfp-langevin-md-run-v1"
TRAJECTORY_SCHEMA = "uma-pyscf-pfp-langevin-md-trajectory-v1"

_PROTOCOL_FIELDS = {
    "schema",
    "model_version",
    "calc_mode",
    "max_retries",
    "required_charge",
    "required_multiplicity",
    "timestep_fs",
    "friction_per_fs",
    "steps",
    "save_interval_steps",
    "temperatures_K",
    "seeds",
    "force_temperature",
    "max_centered_radius_angstrom",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        scratch.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"{path} is not a {PROTOCOL_SCHEMA} protocol")
    unknown = set(protocol) - _PROTOCOL_FIELDS
    missing = _PROTOCOL_FIELDS - set(protocol)
    if unknown or missing:
        raise ValueError(
            f"protocol fields differ: missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    version = str(protocol["model_version"])
    if not version or version.lower() == "latest":
        raise ValueError("model_version must be pinned and cannot be 'latest'")
    if str(protocol["calc_mode"]) != "R2SCAN_PLUS_D3":
        raise ValueError("the H/Si/Ge/Cl PFP MD pilot requires R2SCAN_PLUS_D3")
    _positive_int(protocol["max_retries"], "max_retries")
    if protocol["required_charge"] != 0 or protocol["required_multiplicity"] != 1:
        raise ValueError("PFP MD candidate generation is restricted to neutral singlets")
    timestep = _positive_number(protocol["timestep_fs"], "timestep_fs")
    if timestep > 1.0:
        raise ValueError("timestep_fs must be <= 1.0 for this H-containing pilot")
    _positive_number(protocol["friction_per_fs"], "friction_per_fs")
    steps = _positive_int(protocol["steps"], "steps")
    interval = _positive_int(protocol["save_interval_steps"], "save_interval_steps")
    if steps % interval:
        raise ValueError("steps must be divisible by save_interval_steps")
    temperatures = protocol["temperatures_K"]
    if not isinstance(temperatures, list) or not temperatures:
        raise ValueError("temperatures_K must be a non-empty list")
    invalid_temperature = any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in temperatures
    )
    if invalid_temperature:
        raise ValueError("temperatures_K must contain positive integers")
    if len(temperatures) != len(set(temperatures)):
        raise ValueError("temperatures_K must be unique")
    seeds = protocol["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if any(value > 2**32 - 2 for value in seeds):
        raise ValueError("seeds must leave room for a separate seed+1 thermostat RNG")
    if not isinstance(protocol["force_temperature"], bool):
        raise ValueError("force_temperature must be boolean")
    _positive_number(
        protocol["max_centered_radius_angstrom"], "max_centered_radius_angstrom"
    )
    return protocol


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "uma-pyscf-candidate-manifest-v1"
    ):
        raise ValueError(f"{path} is not an uma-pyscf-candidate-manifest-v1 manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("candidate manifest has no records")
    seen: set[str] = set()
    for candidate in records:
        _validate_candidate(candidate)
        record_id = str(candidate["record_id"])
        if record_id in seen:
            raise ValueError(f"candidate record_id is repeated: {record_id}")
        seen.add(record_id)
    return manifest


def _validate_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    state = candidate.get("state", {})
    if state.get("charge") != 0 or state.get("multiplicity") != 1:
        raise ValueError(
            f"candidate {candidate.get('record_id')} is not a neutral singlet"
        )
    structure = candidate.get("structure", {})
    numbers = structure.get("atomic_numbers")
    positions = structure.get("positions_angstrom")
    if not isinstance(numbers, list) or not numbers or not isinstance(positions, list):
        raise ValueError(f"candidate {candidate.get('record_id')} has no valid structure")
    if len(numbers) != len(positions):
        raise ValueError(f"candidate {candidate.get('record_id')} atom rows do not match")


def _trajectory_id(record_id: str, temperature: int, seed: int) -> str:
    return f"{record_id}_t{temperature:04d}_s{seed}"


def _run_identity(manifest: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    runs = []
    for candidate in manifest["records"]:
        for temperature in protocol["temperatures_K"]:
            for seed in protocol["seeds"]:
                trajectory_id = _trajectory_id(candidate["record_id"], temperature, seed)
                runs.append(
                    {
                        "trajectory_id": trajectory_id,
                        "record_id": candidate["record_id"],
                        "parent_id": candidate["structure"].get("parent_structure_id")
                        or candidate["record_id"],
                        "temperature_K": temperature,
                        "seed": seed,
                        "path": f"trajectories/{trajectory_id}.traj",
                    }
                )
    return {
        "schema": RUN_SCHEMA,
        "sampling_id": manifest["sampling_id"],
        "candidate_manifest_sha256": _fingerprint(manifest),
        "protocol": protocol,
        "protocol_sha256": _fingerprint(protocol),
        "runs": runs,
    }


def _prepare_output(output_dir: Path, identity: dict[str, Any]) -> None:
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if _read_json(identity_path) != identity:
            raise ValueError(f"{output_dir} belongs to different inputs")
    else:
        _write_json_atomic(identity_path, identity)


def _runtime_versions() -> dict[str, Any]:
    import ase
    import pfp_api_client

    return {
        "ase": getattr(ase, "__version__", None),
        "pfp_api_client": getattr(pfp_api_client, "__version__", None),
        "python": platform.python_version(),
    }


def _run_one(
    candidate: dict[str, Any],
    run_spec: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from ase import Atoms, units
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
        ZeroRotation,
        force_temperature,
    )
    import numpy as np
    from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
    from pfp_api_client.pfp.estimator import Estimator

    structure = candidate["structure"]
    atoms = Atoms(
        numbers=structure["atomic_numbers"],
        positions=structure["positions_angstrom"],
        pbc=False,
    )
    estimator = Estimator(
        model_version=protocol["model_version"],
        calc_mode=protocol["calc_mode"],
        max_retries=int(protocol["max_retries"]),
    )
    atoms.calc = ASECalculator(estimator)
    seed = int(run_spec["seed"])
    np.random.seed(seed)
    velocity_rng = np.random.RandomState(seed)
    thermostat_rng = np.random.RandomState(seed + 1)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=float(run_spec["temperature_K"]),
        force_temp=bool(protocol["force_temperature"]),
        rng=velocity_rng,
    )
    Stationary(atoms)
    ZeroRotation(atoms)
    force_temperature(atoms, float(run_spec["temperature_K"]), unit="K")
    dynamics = Langevin(
        atoms,
        float(protocol["timestep_fs"]) * units.fs,
        temperature_K=float(run_spec["temperature_K"]),
        friction=float(protocol["friction_per_fs"]) / units.fs,
        rng=thermostat_rng,
    )
    trajectory_path = output_dir / run_spec["path"]
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = Trajectory(str(trajectory_path), "w", atoms)
    observations: list[dict[str, float]] = []

    def observe() -> None:
        positions = np.asarray(atoms.positions, dtype=float)
        if not np.isfinite(positions).all():
            raise RuntimeError("MD produced non-finite positions")
        centered = positions - positions.mean(axis=0)
        centered_radius = float(np.linalg.norm(centered, axis=1).max())
        if centered_radius > float(protocol["max_centered_radius_angstrom"]):
            raise RuntimeError(
                f"MD centered radius {centered_radius:.6g} Angstrom exceeds limit"
            )
        forces = np.asarray(atoms.get_forces(), dtype=float)
        observations.append(
            {
                "step": float(dynamics.nsteps),
                "potential_energy_ev": float(atoms.get_potential_energy()),
                "kinetic_energy_ev": float(atoms.get_kinetic_energy()),
                "temperature_K": float(atoms.get_temperature()),
                "max_force_ev_per_angstrom": float(
                    np.linalg.norm(forces, axis=1).max()
                ),
                "centered_radius_angstrom": centered_radius,
            }
        )

    trajectory.write()
    observe()
    interval = int(protocol["save_interval_steps"])
    dynamics.attach(trajectory.write, interval=interval)
    dynamics.attach(observe, interval=interval)
    try:
        dynamics.run(int(protocol["steps"]))
    finally:
        trajectory.close()
    totals = [row["potential_energy_ev"] + row["kinetic_energy_ev"] for row in observations]
    return {
        "schema": TRAJECTORY_SCHEMA,
        "status": "completed",
        "run": run_spec,
        "protocol_sha256": _fingerprint(protocol),
        "frames_written": len(observations),
        "observations": observations,
        "diagnostics": {
            "total_energy_range_ev": max(totals) - min(totals),
            "temperature_min_K": min(row["temperature_K"] for row in observations),
            "temperature_max_K": max(row["temperature_K"] for row in observations),
            "max_force_ev_per_angstrom": max(
                row["max_force_ev_per_angstrom"] for row in observations
            ),
            "max_centered_radius_angstrom": max(
                row["centered_radius_angstrom"] for row in observations
            ),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "versions": _runtime_versions(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def run(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.config)
    manifest = _load_manifest(args.manifest)
    identity = _run_identity(manifest, protocol)
    _prepare_output(args.output_dir, identity)
    if args.dry_run:
        print(
            f"dry-run parents={len(manifest['records'])} trajectories={len(identity['runs'])} "
            f"steps_each={protocol['steps']} output={args.output_dir}"
        )
        return 0

    candidates = {record["record_id"]: record for record in manifest["records"]}
    completed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for index, run_spec in enumerate(identity["runs"], start=1):
        trajectory_id = run_spec["trajectory_id"]
        summary_path = args.output_dir / "summaries" / f"{trajectory_id}.json"
        trajectory_path = args.output_dir / run_spec["path"]
        if summary_path.exists():
            summary = _read_json(summary_path)
            if (
                summary.get("schema") != TRAJECTORY_SCHEMA
                or summary.get("status") != "completed"
                or summary.get("run") != run_spec
                or not trajectory_path.is_file()
            ):
                raise ValueError(f"resume artifacts are incompatible for {trajectory_id}")
            skipped += 1
            print(f"[{index}/{len(identity['runs'])}] skip {trajectory_id}", flush=True)
            continue
        if trajectory_path.exists():
            raise ValueError(
                f"partial trajectory exists without a summary: {trajectory_path}"
            )
        print(f"[{index}/{len(identity['runs'])}] run {trajectory_id}", flush=True)
        try:
            summary = _run_one(
                candidates[run_spec["record_id"]], run_spec, protocol, args.output_dir
            )
            _write_json_atomic(summary_path, summary)
            completed += 1
        except Exception as exc:
            failures.append(
                {
                    "trajectory_id": trajectory_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"FAILED {trajectory_id}: {type(exc).__name__}: {exc}", flush=True)
            if not args.keep_going:
                break
    run_summary = {
        "schema": RUN_SCHEMA,
        "identity_sha256": _fingerprint(identity),
        "counts": {
            "trajectories": len(identity["runs"]),
            "completed_this_pass": completed,
            "skipped_this_pass": skipped,
            "failed_this_pass": len(failures),
        },
        "failures": failures,
    }
    _write_json_atomic(args.output_dir / "summary.json", run_summary)
    print(json.dumps(run_summary["counts"], sort_keys=True), flush=True)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
