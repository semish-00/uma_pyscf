#!/usr/bin/env python
"""Run reproducible neutral-singlet PFP Langevin MD in Matlantis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

PROTOCOL_SCHEMA = "uma-pyscf-pfp-langevin-md-protocol-v1"
RUN_SCHEMA = "uma-pyscf-pfp-langevin-md-run-v1"
TRAJECTORY_SCHEMA = "uma-pyscf-pfp-langevin-md-trajectory-v1"
_FIELDS = {
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


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        scratch.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"{path} is not a {PROTOCOL_SCHEMA} protocol")
    if set(protocol) != _FIELDS:
        raise ValueError("protocol fields do not match the v1 schema")
    if protocol["model_version"] in ("", "latest"):
        raise ValueError("model_version must be pinned")
    if protocol["calc_mode"] != "R2SCAN_PLUS_D3":
        raise ValueError("this H/Si/Ge/Cl pilot requires R2SCAN_PLUS_D3")
    if protocol["required_charge"] != 0 or protocol["required_multiplicity"] != 1:
        raise ValueError("PFP MD is restricted to neutral singlets")
    if _positive(protocol["timestep_fs"], "timestep_fs") > 1.0:
        raise ValueError("timestep_fs must be <= 1.0")
    _positive(protocol["friction_per_fs"], "friction_per_fs")
    _positive(protocol["max_centered_radius_angstrom"], "max_centered_radius_angstrom")
    for name in ("max_retries", "steps", "save_interval_steps"):
        value = protocol[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if protocol["steps"] % protocol["save_interval_steps"]:
        raise ValueError("steps must be divisible by save_interval_steps")
    for name, minimum, maximum in (
        ("temperatures_K", 1, None),
        ("seeds", 0, 2**32 - 2),
    ):
        values = protocol[name]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be a non-empty unique integer list")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or (maximum is not None and value > maximum)
            for value in values
        ):
            raise ValueError(f"{name} contains an invalid value")
    if not isinstance(protocol["force_temperature"], bool):
        raise ValueError("force_temperature must be boolean")
    return protocol


def _validate_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    state = candidate.get("state", {})
    if state.get("charge") != 0 or state.get("multiplicity") != 1:
        raise ValueError(f"candidate {candidate.get('record_id')} is not a neutral singlet")
    structure = candidate.get("structure", {})
    numbers = structure.get("atomic_numbers")
    positions = structure.get("positions_angstrom")
    if not isinstance(numbers, list) or not numbers or not isinstance(positions, list):
        raise ValueError(f"candidate {candidate.get('record_id')} has no structure")
    if len(numbers) != len(positions):
        raise ValueError(f"candidate {candidate.get('record_id')} atom rows do not match")


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "uma-pyscf-candidate-manifest-v1"
        or not isinstance(manifest.get("records"), list)
        or not manifest["records"]
    ):
        raise ValueError(f"{path} is not a non-empty candidate manifest")
    for candidate in manifest["records"]:
        _validate_candidate(candidate)
    return manifest


def _run_identity(manifest: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    runs = []
    for candidate in manifest["records"]:
        record_id = candidate["record_id"]
        parent_id = candidate["structure"].get("parent_structure_id") or record_id
        for temperature in protocol["temperatures_K"]:
            for seed in protocol["seeds"]:
                trajectory_id = f"{record_id}_t{temperature:04d}_s{seed}"
                runs.append(
                    {
                        "trajectory_id": trajectory_id,
                        "record_id": record_id,
                        "parent_id": parent_id,
                        "temperature_K": temperature,
                        "seed": seed,
                        "path": f"trajectories/{trajectory_id}.traj",
                    }
                )
    return {
        "schema": RUN_SCHEMA,
        "sampling_id": manifest["sampling_id"],
        "manifest_sha256": _fingerprint(manifest),
        "protocol": protocol,
        "protocol_sha256": _fingerprint(protocol),
        "runs": runs,
    }


def _run_one(
    candidate: dict[str, Any],
    spec: dict[str, Any],
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
    atoms.calc = ASECalculator(
        Estimator(
            model_version=protocol["model_version"],
            calc_mode=protocol["calc_mode"],
            max_retries=protocol["max_retries"],
        )
    )
    seed = spec["seed"]
    np.random.seed(seed)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=spec["temperature_K"],
        force_temp=protocol["force_temperature"],
        rng=np.random.RandomState(seed),
    )
    Stationary(atoms)
    ZeroRotation(atoms)
    force_temperature(atoms, spec["temperature_K"], unit="K")
    dynamics = Langevin(
        atoms,
        protocol["timestep_fs"] * units.fs,
        temperature_K=spec["temperature_K"],
        friction=protocol["friction_per_fs"] / units.fs,
        rng=np.random.RandomState(seed + 1),
    )
    path = output_dir / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = Trajectory(str(path), "w", atoms)
    samples: list[tuple[float, float, float]] = []

    def observe() -> None:
        positions = np.asarray(atoms.positions, dtype=float)
        if not np.isfinite(positions).all():
            raise RuntimeError("MD produced non-finite positions")
        radius = float(np.linalg.norm(positions - positions.mean(axis=0), axis=1).max())
        if radius > protocol["max_centered_radius_angstrom"]:
            raise RuntimeError(f"MD centered radius {radius:.6g} Angstrom exceeds limit")
        forces = np.asarray(atoms.get_forces(), dtype=float)
        total_energy = float(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        max_force = float(np.linalg.norm(forces, axis=1).max())
        samples.append((total_energy, float(atoms.get_temperature()), max_force))

    trajectory.write()
    observe()
    interval = protocol["save_interval_steps"]
    dynamics.attach(trajectory.write, interval=interval)
    dynamics.attach(observe, interval=interval)
    try:
        dynamics.run(protocol["steps"])
    finally:
        trajectory.close()
    energies, temperatures, max_forces = zip(*samples, strict=True)
    return {
        "schema": TRAJECTORY_SCHEMA,
        "run": spec,
        "protocol_sha256": _fingerprint(protocol),
        "frames": len(samples),
        "diagnostics": {
            "total_energy_range_ev": max(energies) - min(energies),
            "temperature_range_K": [min(temperatures), max(temperatures)],
            "max_force_ev_per_angstrom": max(max_forces),
        },
    }


def run(args: argparse.Namespace) -> int:
    protocol = _load_protocol(args.config)
    manifest = _load_manifest(args.manifest)
    identity = _run_identity(manifest, protocol)
    identity_path = args.output_dir / "run_identity.json"
    if identity_path.exists() and _read_json(identity_path) != identity:
        raise ValueError(f"{args.output_dir} belongs to different inputs")
    if not identity_path.exists():
        _write_json(identity_path, identity)
    if args.dry_run:
        print(
            f"dry-run parents={len(manifest['records'])} trajectories={len(identity['runs'])} "
            f"steps_each={protocol['steps']} output={args.output_dir}"
        )
        return 0

    candidates = {record["record_id"]: record for record in manifest["records"]}
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    failures = []
    for spec in identity["runs"]:
        summary_path = args.output_dir / "summaries" / f"{spec['trajectory_id']}.json"
        trajectory_path = args.output_dir / spec["path"]
        if summary_path.exists() and trajectory_path.exists():
            counts["skipped"] += 1
            continue
        if summary_path.exists() or trajectory_path.exists():
            raise ValueError(f"partial output exists for {spec['trajectory_id']}")
        try:
            summary = _run_one(candidates[spec["record_id"]], spec, protocol, args.output_dir)
            _write_json(summary_path, summary)
            counts["completed"] += 1
        except Exception as exc:
            counts["failed"] += 1
            failures.append({"trajectory_id": spec["trajectory_id"], "error": str(exc)})
            if not args.keep_going:
                break
    _write_json(args.output_dir / "summary.json", {"counts": counts, "failures": failures})
    print(json.dumps(counts, sort_keys=True))
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
