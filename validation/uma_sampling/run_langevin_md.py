#!/usr/bin/env python
"""Generate a score-blind C0 pool with short base-UMA Langevin trajectories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import yaml

from uma_pyscf.core.ids import canonical_json_fingerprint, sha256_of_file
from uma_pyscf.core.io import read_json, write_json_atomic
from uma_pyscf.inference.uma import _initialize_uma
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord

if TYPE_CHECKING:
    from ase import Atoms

SCHEMA_VERSION = 1
EXPECTED_PARENT_COUNT = 6


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _sequence(value: Any, path: str) -> list[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return list(value)


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be finite and positive")
    return result


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(loaded, "config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {SCHEMA_VERSION}")
    expected = {
        "schema_version",
        "md_set_id",
        "created",
        "description",
        "model",
        "optimization",
        "dynamics",
        "selection",
        "filters",
    }
    if set(config) != expected:
        raise ValueError("config fields do not match the v1 schema")

    model = _mapping(config["model"], "config.model")
    if model.get("name") != "uma-s-1p2" or model.get("task") != "omol":
        raise ValueError("C0 moderate MD requires uma-s-1p2 with the omol task")
    if model.get("fairchem_core_version") != "2.22.0":
        raise ValueError("fairchem-core must be pinned to 2.22.0")
    _positive_integer(model.get("seed"), "config.model.seed")

    optimization = _mapping(config["optimization"], "config.optimization")
    for key in ("coarse_fmax_ev_per_angstrom", "final_fmax_ev_per_angstrom", "maxstep_angstrom"):
        _positive_number(optimization.get(key), f"config.optimization.{key}")
    for key in ("coarse_steps", "final_steps"):
        _positive_integer(optimization.get(key), f"config.optimization.{key}")

    dynamics = _mapping(config["dynamics"], "config.dynamics")
    timestep = _positive_number(dynamics.get("timestep_fs"), "config.dynamics.timestep_fs")
    if timestep > 1.0:
        raise ValueError("config.dynamics.timestep_fs must be <= 1.0 for H-containing systems")
    _positive_number(dynamics.get("friction_per_fs"), "config.dynamics.friction_per_fs")
    _positive_number(
        dynamics.get("max_centered_radius_angstrom"),
        "config.dynamics.max_centered_radius_angstrom",
    )
    interval = _positive_integer(
        dynamics.get("save_interval_steps"), "config.dynamics.save_interval_steps"
    )
    for key in ("equilibration_steps", "production_steps"):
        steps = _positive_integer(dynamics.get(key), f"config.dynamics.{key}")
        if steps % interval:
            raise ValueError(f"config.dynamics.{key} must be divisible by save_interval_steps")
    temperatures = _sequence(dynamics.get("temperatures_K"), "config.dynamics.temperatures_K")
    seeds = _sequence(dynamics.get("seeds"), "config.dynamics.seeds")
    for name, values in (("temperatures_K", temperatures), ("seeds", seeds)):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"config.dynamics.{name} must be a non-empty unique list")
        for value in values:
            _positive_integer(value, f"config.dynamics.{name}[]")
    if not isinstance(dynamics.get("force_temperature"), bool):
        raise ValueError("config.dynamics.force_temperature must be boolean")
    temperature_bounds = _sequence(
        dynamics.get("temperature_mean_ratio_bounds"),
        "config.dynamics.temperature_mean_ratio_bounds",
    )
    if len(temperature_bounds) != 2:
        raise ValueError("temperature_mean_ratio_bounds must contain lower and upper bounds")
    lower = _positive_number(temperature_bounds[0], "temperature_mean_ratio_bounds[0]")
    upper = _positive_number(temperature_bounds[1], "temperature_mean_ratio_bounds[1]")
    if not lower < 1.0 < upper:
        raise ValueError("temperature_mean_ratio_bounds must bracket 1.0")

    selection = _mapping(config["selection"], "config.selection")
    frames = _positive_integer(
        selection.get("frames_per_trajectory"), "config.selection.frames_per_trajectory"
    )
    minimum_completed = _positive_integer(
        selection.get("minimum_completed_trajectories"),
        "config.selection.minimum_completed_trajectories",
    )
    available = int(dynamics["production_steps"]) // interval + 1
    if frames > available:
        raise ValueError("frames_per_trajectory exceeds the saved production frames")
    planned = EXPECTED_PARENT_COUNT * len(temperatures) * len(seeds)
    if minimum_completed > planned:
        raise ValueError("minimum_completed_trajectories exceeds the planned trajectories")
    return config


def _load_manifest(path: Path) -> CandidateManifest:
    manifest = CandidateManifest.from_dict(read_json(path))
    if len(manifest.records) != EXPECTED_PARENT_COUNT:
        raise ValueError(f"C0 moderate MD requires {EXPECTED_PARENT_COUNT} parents")
    parent_ids: set[str] = set()
    for record in manifest.records:
        if record.state.charge != 0 or record.state.multiplicity != 1:
            raise ValueError(f"{record.record_id} is not a neutral singlet")
        parent_id = record.structure.parent_structure_id
        if parent_id is None or parent_id in parent_ids:
            raise ValueError(f"{record.record_id} has a missing or repeated parent id")
        parent_ids.add(parent_id)
    return manifest


def _run_identity(
    config: dict[str, Any], manifest: CandidateManifest, config_path: Path, manifest_path: Path
) -> dict[str, Any]:
    dynamics = config["dynamics"]
    runs = []
    for record in manifest.records:
        parent_id = str(record.structure.parent_structure_id)
        for temperature in dynamics["temperatures_K"]:
            for seed in dynamics["seeds"]:
                trajectory_id = f"{parent_id}_t{int(temperature):04d}_s{int(seed)}"
                runs.append(
                    {
                        "trajectory_id": trajectory_id,
                        "record_id": record.record_id,
                        "parent_id": parent_id,
                        "temperature_K": int(temperature),
                        "seed": int(seed),
                        "path": f"trajectories/{trajectory_id}.traj",
                    }
                )
    return {
        "schema": "uma-pyscf-base-uma-langevin-md-run-v1",
        "md_set_id": config["md_set_id"],
        "config": {
            "path": str(config_path),
            "sha256": sha256_of_file(config_path),
            "canonical_sha256": canonical_json_fingerprint(config),
        },
        "parent_manifest": {
            "id": manifest.sampling_id,
            "path": str(manifest_path),
            "sha256": sha256_of_file(manifest_path),
        },
        "runs": runs,
    }


def _atoms_from_candidate(record: CandidateRecord) -> Atoms:
    from ase import Atoms  # type: ignore[import-not-found]

    atoms = Atoms(
        numbers=record.structure.atomic_numbers,
        positions=record.structure.positions_angstrom,
        pbc=False,
    )
    atoms.info.update(charge=record.state.charge, spin=record.state.multiplicity)
    return atoms


def _max_force(atoms: Atoms) -> float:
    import numpy as np  # type: ignore[import-not-found]

    forces = np.asarray(atoms.get_forces(), dtype=float)
    if not np.isfinite(forces).all():
        raise RuntimeError("base UMA produced non-finite forces")
    return float(np.linalg.norm(forces, axis=1).max())


def _relax_parent(
    record: CandidateRecord,
    calculator: Any,
    settings: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Atoms, dict[str, Any]]:
    from ase.io import read, write  # type: ignore[import-not-found]
    from ase.optimize import BFGS, FIRE  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    parent_id = str(record.structure.parent_structure_id)
    structure_path = output_dir / "relaxed" / f"{parent_id}.xyz"
    summary_path = output_dir / "relaxation_summaries" / f"{parent_id}.json"
    if structure_path.exists() or summary_path.exists():
        if not structure_path.exists() or not summary_path.exists():
            raise ValueError(f"partial relaxation output exists for {parent_id}")
        summary = _mapping(read_json(summary_path), f"relaxation summary {parent_id}")
        if not summary.get("converged"):
            raise ValueError(f"saved relaxation for {parent_id} is not converged")
        if sha256_of_file(structure_path) != summary.get("output_sha256"):
            raise ValueError(f"saved relaxation checksum mismatch for {parent_id}")
        atoms = read(structure_path, index=0)
        atoms.info.update(charge=0, spin=1)
        return atoms, summary

    structure_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atoms = _atoms_from_candidate(record)
    calculator.reset()
    atoms.calc = calculator
    initial_energy = float(atoms.get_potential_energy())
    initial_force = _max_force(atoms)
    initial_positions = np.asarray(atoms.positions, dtype=float).copy()

    coarse = FIRE(
        atoms,
        maxstep=float(settings["maxstep_angstrom"]),
        logfile=str(output_dir / "relaxation_summaries" / f"{parent_id}_coarse.log"),
    )
    coarse_converged = bool(
        coarse.run(
            fmax=float(settings["coarse_fmax_ev_per_angstrom"]),
            steps=int(settings["coarse_steps"]),
        )
    )
    final = BFGS(
        atoms,
        maxstep=float(settings["maxstep_angstrom"]),
        trajectory=str(output_dir / "relaxation_summaries" / f"{parent_id}_final.traj"),
        logfile=str(output_dir / "relaxation_summaries" / f"{parent_id}_final.log"),
    )
    final_converged = bool(
        final.run(
            fmax=float(settings["final_fmax_ev_per_angstrom"]),
            steps=int(settings["final_steps"]),
        )
    )
    final_force = _max_force(atoms)
    write(structure_path, atoms, format="xyz")
    summary = {
        "record_id": record.record_id,
        "parent_id": parent_id,
        "coarse_converged": coarse_converged,
        "converged": final_converged,
        "coarse_steps": coarse.nsteps,
        "final_steps": final.nsteps,
        "initial_energy_ev": initial_energy,
        "final_energy_ev": float(atoms.get_potential_energy()),
        "initial_max_force_ev_per_angstrom": initial_force,
        "final_max_force_ev_per_angstrom": final_force,
        "max_displacement_angstrom": float(
            np.linalg.norm(np.asarray(atoms.positions) - initial_positions, axis=1).max()
        ),
        "path": structure_path.relative_to(output_dir).as_posix(),
        "output_sha256": sha256_of_file(structure_path),
    }
    write_json_atomic(summary_path, summary)
    if not final_converged:
        raise RuntimeError(f"base-UMA relaxation did not converge for {parent_id}")
    atoms.info.update(charge=0, spin=1)
    return atoms, summary


def _observe(atoms: Atoms, radius_limit: float) -> tuple[float, float, float, float]:
    import numpy as np  # type: ignore[import-not-found]

    positions = np.asarray(atoms.positions, dtype=float)
    if not np.isfinite(positions).all():
        raise RuntimeError("MD produced non-finite positions")
    radius = float(np.linalg.norm(positions - positions.mean(axis=0), axis=1).max())
    if radius > radius_limit:
        raise RuntimeError(f"MD centered radius {radius:.6g} Angstrom exceeds limit")
    force = _max_force(atoms)
    potential = float(atoms.get_potential_energy())
    temperature = float(atoms.get_temperature())
    if not all(math.isfinite(value) for value in (potential, temperature, force)):
        raise RuntimeError("MD produced non-finite diagnostics")
    return potential, temperature, force, radius


def _run_trajectory(
    start: Atoms,
    calculator: Any,
    spec: Mapping[str, Any],
    settings: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from ase import units  # type: ignore[import-not-found]
    from ase.io.trajectory import Trajectory  # type: ignore[import-not-found]
    from ase.md.langevin import Langevin  # type: ignore[import-not-found]
    from ase.md.velocitydistribution import (  # type: ignore[import-not-found]
        MaxwellBoltzmannDistribution,
        Stationary,
        ZeroRotation,
        force_temperature,
    )
    import numpy as np  # type: ignore[import-not-found]

    trajectory_id = str(spec["trajectory_id"])
    trajectory_path = output_dir / str(spec["path"])
    summary_path = output_dir / "trajectory_summaries" / f"{trajectory_id}.json"
    if trajectory_path.exists() or summary_path.exists():
        if not trajectory_path.exists() or not summary_path.exists():
            raise ValueError(f"partial trajectory output exists for {trajectory_id}")
        summary = _mapping(read_json(summary_path), f"trajectory summary {trajectory_id}")
        if sha256_of_file(trajectory_path) != summary.get("output_sha256"):
            raise ValueError(f"saved trajectory checksum mismatch for {trajectory_id}")
        return summary

    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atoms = start.copy()
    atoms.info.update(charge=0, spin=1)
    calculator.reset()
    atoms.calc = calculator
    seed = int(spec["seed"])
    temperature = int(spec["temperature_K"])
    velocity_rng = np.random.RandomState(seed)
    thermostat_rng = np.random.RandomState(seed + 1)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature,
        force_temp=bool(settings["force_temperature"]),
        rng=velocity_rng,
    )
    Stationary(atoms)
    ZeroRotation(atoms)
    if bool(settings["force_temperature"]):
        force_temperature(atoms, temperature, unit="K")
    dynamics = Langevin(
        atoms,
        float(settings["timestep_fs"]) * units.fs,
        temperature_K=temperature,
        friction=float(settings["friction_per_fs"]) / units.fs,
        rng=thermostat_rng,
    )
    interval = int(settings["save_interval_steps"])
    radius_limit = float(settings["max_centered_radius_angstrom"])
    for _ in range(int(settings["equilibration_steps"]) // interval):
        dynamics.run(interval)
        _observe(atoms, radius_limit)

    temporary = trajectory_path.with_name(f".{trajectory_path.name}.tmp-{os.getpid()}")
    samples: list[tuple[float, float, float, float]] = []
    trajectory = Trajectory(str(temporary), "w", atoms)
    try:
        trajectory.write()
        samples.append(_observe(atoms, radius_limit))
        for _ in range(int(settings["production_steps"]) // interval):
            dynamics.run(interval)
            trajectory.write()
            samples.append(_observe(atoms, radius_limit))
    except Exception:
        trajectory.close()
        temporary.unlink(missing_ok=True)
        raise
    trajectory.close()
    potentials, temperatures, max_forces, radii = zip(*samples, strict=True)
    mean_temperature = sum(temperatures) / len(temperatures)
    ratio = mean_temperature / temperature
    lower, upper = (float(value) for value in settings["temperature_mean_ratio_bounds"])
    if not lower <= ratio <= upper:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"MD mean temperature ratio {ratio:.6g} is outside [{lower}, {upper}]")
    os.replace(temporary, trajectory_path)

    summary = {
        "schema": "uma-pyscf-base-uma-langevin-md-trajectory-v1",
        "run": dict(spec),
        "frames": len(samples),
        "equilibration_steps": int(settings["equilibration_steps"]),
        "production_steps": int(settings["production_steps"]),
        "diagnostics": {
            "potential_energy_range_ev": max(potentials) - min(potentials),
            "temperature_mean_K": mean_temperature,
            "temperature_mean_ratio": ratio,
            "temperature_range_K": [min(temperatures), max(temperatures)],
            "max_force_ev_per_angstrom": max(max_forces),
            "max_centered_radius_angstrom": max(radii),
        },
        "output_sha256": sha256_of_file(trajectory_path),
    }
    write_json_atomic(summary_path, summary)
    return summary


def _write_import_config(
    config: Mapping[str, Any], trajectory_summaries: Sequence[Mapping[str, Any]], path: Path
) -> None:
    frames = int(config["selection"]["frames_per_trajectory"])
    artifact = {
        "schema_version": 1,
        "sampling_id": "c0_moderate_md_pool_v1",
        "created": str(config["created"]),
        "description": (
            "Mass-weighted thinning of short base-UMA finite-temperature trajectories; "
            "selection is independent of model score and teacher labels."
        ),
        "state": {"charge": 0, "multiplicity": 1},
        "trajectories": [
            {
                "trajectory_id": run["trajectory_id"],
                "parent_id": run["parent_id"],
                "path": run["path"],
                "count": frames,
                "frame_selection": "mass_weighted_arc_length",
            }
            for summary in trajectory_summaries
            for run in [_mapping(summary["run"], "trajectory summary run")]
        ],
        "filters": dict(config["filters"]),
    }
    write_json_atomic(path, artifact)


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = _load_manifest(args.manifest)
    identity = _run_identity(config, manifest, args.config, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_dir / "run_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError(f"{args.output_dir} belongs to different inputs")
    if not identity_path.exists():
        write_json_atomic(identity_path, identity)
    if args.dry_run:
        dynamics = config["dynamics"]
        steps_each = dynamics["equilibration_steps"] + dynamics["production_steps"]
        print(
            f"dry-run parents={len(manifest.records)} trajectories={len(identity['runs'])} "
            f"steps_each={steps_each}"
        )
        return 0

    model = config["model"]
    context = _initialize_uma(
        model_name=str(model["name"]),
        checkpoint_path=None,
        model_cache_dir=args.model_cache_dir,
        task=str(model["task"]),
        device="cuda",
        inference_settings=str(model["inference_settings"]),
        seed=int(model["seed"]),
        fairchem_core_version=str(model["fairchem_core_version"]),
    )
    calculator = context["calculator"]
    relaxed: dict[str, Atoms] = {}
    relaxation_summaries = []
    for record in manifest.records:
        atoms, summary = _relax_parent(record, calculator, config["optimization"], args.output_dir)
        relaxed[str(record.structure.parent_structure_id)] = atoms
        relaxation_summaries.append(summary)
    write_json_atomic(
        args.output_dir / "relaxation_summary.json",
        {
            "schema": "uma-pyscf-base-uma-md-parent-relaxation-v1",
            "all_converged": all(bool(item["converged"]) for item in relaxation_summaries),
            "records": relaxation_summaries,
        },
    )

    counts = {"completed": 0, "skipped": 0, "failed": 0}
    failures = []
    trajectory_summaries = []
    for spec in identity["runs"]:
        summary_path = args.output_dir / "trajectory_summaries" / f"{spec['trajectory_id']}.json"
        was_complete = summary_path.exists() and (args.output_dir / spec["path"]).exists()
        try:
            summary = _run_trajectory(
                relaxed[spec["parent_id"]],
                calculator,
                spec,
                config["dynamics"],
                args.output_dir,
            )
            trajectory_summaries.append(summary)
            counts["skipped" if was_complete else "completed"] += 1
            print(f"completed {spec['trajectory_id']}", flush=True)
        except Exception as exc:
            counts["failed"] += 1
            failures.append({"trajectory_id": spec["trajectory_id"], "error": str(exc)})
            if not args.keep_going:
                break

    completed_trajectories = counts["completed"] + counts["skipped"]
    minimum_completed = int(config["selection"]["minimum_completed_trajectories"])
    ready_for_import = completed_trajectories >= minimum_completed
    summary = {
        "schema": "uma-pyscf-base-uma-langevin-md-summary-v1",
        "md_set_id": config["md_set_id"],
        "counts": counts,
        "minimum_completed_trajectories": minimum_completed,
        "ready_for_import": ready_for_import,
        "failures": failures,
        "runtime": {
            "ase": str(context["ase"].__version__),
            "fairchem_core": context["installed_fairchem"],
            "calculator_sharing": "one local UMA predictor; reset before each parent/run",
        },
        "trajectory_summaries": trajectory_summaries,
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    if not ready_for_import:
        return 1
    _write_import_config(
        config,
        trajectory_summaries,
        args.output_dir / "trajectory_import_config.json",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
