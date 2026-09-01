#!/usr/bin/env python
"""Run two-stage base-UMA CI-NEB and prepare the fixed C0 path tranche."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ase import Atoms  # type: ignore[import-not-found]
from ase.calculators.singlepoint import (  # type: ignore[import-not-found]
    SinglePointCalculator,
)
from ase.io import read, write  # type: ignore[import-not-found]
from ase.mep import NEB  # type: ignore[import-not-found]
from ase.optimize import FIRE  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import yaml

from uma_pyscf.core.ids import canonical_json_fingerprint, sha256_of_file
from uma_pyscf.core.io import write_json_atomic
from uma_pyscf.inference.uma import _initialize_uma

SCHEMA_VERSION = 1
EXPECTED_REACTION_COUNT = 4
SUPPORTED_MASSES = {1: 1.008, 14: 28.085, 17: 35.45, 32: 72.630}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _sequence(value: Any, path: str) -> list[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return list(value)


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{path}.{key} is required")
    return mapping[key]


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(loaded, "config")
    version = _integer(_required(config, "schema_version", "config"), "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {SCHEMA_VERSION}; got {version}")
    for key in (
        "path_set_id",
        "endpoint_set_id",
        "reaction_ids",
        "model",
        "neb",
        "selection",
        "filters",
    ):
        _required(config, key, "config")
    reaction_ids = [str(value) for value in _sequence(config["reaction_ids"], "reaction_ids")]
    if len(reaction_ids) != EXPECTED_REACTION_COUNT or len(set(reaction_ids)) != len(reaction_ids):
        raise ValueError("config.reaction_ids must contain four unique C0 reactions")
    neb = _mapping(config["neb"], "config.neb")
    image_count = _integer(_required(neb, "image_count", "config.neb"), "neb.image_count")
    if image_count < 9 or image_count % 2 == 0:
        raise ValueError("config.neb.image_count must be an odd integer of at least nine")
    selection = _mapping(config["selection"], "config.selection")
    count = _integer(
        _required(selection, "count_per_reaction", "config.selection"),
        "selection.count_per_reaction",
    )
    if count < 5 or count > image_count:
        raise ValueError("selection.count_per_reaction must be between five and image_count")
    return config


def _endpoint_index(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    reactions = _sequence(_required(summary, "reactions", "endpoint_summary"), "reactions")
    for raw in reactions:
        reaction = _mapping(raw, "endpoint_summary.reactions[]")
        reaction_id = str(_required(reaction, "reaction_id", "endpoint reaction"))
        if reaction_id in result:
            raise ValueError(f"endpoint summary repeats reaction {reaction_id!r}")
        result[reaction_id] = reaction
    return result


def _read_endpoint(
    endpoint_run_root: Path,
    reaction: Mapping[str, Any],
    reaction_id: str,
    endpoint_name: str,
) -> Atoms:
    endpoint = _mapping(
        _required(reaction["endpoints"], endpoint_name, f"reaction {reaction_id}.endpoints"),
        f"reaction {reaction_id}.{endpoint_name}",
    )
    path = endpoint_run_root / "structures" / reaction_id / f"{endpoint_name}.xyz"
    if sha256_of_file(path) != str(endpoint["output_sha256"]):
        raise ValueError(f"endpoint checksum mismatch: {path}")
    atoms = read(path, index=0)
    atoms.set_pbc(False)
    return atoms


def _snapshot(images: list[Atoms]) -> tuple[list[Atoms], list[float], list[float]]:
    snapshots: list[Atoms] = []
    energies: list[float] = []
    max_forces: list[float] = []
    for image in images:
        energy = float(image.get_potential_energy())
        forces = np.asarray(image.get_forces(), dtype=float)
        saved = image.copy()
        saved.calc = SinglePointCalculator(saved, energy=energy, forces=forces)
        snapshots.append(saved)
        energies.append(energy)
        max_forces.append(float(np.linalg.norm(forces, axis=1).max()))
    return snapshots, energies, max_forces


def _mass_weighted_coordinates(images: list[Atoms]) -> tuple[float, ...]:
    numbers = tuple(int(value) for value in images[0].numbers)
    try:
        masses = np.asarray([SUPPORTED_MASSES[value] for value in numbers], dtype=float)
    except KeyError as exc:
        raise ValueError(f"unsupported atomic number in path: {exc.args[0]}") from exc
    total_mass = float(masses.sum())
    coordinates = [0.0]
    for index in range(1, len(images)):
        if tuple(int(value) for value in images[index].numbers) != numbers:
            raise ValueError(f"path image {index} changes atom identity or ordering")
        delta = np.asarray(images[index].positions - images[index - 1].positions, dtype=float)
        step = float(np.sqrt(np.sum(masses[:, None] * delta * delta) / total_mass))
        coordinates.append(coordinates[-1] + step)
    return tuple(coordinates)


def _selected_indices(
    images: list[Atoms], climbing_index: int, requested_count: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    coordinates = _mass_weighted_coordinates(images)
    mandatory = {0, len(images) - 1, climbing_index}
    mandatory.update(
        index for index in (climbing_index - 1, climbing_index + 1) if 0 <= index < len(images)
    )
    if len(mandatory) > requested_count:
        raise ValueError("selection count cannot retain endpoints and climbing-image neighborhood")
    selected = set(mandatory)
    while len(selected) < requested_count:
        candidates = [index for index in range(len(images)) if index not in selected]
        chosen = max(
            candidates,
            key=lambda index: (
                min(abs(coordinates[index] - coordinates[other]) for other in selected),
                -index,
            ),
        )
        selected.add(chosen)
    return tuple(sorted(selected)), coordinates


def _neb_fmax(neb: NEB) -> float:
    forces = np.asarray(neb.get_forces(), dtype=float)
    if not np.isfinite(forces).all():
        raise FloatingPointError("NEB produced a non-finite projected force")
    return float(np.linalg.norm(forces, axis=1).max())


def _attach_finite_force_guard(optimizer: FIRE, neb: NEB) -> None:
    def check() -> None:
        _neb_fmax(neb)

    optimizer.attach(check, interval=1)


def _run_reaction(
    *,
    reaction_id: str,
    reaction: Mapping[str, Any],
    endpoint_run_root: Path,
    calculator: Any,
    settings: Mapping[str, Any],
    selection: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    reactant = _read_endpoint(endpoint_run_root, reaction, reaction_id, "reactant")
    product = _read_endpoint(endpoint_run_root, reaction, reaction_id, "product")
    if reactant.numbers.tolist() != product.numbers.tolist():
        raise ValueError(f"reaction {reaction_id} changes atom identity/order")
    charge = int(reaction["charge"])
    multiplicity = int(reaction["multiplicity"])
    image_count = _integer(settings["image_count"], "neb.image_count")
    images = [reactant.copy()]
    images.extend(reactant.copy() for _ in range(image_count - 2))
    images.append(product.copy())
    calculator.reset()
    for image in images:
        image.info.update(charge=charge, spin=multiplicity)
        image.calc = calculator
    neb = NEB(
        images,
        k=_number(settings["spring_constant_ev_per_angstrom2"], "neb.spring_constant"),
        climb=False,
        method=str(settings["method"]),
        allow_shared_calculator=True,
        parallel=False,
    )
    neb.interpolate(method=str(settings["interpolation"]))
    reaction_dir = output_dir / "paths" / reaction_id
    reaction_dir.mkdir(parents=True, exist_ok=True)
    initial, initial_energies, _ = _snapshot(images)
    write(reaction_dir / "initial.traj", initial)

    coarse = FIRE(
        neb,
        logfile=str(reaction_dir / "coarse.log"),
        maxstep=_number(settings["maxstep_angstrom"], "neb.maxstep_angstrom"),
    )
    _attach_finite_force_guard(coarse, neb)
    coarse_converged = bool(
        coarse.run(
            fmax=_number(settings["coarse_fmax_ev_per_angstrom"], "neb.coarse_fmax"),
            steps=_integer(settings["coarse_steps"], "neb.coarse_steps"),
        )
    )
    coarse_images, coarse_energies, _ = _snapshot(images)
    write(reaction_dir / "coarse.traj", coarse_images)

    neb.climb = True
    final = FIRE(
        neb,
        logfile=str(reaction_dir / "final.log"),
        maxstep=_number(settings["maxstep_angstrom"], "neb.maxstep_angstrom"),
    )
    _attach_finite_force_guard(final, neb)
    final_converged = bool(
        final.run(
            fmax=_number(settings["final_fmax_ev_per_angstrom"], "neb.final_fmax"),
            steps=_integer(settings["final_steps"], "neb.final_steps"),
        )
    )
    final_neb_fmax = _neb_fmax(neb)
    final_images, energies, max_forces = _snapshot(images)
    if not np.isfinite(np.asarray(energies + max_forces, dtype=float)).all():
        raise FloatingPointError(f"reaction {reaction_id} produced non-finite path data")
    adjacent_displacements = [
        float(np.linalg.norm(after.positions - before.positions))
        for before, after in zip(final_images, final_images[1:], strict=False)
    ]
    if min(adjacent_displacements) <= 1.0e-8:
        raise RuntimeError(f"reaction {reaction_id} contains duplicate adjacent images")
    final_path = reaction_dir / "final.traj"
    write(final_path, final_images)
    climbing_index = 1 + int(np.argmax(np.asarray(energies[1:-1], dtype=float)))
    requested_count = _integer(selection["count_per_reaction"], "selection.count")
    selected_indices, arc_coordinates = _selected_indices(
        final_images, climbing_index, requested_count
    )
    selected_path = reaction_dir / "selected9.traj"
    write(selected_path, [final_images[index] for index in selected_indices])
    highest_energy = max(energies)
    target = _number(settings["final_fmax_ev_per_angstrom"], "neb.final_fmax")
    return {
        "reaction_id": reaction_id,
        "charge": charge,
        "multiplicity": multiplicity,
        "coarse_converged": coarse_converged,
        "coarse_steps": coarse.nsteps,
        "final_converged": final_converged and final_neb_fmax <= target,
        "final_steps": final.nsteps,
        "final_neb_fmax_ev_per_angstrom": final_neb_fmax,
        "climbing_image_index": climbing_index,
        "selected_image_indices": list(selected_indices),
        "selected_contains_climbing_neighbors": all(
            index in selected_indices
            for index in (climbing_index - 1, climbing_index, climbing_index + 1)
        ),
        "mass_weighted_arc_coordinates_angstrom": list(arc_coordinates),
        "initial_energies_ev": initial_energies,
        "coarse_energies_ev": coarse_energies,
        "final_energies_ev": energies,
        "final_image_max_forces_ev_per_angstrom": max_forces,
        "adjacent_cartesian_displacements_angstrom": adjacent_displacements,
        "forward_barrier_ev": highest_energy - energies[0],
        "backward_barrier_ev": highest_energy - energies[-1],
        "reaction_energy_ev": energies[-1] - energies[0],
        "final_path": {
            "path": str(final_path.relative_to(output_dir)),
            "sha256": sha256_of_file(final_path),
        },
        "selected_path": {
            "path": str(selected_path.relative_to(output_dir)),
            "sha256": sha256_of_file(selected_path),
        },
    }


def _trajectory_import_config(
    config: Mapping[str, Any], records: list[Mapping[str, Any]], output_dir: Path
) -> Path:
    count = int(config["selection"]["count_per_reaction"])
    runtime_config = {
        "schema_version": 1,
        "sampling_id": "c0_independent_reaction_paths_36_v1",
        "created": str(config["created"]),
        "description": (
            "Fixed C0-independent CI-NEB tranche: four paths with endpoints, "
            "climbing-image neighborhoods, and mass-weighted arc-length coverage."
        ),
        "state": {"charge": 0, "multiplicity": 1},
        "trajectories": [
            {
                "trajectory_id": str(record["reaction_id"]),
                "parent_id": str(record["reaction_id"]),
                "path": str(record["selected_path"]["path"]),
                "count": count,
                "frame_selection": "mass_weighted_arc_length",
            }
            for record in records
        ],
        "filters": dict(config["filters"]),
    }
    path = output_dir / "trajectory_import_config.json"
    write_json_atomic(path, runtime_config)
    return path


def run(
    config_path: Path,
    endpoint_run_root: Path,
    output_dir: Path,
    model_cache_dir: Path,
) -> None:
    config = load_config(config_path)
    endpoint_summary_path = endpoint_run_root / "summary.json"
    endpoint_summary = _mapping(
        yaml.safe_load(endpoint_summary_path.read_text(encoding="utf-8")), "endpoint_summary"
    )
    if endpoint_summary.get("endpoint_set_id") != config["endpoint_set_id"]:
        raise ValueError("endpoint summary does not match config.endpoint_set_id")
    if not bool(endpoint_summary.get("all_fragments_converged")):
        raise ValueError("endpoint summary is not fully converged")
    endpoint_reactions = _endpoint_index(endpoint_summary)
    reaction_ids = [str(value) for value in config["reaction_ids"]]
    for reaction_id in reaction_ids:
        if reaction_id not in endpoint_reactions:
            raise ValueError(f"endpoint summary does not contain {reaction_id!r}")
        if endpoint_reactions[reaction_id].get("tier") != "c0_independent":
            raise ValueError(f"reaction {reaction_id!r} is not c0_independent")

    model = _mapping(config["model"], "config.model")
    context = _initialize_uma(
        model_name=str(model["name"]),
        checkpoint_path=None,
        model_cache_dir=model_cache_dir,
        task=str(model["task"]),
        device="cuda",
        inference_settings=str(model["inference_settings"]),
        seed=int(model["seed"]),
        fairchem_core_version=str(model["fairchem_core_version"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _run_reaction(
            reaction_id=reaction_id,
            reaction=endpoint_reactions[reaction_id],
            endpoint_run_root=endpoint_run_root,
            calculator=context["calculator"],
            settings=_mapping(config["neb"], "config.neb"),
            selection=_mapping(config["selection"], "config.selection"),
            output_dir=output_dir,
        )
        for reaction_id in reaction_ids
    ]
    import_config = _trajectory_import_config(config, records, output_dir)
    all_converged = all(bool(record["final_converged"]) for record in records)
    summary = {
        "schema": "uma-pyscf-c0-ci-neb-v1",
        "path_set_id": config["path_set_id"],
        "config": {
            "path": str(config_path),
            "sha256": sha256_of_file(config_path),
            "canonical_sha256": canonical_json_fingerprint(config),
        },
        "endpoint_summary": {
            "path": str(endpoint_summary_path),
            "sha256": sha256_of_file(endpoint_summary_path),
        },
        "model": model,
        "runtime": {
            "ase": str(context["ase"].__version__),
            "fairchem_core": context["installed_fairchem"],
            "inference_settings": str(model["inference_settings"]),
            "calculator_sharing": "one local UMA predictor; serial ASE NEB evaluation",
        },
        "all_paths_converged": all_converged,
        "reaction_count": len(records),
        "selected_candidate_count": sum(
            len(record["selected_image_indices"]) for record in records
        ),
        "reactions": records,
        "trajectory_import_config": {
            "path": str(import_config),
            "sha256": sha256_of_file(import_config),
        },
    }
    write_json_atomic(output_dir / "summary.json", summary)
    if not all_converged:
        raise RuntimeError("At least one CI-NEB path did not reach the final fmax target")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(
        arguments.config,
        arguments.endpoint_run_root,
        arguments.output_dir,
        arguments.model_cache_dir,
    )
