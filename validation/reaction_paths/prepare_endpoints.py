#!/usr/bin/env python
"""Validate atom mapping, relax endpoint fragments with base UMA, and prepare C0-S."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ase import Atoms  # type: ignore[import-not-found]
from ase.io import read, write  # type: ignore[import-not-found]
from ase.optimize import BFGS, FIRE  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import yaml

from uma_pyscf.core.ids import canonical_json_fingerprint, sha256_of_file
from uma_pyscf.core.io import write_json_atomic
from uma_pyscf.core.spin import electron_count, validate_electron_spin_parity
from uma_pyscf.inference.uma import _initialize_uma

SCHEMA_VERSION = 1
ENDPOINT_NAMES = ("reactant", "product")


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
    version = _integer(_required(config, "schema_version", "config"), "config.schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {SCHEMA_VERSION}; got {version}")
    for key in ("endpoint_set_id", "model", "optimization", "reactions", "state_audit"):
        _required(config, key, "config")
    reactions = _sequence(config["reactions"], "config.reactions")
    if len(reactions) != 8:
        raise ValueError(
            f"config.reactions must contain the fixed eight families; got {len(reactions)}"
        )
    return config


def _read_xyz(path: Path) -> Atoms:
    if not path.is_file():
        raise FileNotFoundError(path)
    atoms = read(path, index=0)
    atoms.set_pbc(False)
    return atoms


def _state(mapping: Mapping[str, Any], path: str) -> tuple[int, int]:
    charge = _integer(_required(mapping, "charge", path), f"{path}.charge")
    multiplicity = _integer(_required(mapping, "multiplicity", path), f"{path}.multiplicity")
    if multiplicity < 1:
        raise ValueError(f"{path}.multiplicity must be positive")
    return charge, multiplicity


def _validate_fragments(
    atoms: Atoms, endpoint: Mapping[str, Any], path: str
) -> list[dict[str, Any]]:
    fragments = [
        _mapping(value, f"{path}.fragments[{index}]")
        for index, value in enumerate(
            _sequence(_required(endpoint, "fragments", path), f"{path}.fragments")
        )
    ]
    seen: list[int] = []
    ids: set[str] = set()
    for index, fragment in enumerate(fragments):
        fragment_path = f"{path}.fragments[{index}]"
        fragment_id = str(_required(fragment, "fragment_id", fragment_path))
        if fragment_id in ids:
            raise ValueError(f"{fragment_path}.fragment_id {fragment_id!r} is repeated")
        ids.add(fragment_id)
        indices = [
            _integer(value, f"{fragment_path}.atom_indices[]")
            for value in _sequence(
                _required(fragment, "atom_indices", fragment_path),
                f"{fragment_path}.atom_indices",
            )
        ]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"{fragment_path}.atom_indices must be non-empty and unique")
        if min(indices) < 0 or max(indices) >= len(atoms):
            raise ValueError(f"{fragment_path}.atom_indices is outside 0..{len(atoms) - 1}")
        charge, multiplicity = _state(fragment, fragment_path)
        numbers = tuple(int(atoms.numbers[value]) for value in indices)
        validate_electron_spin_parity(electron_count(numbers, charge), multiplicity)
        seen.extend(indices)
        fragment["fragment_id"] = fragment_id
        fragment["atom_indices"] = indices
        fragment["charge"] = charge
        fragment["multiplicity"] = multiplicity
    if sorted(seen) != list(range(len(atoms))):
        raise ValueError(f"{path}.fragments must partition every atom exactly once")
    return fragments


def validate_reactions(config: Mapping[str, Any], config_path: Path) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(_sequence(config["reactions"], "config.reactions")):
        reaction = _mapping(raw, f"config.reactions[{index}]")
        reaction_id = str(_required(reaction, "reaction_id", f"config.reactions[{index}]"))
        if reaction_id in ids:
            raise ValueError(f"reaction_id {reaction_id!r} is repeated")
        ids.add(reaction_id)
        total_charge, total_multiplicity = _state(reaction, f"reaction {reaction_id}")
        endpoints: dict[str, dict[str, Any]] = {}
        endpoint_atoms: dict[str, Atoms] = {}
        for endpoint_name in ENDPOINT_NAMES:
            endpoint = _mapping(
                _required(reaction, endpoint_name, f"reaction {reaction_id}"),
                f"reaction {reaction_id}.{endpoint_name}",
            )
            source = (
                config_path.parent / str(_required(endpoint, "xyz_path", endpoint_name))
            ).resolve()
            atoms = _read_xyz(source)
            fragments = _validate_fragments(
                atoms, endpoint, f"reaction {reaction_id}.{endpoint_name}"
            )
            if sum(fragment["charge"] for fragment in fragments) != total_charge:
                raise ValueError(
                    f"reaction {reaction_id}.{endpoint_name} fragment charges do not sum "
                    f"to total charge {total_charge}"
                )
            endpoints[endpoint_name] = {
                "xyz_path": str(source),
                "source_sha256": sha256_of_file(source),
                "fragments": fragments,
            }
            endpoint_atoms[endpoint_name] = atoms
        reactant_numbers = endpoint_atoms["reactant"].numbers.tolist()
        product_numbers = endpoint_atoms["product"].numbers.tolist()
        if reactant_numbers != product_numbers:
            raise ValueError(
                f"reaction {reaction_id} changes atom identity/order between endpoints"
            )
        numbers = tuple(int(value) for value in endpoint_atoms["reactant"].numbers)
        validate_electron_spin_parity(electron_count(numbers, total_charge), total_multiplicity)
        validated.append(
            {
                "reaction_id": reaction_id,
                "tier": str(_required(reaction, "tier", f"reaction {reaction_id}")),
                "charge": total_charge,
                "multiplicity": total_multiplicity,
                "endpoints": endpoints,
                "atoms": endpoint_atoms,
            }
        )
    return validated


def _max_force(atoms: Atoms) -> float:
    return float(np.linalg.norm(np.asarray(atoms.get_forces(), dtype=float), axis=1).max())


def _optimize_fragment(
    atoms: Atoms,
    *,
    charge: int,
    multiplicity: int,
    calculator: Any,
    output_root: Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    atoms.info.update(charge=charge, spin=multiplicity)
    calculator.reset()
    atoms.calc = calculator
    initial_energy = float(atoms.get_potential_energy())
    initial_max_force = _max_force(atoms)
    coarse = FIRE(
        atoms,
        trajectory=str(output_root.with_suffix(".coarse.traj")),
        logfile=str(output_root.with_suffix(".coarse.log")),
        maxstep=_number(settings["maxstep_angstrom"], "optimization.maxstep_angstrom"),
    )
    coarse_converged = bool(
        coarse.run(
            fmax=_number(
                settings["coarse_fmax_ev_per_angstrom"],
                "optimization.coarse_fmax_ev_per_angstrom",
            ),
            steps=_integer(settings["coarse_steps"], "optimization.coarse_steps"),
        )
    )
    final = BFGS(
        atoms,
        trajectory=str(output_root.with_suffix(".final.traj")),
        logfile=str(output_root.with_suffix(".final.log")),
        maxstep=_number(settings["maxstep_angstrom"], "optimization.maxstep_angstrom"),
    )
    final_converged = bool(
        final.run(
            fmax=_number(
                settings["final_fmax_ev_per_angstrom"],
                "optimization.final_fmax_ev_per_angstrom",
            ),
            steps=_integer(settings["final_steps"], "optimization.final_steps"),
        )
    )
    return {
        "atoms": atoms,
        "summary": {
            "charge": charge,
            "multiplicity": multiplicity,
            "coarse_converged": coarse_converged,
            "coarse_steps": coarse.nsteps,
            "final_converged": final_converged,
            "final_steps": final.nsteps,
            "initial_energy_ev": initial_energy,
            "final_energy_ev": float(atoms.get_potential_energy()),
            "initial_max_force_ev_per_angstrom": initial_max_force,
            "final_max_force_ev_per_angstrom": _max_force(atoms),
        },
    }


def _optimize_endpoint(
    reaction: Mapping[str, Any],
    endpoint_name: str,
    calculator: Any,
    output_dir: Path,
    settings: Mapping[str, Any],
) -> tuple[Atoms, list[dict[str, Any]]]:
    endpoint = reaction["endpoints"][endpoint_name]
    seed = reaction["atoms"][endpoint_name]
    assembled = seed.copy()
    fragment_records: list[dict[str, Any]] = []
    for fragment in endpoint["fragments"]:
        indices = fragment["atom_indices"]
        fragment_atoms = seed[indices]
        seed_center = np.asarray(fragment_atoms.positions, dtype=float).mean(axis=0)
        log_root = (
            output_dir
            / "logs"
            / (f"{reaction['reaction_id']}_{endpoint_name}_{fragment['fragment_id']}")
        )
        log_root.parent.mkdir(parents=True, exist_ok=True)
        result = _optimize_fragment(
            fragment_atoms,
            charge=fragment["charge"],
            multiplicity=fragment["multiplicity"],
            calculator=calculator,
            output_root=log_root,
            settings=settings,
        )
        optimized = result["atoms"]
        optimized.positions += seed_center - np.asarray(optimized.positions).mean(axis=0)
        assembled.positions[indices] = optimized.positions
        fragment_path = (
            output_dir
            / "fragments"
            / f"{reaction['reaction_id']}_{endpoint_name}_{fragment['fragment_id']}.xyz"
        )
        fragment_path.parent.mkdir(parents=True, exist_ok=True)
        write(fragment_path, optimized, format="xyz")
        fragment_records.append(
            {
                "fragment_id": fragment["fragment_id"],
                "atom_indices": indices,
                "output_xyz": str(fragment_path),
                "output_sha256": sha256_of_file(fragment_path),
                **result["summary"],
                "atoms": optimized,
            }
        )
    return assembled, fragment_records


def _factor_id(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _scale_dimer(atoms: Atoms, first: int, second: int, factor: float) -> Atoms:
    positions = np.asarray(atoms.positions, dtype=float)
    left = positions[first]
    right = positions[second]
    midpoint = 0.5 * (left + right)
    new_left = midpoint + factor * (left - midpoint)
    new_right = midpoint + factor * (right - midpoint)
    result = atoms.copy()
    for index, position in enumerate(positions):
        if index == first or np.linalg.norm(position - left) <= np.linalg.norm(position - right):
            result.positions[index] += new_left - left
        else:
            result.positions[index] += new_right - right
    return result


def _write_state_audit_config(
    config: Mapping[str, Any],
    fragment_index: Mapping[tuple[str, str, str], Mapping[str, Any]],
    output_dir: Path,
    config_sha256: str,
) -> Path:
    audit = _mapping(config["state_audit"], "config.state_audit")
    scales = [
        _number(value, "config.state_audit.dimer_scale_factors[]")
        for value in _sequence(audit["dimer_scale_factors"], "state_audit.dimer_scale_factors")
    ]
    states = [_mapping(value, "state_audit.states[]") for value in audit["states"]]
    audit_root = output_dir / "state_audit"
    structures_dir = audit_root / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    structures: list[dict[str, str]] = []
    operations: list[dict[str, Any]] = []
    for raw in _sequence(audit["sources"], "state_audit.sources"):
        source = _mapping(raw, "state_audit.sources[]")
        key = (str(source["reaction_id"]), str(source["endpoint"]), str(source["fragment_id"]))
        if key not in fragment_index:
            raise ValueError(f"state_audit source {key!r} does not identify an optimized fragment")
        record = fragment_index[key]
        original_indices = list(record["atom_indices"])
        dimer_original = [int(value) for value in source["dimer_atom_indices"]]
        try:
            dimer_local = [original_indices.index(value) for value in dimer_original]
        except ValueError as exc:
            raise ValueError(f"state_audit dimer indices are not in fragment {key!r}") from exc
        for factor in scales:
            structure_id = f"{source['source_id']}_mm_x{_factor_id(factor)}"
            geometry = _scale_dimer(record["atoms"], dimer_local[0], dimer_local[1], factor)
            xyz_path = structures_dir / f"{structure_id}.xyz"
            write(xyz_path, geometry, format="xyz")
            structures.append({"id": structure_id, "xyz_path": f"structures/{xyz_path.name}"})
            operations.append(
                {"kind": "state_expansion", "structure": structure_id, "states": states}
            )
    sampling_config = {
        "schema_version": 1,
        "sampling_id": str(audit["sampling_id"]),
        "created": str(config["created"]),
        "derived_from": {
            "endpoint_set_id": str(config["endpoint_set_id"]),
            "endpoint_config_sha256": config_sha256,
        },
        "description": (
            "C0-S doublet/quartet audit at three M-M distances for Si2H3, Si2H5, "
            "Ge2H3, and Ge2H5. These candidates are not teacher-data approved."
        ),
        "structures": structures,
        "operations": operations,
        "filters": {
            "covalent_factor": 0.65,
            "bond_factor": 1.3,
            "allow_fragments": False,
            "duplicate_decimals": 3,
        },
    }
    output = audit_root / "sampling_config.json"
    write_json_atomic(output, sampling_config)
    return output


def run(config_path: Path, output_dir: Path, model_cache_dir: Path) -> None:
    config = load_config(config_path)
    reactions = validate_reactions(config, config_path)
    model = _mapping(config["model"], "config.model")
    settings = _mapping(config["optimization"], "config.optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _initialize_uma(
        model_name=str(model["name"]),
        checkpoint_path=None,
        model_cache_dir=model_cache_dir,
        task=str(model["task"]),
        device="cuda",
        inference_settings="default",
        seed=int(model["seed"]),
        fairchem_core_version=str(model["fairchem_core_version"]),
    )
    calculator = context["calculator"]
    config_digest = canonical_json_fingerprint(config)
    reaction_records: list[dict[str, Any]] = []
    fragment_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    all_converged = True
    for reaction in reactions:
        endpoint_records: dict[str, Any] = {}
        for endpoint_name in ENDPOINT_NAMES:
            assembled, fragments = _optimize_endpoint(
                reaction, endpoint_name, calculator, output_dir, settings
            )
            endpoint_path = (
                output_dir / "structures" / reaction["reaction_id"] / f"{endpoint_name}.xyz"
            )
            endpoint_path.parent.mkdir(parents=True, exist_ok=True)
            write(endpoint_path, assembled, format="xyz")
            public_fragments: list[dict[str, Any]] = []
            for record in fragments:
                key = (reaction["reaction_id"], endpoint_name, record["fragment_id"])
                fragment_index[key] = record
                public = {name: value for name, value in record.items() if name != "atoms"}
                public_fragments.append(public)
                all_converged = all_converged and bool(record["final_converged"])
            endpoint_records[endpoint_name] = {
                "source_xyz": reaction["endpoints"][endpoint_name]["xyz_path"],
                "source_sha256": reaction["endpoints"][endpoint_name]["source_sha256"],
                "output_xyz": str(endpoint_path),
                "output_sha256": sha256_of_file(endpoint_path),
                "atomic_numbers": [int(value) for value in assembled.numbers],
                "fragments": public_fragments,
            }
        reaction_records.append(
            {
                "reaction_id": reaction["reaction_id"],
                "tier": reaction["tier"],
                "charge": reaction["charge"],
                "multiplicity": reaction["multiplicity"],
                "atom_mapping": "same atomic number at every reactant/product index",
                "endpoints": endpoint_records,
            }
        )
    audit_config = _write_state_audit_config(config, fragment_index, output_dir, config_digest)
    summary = {
        "schema": "uma-pyscf-reaction-endpoint-preparation-v1",
        "endpoint_set_id": config["endpoint_set_id"],
        "config": {"path": str(config_path), "sha256": config_digest},
        "model": model,
        "optimization": settings,
        "all_fragments_converged": all_converged,
        "reaction_count": len(reaction_records),
        "reactions": reaction_records,
        "state_audit_sampling_config": {
            "path": str(audit_config),
            "sha256": sha256_of_file(audit_config),
        },
    }
    write_json_atomic(output_dir / "summary.json", summary)
    if not all_converged:
        raise RuntimeError("At least one endpoint fragment did not converge to final fmax")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.config, arguments.output_dir, arguments.model_cache_dir)
