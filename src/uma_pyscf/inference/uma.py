"""Evaluate a named base UMA model on a verified ASE-LMDB dataset."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import sha256_of_file
from ..core.io import write_json_atomic
from ..schemas.dataset_manifest import AseDatasetManifest
from .metrics import PredictionRecord, Vector, summarize_predictions

__all__ = ["evaluate_ase_lmdb"]


def _inference_api() -> tuple[Any, Any, Any, Any]:
    try:
        import ase  # type: ignore[import-not-found]
        from ase.db import connect  # type: ignore[import-not-found]
        from fairchem.core import (  # type: ignore[import-not-found]
            FAIRChemCalculator,
            pretrained_mlip,
        )
        from fairchem.core.units.mlip_unit import (  # type: ignore[import-not-found]
            load_predict_unit,
        )
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValidationError(
            "UMA evaluation requires Python 3.11+ and the inference extra; install with "
            "`pip install 'uma-pyscf[inference]'`."
        ) from exc
    return ase, connect, FAIRChemCalculator, (pretrained_mlip, load_predict_unit, torch)


def _git_commit(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _vectors(values: Any, *, path: str) -> tuple[Vector, ...]:
    rows: list[Vector] = []
    for index, raw in enumerate(values):
        if len(raw) != 3:
            raise ValidationError(f"{path}[{index}] must have three components.")
        rows.append((float(raw[0]), float(raw[1]), float(raw[2])))
    return tuple(rows)


def _loaded_atoms(row: Any) -> Any:
    atoms = row.toatoms()
    if isinstance(row.data, Mapping):
        atoms.info.update(row.data)
    return atoms


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise ValidationError(f"Required distribution {distribution!r} is not installed.") from exc


def evaluate_ase_lmdb(
    manifest: AseDatasetManifest,
    *,
    manifest_sha256: str,
    dataset_dir: str | Path,
    evaluation_id: str,
    model_name: str,
    model_source: str,
    model_license: str,
    checkpoint_path: str | Path | None,
    model_cache_dir: str | Path,
    task: str,
    device: str,
    inference_settings: str,
    seed: int,
    fairchem_core_version: str,
    partitions: tuple[str, ...],
    output_path: str | Path,
    repository: str | Path,
    container_sha256: str,
) -> dict[str, Any]:
    """Run base-model inference and atomically publish predictions plus metrics."""
    cache_root = Path(model_cache_dir).resolve()
    configured_cache = os.environ.get("FAIRCHEM_CACHE_DIR")
    if configured_cache is None or Path(configured_cache).resolve() != cache_root:
        raise ValidationError(
            "FAIRCHEM_CACHE_DIR must equal model_cache_dir before fairchem is imported; "
            "fairchem-core 2.22.0 otherwise stores the checkpoint in its default cache."
        )
    ase, connect, calculator_type, dependencies = _inference_api()
    pretrained_mlip, load_predict_unit, torch = dependencies
    installed_fairchem = _package_version("fairchem-core")
    if installed_fairchem != fairchem_core_version:
        raise ValidationError(
            f"fairchem-core is {installed_fairchem}, expected {fairchem_core_version}."
        )
    if task != manifest.task:
        raise ValidationError(
            f"Evaluation task {task!r} does not match dataset task {manifest.task!r}."
        )
    unknown = sorted(set(partitions) - set(manifest.partitions))
    if unknown:
        raise ValidationError(f"Evaluation names unknown dataset partitions: {unknown!r}.")

    cache_root.mkdir(parents=True, exist_ok=True)
    local_checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    if local_checkpoint is not None:
        if not local_checkpoint.is_file():
            raise ValidationError(f"Fine-tuned checkpoint {local_checkpoint} is not a file.")
        predictor = load_predict_unit(
            local_checkpoint,
            device=device,
            inference_settings=inference_settings,
            seed=seed,
        )
    else:
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
            inference_settings=inference_settings,
            cache_dir=str(cache_root),
            seed=seed,
        )
    cache_paths = [
        path
        for path in sorted(cache_root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    if not any(path.stat().st_size > 1_000_000 for path in cache_paths):
        raise ValidationError(
            f"Model cache {cache_root} contains no checkpoint-sized regular file."
        )
    cache_files = [
        {
            "path": path.relative_to(cache_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_of_file(path),
        }
        for path in cache_paths
    ]
    calculator = calculator_type(predictor, task_name=task)
    root = Path(dataset_dir)
    predictions: list[PredictionRecord] = []
    for partition in partitions:
        block = manifest.partitions[partition]
        for shard in block["shards"]:
            shard_path = root / shard["path"]
            actual_sha256 = sha256_of_file(shard_path)
            if actual_sha256 != shard["sha256"]:
                raise ValidationError(
                    f"ASE-LMDB shard checksum mismatch for {shard_path}: "
                    f"{actual_sha256} != {shard['sha256']}."
                )
            with connect(str(shard_path), readonly=True, use_lock_file=False) as database:
                if len(database) != shard["record_count"]:
                    raise ValidationError(
                        f"{shard_path} has {len(database)} rows, expected {shard['record_count']}."
                    )
                for row_index, expected_id in enumerate(shard["record_ids"], start=1):
                    atoms = _loaded_atoms(database.get(row_index))
                    record_id = str(atoms.info.get("record_id", ""))
                    if record_id != expected_id:
                        raise ValidationError(
                            f"{shard_path}[{row_index}] is {record_id!r}, expected "
                            f"{expected_id!r}."
                        )
                    if atoms.pbc.any():
                        raise ValidationError(f"{record_id!r} is periodic but task is OMol.")
                    if "charge" not in atoms.info or "spin" not in atoms.info:
                        raise ValidationError(f"{record_id!r} is missing charge or spin metadata.")
                    reference_energy = float(atoms.get_potential_energy())
                    reference_forces = _vectors(
                        atoms.get_forces(), path=f"{record_id}.reference_forces"
                    )
                    atoms.calc = calculator
                    predicted_energy = float(atoms.get_potential_energy())
                    predicted_forces = _vectors(
                        atoms.get_forces(), path=f"{record_id}.predicted_forces"
                    )
                    predictions.append(
                        PredictionRecord(
                            partition=partition,
                            record_id=record_id,
                            atomic_numbers=tuple(
                                int(value) for value in atoms.get_atomic_numbers()
                            ),
                            charge=int(atoms.info["charge"]),
                            multiplicity=int(atoms.info["spin"]),
                            reference_energy_ev=reference_energy,
                            predicted_energy_ev=predicted_energy,
                            reference_forces_ev_per_angstrom=reference_forces,
                            predicted_forces_ev_per_angstrom=predicted_forces,
                        )
                    )

    by_partition = {
        partition: tuple(record for record in predictions if record.partition == partition)
        for partition in partitions
    }
    artifact: dict[str, Any] = {
        "schema": "uma-pyscf-uma-evaluation-v1",
        "evaluation_id": evaluation_id,
        "dataset": {
            "id": manifest.dataset_id,
            "manifest_sha256": manifest_sha256,
            "split_id": manifest.split["id"],
            "split_sha256": manifest.split["sha256"],
            "partitions": list(partitions),
        },
        "model": {
            "name": model_name,
            "source": model_source,
            "license": model_license,
            "task": task,
            "device": device,
            "inference_settings": inference_settings,
            "seed": seed,
            "evaluated_checkpoint": (
                {
                    "path": str(local_checkpoint),
                    "bytes": local_checkpoint.stat().st_size,
                    "sha256": sha256_of_file(local_checkpoint),
                }
                if local_checkpoint is not None
                else None
            ),
            "cache_files": cache_files,
        },
        "runtime": {
            "python": platform.python_version(),
            "ase": str(ase.__version__),
            "fairchem_core": installed_fairchem,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cuda_device": (
                str(torch.cuda.get_device_name()) if torch.cuda.is_available() else None
            ),
            "container_sha256": container_sha256,
            "git_commit": _git_commit(Path(repository)),
        },
        "units": {"energy": "eV", "forces": "eV/angstrom"},
        "metrics_by_partition": {
            partition: summarize_predictions(records)
            for partition, records in by_partition.items()
        },
        "predictions": [record.to_dict() for record in predictions],
    }
    write_json_atomic(output_path, artifact)
    return artifact
