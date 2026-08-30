"""Convert accepted canonical labels to verified ASE-LMDB shards.

This module is the sole place where a canonical gradient becomes a force.
Canonical records remain in Hartree and Hartree/Bohr; exported ASE objects use
eV and eV/Angstrom, as required by fairchem.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import shutil
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import sha256_of_file
from ..schemas.dataset_manifest import AseDatasetManifest
from ..schemas.label_record import LabelRecord
from ..schemas.split_manifest import SplitManifest

__all__ = [
    "export_ase_lmdb_dataset",
    "label_record_to_atoms",
    "verify_ase_lmdb_dataset",
]

_ABS_TOLERANCE = 1e-10


def _ase_api() -> tuple[Any, Any, Any, Any]:
    """Return the optional ASE interfaces or explain how to install them."""
    try:
        import ase  # type: ignore[import-not-found]
        from ase import units
        from ase.calculators.singlepoint import (  # type: ignore[import-not-found]
            SinglePointCalculator,
        )
        from ase.db import connect  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValidationError(
            "ASE-LMDB export requires the dataset extra; install with "
            "`pip install 'uma-pyscf[dataset]'`."
        ) from exc
    return ase, SinglePointCalculator, connect, units


def _backend_version() -> str:
    try:
        return version("ase-db-backends")
    except PackageNotFoundError as exc:
        raise ValidationError(
            "ASE-LMDB export requires ase-db-backends; install the dataset extra."
        ) from exc


def label_record_to_atoms(record: LabelRecord, *, record_sha256: str) -> Any:
    """Build a nonperiodic ASE object with fairchem labels and state metadata."""
    ase, single_point_calculator, _, units = _ase_api()
    if record.qc.status != "accepted":
        raise ValidationError(
            f"Record {record.record_id!r} has qc.status {record.qc.status!r}; only accepted "
            "records may enter an ASE-LMDB dataset."
        )
    if not record.results.converged:
        raise ValidationError(
            f"Record {record.record_id!r} is not converged and cannot become a teaching label."
        )

    energy_ev = record.results.energy_hartree * units.Hartree
    conversion = units.Hartree / units.Bohr
    forces = [
        [-component * conversion for component in row]
        for row in record.results.gradient_hartree_per_bohr
    ]
    atoms = ase.Atoms(
        numbers=record.structure.atomic_numbers,
        positions=record.structure.positions_angstrom,
        pbc=False,
    )
    atoms.info.update(
        {
            "sid": record.record_id,
            "record_id": record.record_id,
            "charge": record.state.charge,
            # fairchem's OMol `spin` input is total multiplicity (2S+1), not PySCF spin_2s.
            "spin": record.state.multiplicity,
            "multiplicity": record.state.multiplicity,
            "task": "omol",
            "source_record_sha256": record_sha256,
        }
    )
    atoms.calc = single_point_calculator(atoms, energy=energy_ev, forces=forces)
    return atoms


def _assert_close(actual: float, expected: float, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=_ABS_TOLERANCE):
        raise ValidationError(f"{path} is {actual!r}, expected {expected!r}.")


def _verify_atoms(atoms: Any, record: LabelRecord, record_sha256: str, path: str) -> None:
    """Compare one loaded ASE row with its canonical source record."""
    _, _, _, units = _ase_api()
    expected_info = {
        "sid": record.record_id,
        "record_id": record.record_id,
        "charge": record.state.charge,
        "spin": record.state.multiplicity,
        "multiplicity": record.state.multiplicity,
        "task": "omol",
        "source_record_sha256": record_sha256,
    }
    for key, expected in expected_info.items():
        if atoms.info.get(key) != expected:
            raise ValidationError(
                f"{path}.info[{key!r}] is {atoms.info.get(key)!r}, expected {expected!r}."
            )
    if atoms.pbc.any():
        raise ValidationError(f"{path} is periodic; OMol teaching structures must be aperiodic.")
    if (
        tuple(int(value) for value in atoms.get_atomic_numbers())
        != record.structure.atomic_numbers
    ):
        raise ValidationError(f"{path} changed atomic numbers or atom ordering.")
    positions = atoms.get_positions()
    for atom_index, (actual, expected) in enumerate(
        zip(positions, record.structure.positions_angstrom, strict=True)
    ):
        for component, (got, wanted) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(float(got), wanted, f"{path}.positions[{atom_index}][{component}]")

    _assert_close(
        float(atoms.get_potential_energy()),
        record.results.energy_hartree * units.Hartree,
        f"{path}.energy_ev",
    )
    conversion = units.Hartree / units.Bohr
    forces = atoms.get_forces()
    for atom_index, (actual, gradient) in enumerate(
        zip(forces, record.results.gradient_hartree_per_bohr, strict=True)
    ):
        for component, (got, source_gradient) in enumerate(zip(actual, gradient, strict=True)):
            _assert_close(
                float(got),
                -source_gradient * conversion,
                f"{path}.forces[{atom_index}][{component}]",
            )


def _loaded_atoms(row: Any) -> Any:
    atoms = row.toatoms()
    if isinstance(row.data, dict):
        atoms.info.update(row.data)
    return atoms


def _chunks(values: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(values[start : start + size]) for start in range(0, len(values), size)]


def _write_shard(
    path: Path,
    record_ids: Sequence[str],
    records: dict[str, LabelRecord],
    checksums: dict[str, str],
) -> None:
    _, _, connect, _ = _ase_api()
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(path)) as database:
        for record_id in record_ids:
            atoms = label_record_to_atoms(records[record_id], record_sha256=checksums[record_id])
            database.write(atoms, data=atoms.info)
    path.with_name(f"{path.name}-lock").unlink(missing_ok=True)


def _verify_shard(
    path: Path,
    record_ids: Sequence[str],
    records: dict[str, LabelRecord],
    checksums: dict[str, str],
) -> None:
    _, _, connect, _ = _ase_api()
    if not path.is_file():
        raise ValidationError(f"Missing ASE-LMDB shard {path}.")
    with connect(str(path), readonly=True, use_lock_file=False) as database:
        if len(database) != len(record_ids):
            raise ValidationError(
                f"{path} contains {len(database)} rows, expected {len(record_ids)}."
            )
        for row_index, record_id in enumerate(record_ids, start=1):
            row = database.get(row_index)
            _verify_atoms(
                _loaded_atoms(row),
                records[record_id],
                checksums[record_id],
                f"{path}[{row_index}]",
            )


def export_ase_lmdb_dataset(
    records: Sequence[LabelRecord],
    split: SplitManifest,
    *,
    dataset_id: str,
    shard_size: int,
    fairchem_core_version: str,
    record_checksums_sha256: dict[str, str],
    split_sha256: str,
    output_dir: str | Path,
) -> tuple[AseDatasetManifest, Path]:
    """Write, load back, checksum, and atomically publish an ASE-LMDB dataset."""
    ase, _, _, _ = _ase_api()
    backend_version = _backend_version()
    if shard_size < 1:
        raise ValidationError(f"shard_size must be at least 1; got {shard_size}.")

    records_by_id: dict[str, LabelRecord] = {}
    for record in records:
        if record.record_id in records_by_id:
            raise ValidationError(f"Duplicate record id {record.record_id!r} in --records.")
        records_by_id[record.record_id] = record
    assigned = {
        record_id
        for partition_ids in split.record_assignments.values()
        for record_id in partition_ids
    }
    if assigned != set(records_by_id):
        missing = sorted(assigned - set(records_by_id))
        extra = sorted(set(records_by_id) - assigned)
        raise ValidationError(
            f"Records must exactly match split assignments; missing={missing!r}, extra={extra!r}."
        )
    if set(record_checksums_sha256) != assigned:
        raise ValidationError("Record checksum keys must exactly match split assignments.")

    destination = Path(output_dir)
    if destination.exists():
        raise ValidationError(
            f"Dataset output {destination} already exists; choose a new versioned directory."
        )
    scratch = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if scratch.exists():
        raise ValidationError(f"Dataset scratch path already exists: {scratch}.")
    scratch.mkdir(parents=True)

    try:
        partition_blocks: dict[str, dict[str, Any]] = {}
        for partition in sorted(split.record_assignments):
            record_ids = split.record_assignments[partition]
            shards: list[dict[str, Any]] = []
            for shard_index, shard_ids in enumerate(_chunks(record_ids, shard_size)):
                relative = Path(partition) / f"data.{shard_index:04d}.aselmdb"
                shard_path = scratch / relative
                _write_shard(shard_path, shard_ids, records_by_id, record_checksums_sha256)
                _verify_shard(shard_path, shard_ids, records_by_id, record_checksums_sha256)
                shards.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": sha256_of_file(shard_path),
                        "record_count": len(shard_ids),
                        "record_ids": list(shard_ids),
                    }
                )
            partition_blocks[partition] = {
                "record_count": len(record_ids),
                "record_ids": list(record_ids),
                "shards": shards,
            }

        manifest = AseDatasetManifest(
            dataset_id=dataset_id,
            compatibility={
                "ase_version": str(ase.__version__),
                "ase_db_backends_version": backend_version,
                "fairchem_core_version": fairchem_core_version,
                "fairchem_a2g_data_keys": ["charge", "spin"],
            },
            split={"id": split.split_id, "sha256": split_sha256},
            record_checksums_sha256=record_checksums_sha256,
            partitions=partition_blocks,
        )
        from ..core.io import write_json_atomic

        write_json_atomic(scratch / "dataset_manifest.json", manifest.to_dict())
        os.replace(scratch, destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return manifest, destination / "dataset_manifest.json"


def verify_ase_lmdb_dataset(
    manifest: AseDatasetManifest,
    records: Sequence[LabelRecord],
    *,
    record_checksums_sha256: dict[str, str],
    output_dir: str | Path,
) -> None:
    """Recheck shard hashes and every loaded row against the canonical records."""
    records_by_id = {record.record_id: record for record in records}
    if len(records_by_id) != len(records):
        raise ValidationError("Canonical records contain duplicate record ids.")
    if record_checksums_sha256 != manifest.record_checksums_sha256:
        raise ValidationError(
            "Canonical record checksums do not match dataset.record_checksums_sha256."
        )
    root = Path(output_dir)
    for partition, block in manifest.partitions.items():
        del partition
        for shard in block["shards"]:
            path = root / shard["path"]
            actual = sha256_of_file(path)
            if actual != shard["sha256"]:
                raise ValidationError(
                    f"ASE-LMDB shard checksum mismatch for {path}: {actual} != {shard['sha256']}."
                )
            _verify_shard(
                path,
                shard["record_ids"],
                records_by_id,
                record_checksums_sha256,
            )
