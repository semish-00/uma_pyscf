"""ASE-LMDB conversion and load-back verification.

This suite is skipped by the normal dependency-light test environment. Run it
with the ``dataset`` extra to exercise ASE and the LMDB backend.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    import ase  # noqa: F401
    from ase.db import connect
    import ase_db_backends  # noqa: F401
except ImportError:
    HAS_ASE_LMDB = False
else:
    HAS_ASE_LMDB = True

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.datasets.ase_lmdb import (
    export_ase_lmdb_dataset,
    verify_ase_lmdb_dataset,
)
from uma_pyscf.schemas.label_record import (
    ElectronicState,
    Engine,
    LabelRecord,
    Method,
    QcState,
    RawArtifact,
    Results,
    Structure,
)
from uma_pyscf.schemas.split_manifest import SplitManifest


def record(
    record_id: str,
    atomic_numbers: tuple[int, ...],
    multiplicity: int,
    gradient: tuple[tuple[float, float, float], ...],
) -> LabelRecord:
    return LabelRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=atomic_numbers,
            positions_angstrom=tuple(
                (0.0, 0.0, float(index) * 0.74) for index in range(len(atomic_numbers))
            ),
            parent_structure_id=f"{record_id}_parent",
        ),
        state=ElectronicState(
            charge=0,
            multiplicity=multiplicity,
            spin_2s=multiplicity - 1,
        ),
        method=Method(
            functional="wb97m-v",
            basis="def2-tzvpd",
            ecp=None,
            aux_basis=None,
            grid_level=5,
            nlc_grid_level=5,
            grid_response=True,
            density_fit=True,
            scf_conv_tol=1e-10,
            scf_max_cycle=250,
        ),
        engine=Engine(name="gpu4pyscf", versions={"gpu4pyscf": "1.8.1"}),
        results=Results(
            energy_hartree=-float(len(atomic_numbers)),
            gradient_hartree_per_bohr=gradient,
            converged=True,
        ),
        raw=RawArtifact(),
        qc=QcState(status="accepted"),
    )


def fixture() -> tuple[tuple[LabelRecord, ...], SplitManifest]:
    records = (
        record("h_doublet", (1,), 2, ((0.25, 0.0, 0.0),)),
        record("h2_singlet", (1, 1), 1, ((0.0, 0.0, -0.1), (0.0, 0.0, 0.1))),
    )
    split = SplitManifest(
        split_id="split_unit_parent_v1",
        axis="parent",
        seed=1,
        partitions={"train": 0.5, "holdout": 0.5},
        source={"id": "source_v1", "sha256": "a" * 64},
        group_assignments={"h_doublet_parent": "holdout", "h2_singlet_parent": "train"},
        record_assignments={"train": ("h2_singlet",), "holdout": ("h_doublet",)},
    )
    return records, split


@unittest.skipUnless(HAS_ASE_LMDB, "requires ASE and ase-db-backends")
class AseLmdbIntegrationTests(unittest.TestCase):
    def test_export_preserves_units_sign_state_and_split(self) -> None:
        records, split = fixture()
        checksums = {entry.record_id: "b" * 64 for entry in records}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ds_unit_001"
            manifest, path = export_ase_lmdb_dataset(
                records,
                split,
                dataset_id="ds_unit_001",
                shard_size=1,
                fairchem_core_version="2.22.0",
                record_checksums_sha256=checksums,
                split_sha256="c" * 64,
                output_dir=root,
            )
            verify_ase_lmdb_dataset(
                manifest,
                records,
                record_checksums_sha256=checksums,
                output_dir=root,
            )
            self.assertTrue(path.is_file())
            self.assertEqual(manifest.record_count, 2)
            self.assertEqual(manifest.partitions["train"]["record_ids"], ["h2_singlet"])

            shard = root / manifest.partitions["holdout"]["shards"][0]["path"]
            with connect(str(shard), readonly=True, use_lock_file=False) as database:
                row = database.get(1)
                atoms = row.toatoms()
                atoms.info.update(row.data)
            self.assertEqual(atoms.info["charge"], 0)
            self.assertEqual(atoms.info["spin"], 2)
            self.assertEqual(atoms.info["multiplicity"], 2)
            self.assertEqual(atoms.info["task"], "omol")
            self.assertLess(atoms.get_forces()[0][0], 0.0)

    def test_changed_shard_is_detected_before_training(self) -> None:
        records, split = fixture()
        checksums = {entry.record_id: "b" * 64 for entry in records}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ds_unit_001"
            manifest, _ = export_ase_lmdb_dataset(
                records,
                split,
                dataset_id="ds_unit_001",
                shard_size=10,
                fairchem_core_version="2.22.0",
                record_checksums_sha256=checksums,
                split_sha256="c" * 64,
                output_dir=root,
            )
            shard = root / manifest.partitions["train"]["shards"][0]["path"]
            with shard.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaises(ValidationError) as caught:
                verify_ase_lmdb_dataset(
                    manifest,
                    records,
                    record_checksums_sha256=checksums,
                    output_dir=root,
                )
        self.assertIn("checksum mismatch", str(caught.exception))

    def test_nonaccepted_record_and_existing_destination_fail_closed(self) -> None:
        records, split = fixture()
        rejected = (replace(records[0], qc=QcState(status="rejected")), records[1])
        checksums = {entry.record_id: "b" * 64 for entry in records}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing"
            destination.mkdir()
            with self.assertRaises(ValidationError):
                export_ase_lmdb_dataset(
                    records,
                    split,
                    dataset_id="ds_unit_001",
                    shard_size=10,
                    fairchem_core_version="2.22.0",
                    record_checksums_sha256=checksums,
                    split_sha256="c" * 64,
                    output_dir=destination,
                )
            with self.assertRaises(ValidationError) as caught:
                export_ase_lmdb_dataset(
                    rejected,
                    split,
                    dataset_id="ds_unit_002",
                    shard_size=10,
                    fairchem_core_version="2.22.0",
                    record_checksums_sha256=checksums,
                    split_sha256="c" * 64,
                    output_dir=Path(directory) / "rejected",
                )
        self.assertIn("accepted", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
