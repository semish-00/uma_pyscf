"""ASE dataset manifest integrity invariants."""

from __future__ import annotations

import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.schemas.dataset_manifest import AseDatasetManifest


def manifest_dict() -> dict[str, object]:
    return {
        "schema": "uma-pyscf-ase-dataset-manifest-v1",
        "dataset_id": "ds_unit_001",
        "format": "ase-lmdb",
        "task": "omol",
        "regression_tasks": "ef",
        "units": {
            "energy": "eV",
            "forces": "eV/angstrom",
            "positions": "angstrom",
        },
        "force_convention": "forces=-gradient",
        "compatibility": {
            "ase_version": "3.26.0",
            "ase_db_backends_version": "0.11.0",
            "fairchem_core_version": "2.22.0",
            "fairchem_a2g_data_keys": ["charge", "spin"],
        },
        "split": {"id": "split_unit_v1", "sha256": "a" * 64},
        "record_checksums_sha256": {"h": "b" * 64, "h2": "c" * 64},
        "partitions": {
            "holdout": {
                "record_count": 1,
                "record_ids": ["h"],
                "shards": [
                    {
                        "path": "holdout/data.0000.aselmdb",
                        "sha256": "d" * 64,
                        "record_count": 1,
                        "record_ids": ["h"],
                    }
                ],
            },
            "train": {
                "record_count": 1,
                "record_ids": ["h2"],
                "shards": [
                    {
                        "path": "train/data.0000.aselmdb",
                        "sha256": "e" * 64,
                        "record_count": 1,
                        "record_ids": ["h2"],
                    }
                ],
            },
        },
    }


class DatasetManifestTests(unittest.TestCase):
    def test_manifest_round_trips_and_counts_records(self) -> None:
        manifest = AseDatasetManifest.from_dict(manifest_dict())
        self.assertEqual(manifest.record_count, 2)
        self.assertEqual(manifest.to_dict(), manifest_dict())

    def test_charge_and_spin_must_be_explicit_fairchem_data_keys(self) -> None:
        data = manifest_dict()
        compatibility = dict(data["compatibility"])  # type: ignore[arg-type]
        compatibility["fairchem_a2g_data_keys"] = []
        data["compatibility"] = compatibility
        with self.assertRaises(ValidationError) as caught:
            AseDatasetManifest.from_dict(data)
        self.assertIn("charge", str(caught.exception))

    def test_shard_records_must_exactly_cover_the_partition(self) -> None:
        data = manifest_dict()
        checksums = dict(data["record_checksums_sha256"])  # type: ignore[arg-type]
        checksums["h3"] = "f" * 64
        data["record_checksums_sha256"] = checksums
        partitions = dict(data["partitions"])  # type: ignore[arg-type]
        train = dict(partitions["train"])
        train["record_ids"] = ["h2", "h3"]
        train["record_count"] = 2
        partitions["train"] = train
        data["partitions"] = partitions
        with self.assertRaises(ValidationError) as caught:
            AseDatasetManifest.from_dict(data)
        self.assertIn("shards", str(caught.exception))

    def test_a_shard_path_cannot_escape_the_dataset_root(self) -> None:
        data = manifest_dict()
        partitions = dict(data["partitions"])  # type: ignore[arg-type]
        train = dict(partitions["train"])
        shards = [dict(train["shards"][0])]  # type: ignore[index]
        shards[0]["path"] = "../outside.aselmdb"
        train["shards"] = shards
        partitions["train"] = train
        data["partitions"] = partitions
        with self.assertRaises(ValidationError):
            AseDatasetManifest.from_dict(data)


if __name__ == "__main__":
    unittest.main()
