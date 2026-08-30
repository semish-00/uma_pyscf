"""Dataset export config validation without scientific dependencies."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.datasets.ase_lmdb_cli import load_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINEERING_CONFIG = REPO_ROOT / "configs" / "datasets" / "engineering_50_ase_lmdb_v1.yaml"

VALID = (
    "schema_version: 1\n"
    "dataset_id: ds_unit_001\n"
    "task: omol\n"
    "regression_tasks: ef\n"
    "shard_size: 100\n"
    'fairchem_core_version: "2.22.0"\n'
)


class DatasetConfigTests(unittest.TestCase):
    def test_committed_engineering_config_is_valid(self) -> None:
        config = load_dataset_config(ENGINEERING_CONFIG)
        self.assertEqual(config["dataset_id"], "ds_sigehcl_001")
        self.assertEqual(config["fairchem_core_version"], "2.22.0")

    def test_unknown_keys_and_non_ef_targets_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.yaml"
            path.write_text(VALID + "shuffle: true\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_dataset_config(path)
            path.write_text(VALID.replace("regression_tasks: ef", "regression_tasks: efs"))
            with self.assertRaises(ValidationError):
                load_dataset_config(path)

    def test_shard_size_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.yaml"
            path.write_text(VALID.replace("shard_size: 100", "shard_size: 0"))
            with self.assertRaises(ValidationError):
                load_dataset_config(path)


if __name__ == "__main__":
    unittest.main()
