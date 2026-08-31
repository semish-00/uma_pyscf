"""Fail-closed checks for the committed fairchem overfit-smoke configuration."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIG = REPO_ROOT / "configs/finetune/data/engineering_50_omol_ef_v1.yaml"
TRAIN_CONFIG = REPO_ROOT / "configs/finetune/engineering_50_overfit_v1.yaml"


class FineTuneConfigTests(unittest.TestCase):
    def test_omol_state_keys_and_train_only_normalization_are_fixed(self) -> None:
        config = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["dataset_name"], "omol")
        self.assertEqual(len(config["elem_refs"]), 100)
        self.assertEqual(
            {index for index, value in enumerate(config["elem_refs"]) if value != 0.0},
            {1, 14, 17, 32},
        )
        self.assertAlmostEqual(config["normalizer_rmsd"], 2.0571008908724995)
        for partition in ("train_dataset", "val_dataset"):
            self.assertEqual(
                config[partition]["a2g_args"]["r_data_keys"], ["charge", "spin"]
            )

    def test_overfit_smoke_is_single_gpu_bounded_and_debug_only(self) -> None:
        config = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["base_model_name"], "uma-s-1p2")
        self.assertIsNone(config["epochs"])
        self.assertEqual(config["steps"], 200)
        self.assertEqual(config["batch_size"], 2)
        self.assertTrue(config["job"]["debug"])
        self.assertEqual(config["job"]["scheduler"]["ranks_per_node"], 1)
        self.assertEqual(
            config["train_dataloader"]["batch_sampler_fn"]["on_error"],
            "warn_and_no_balance",
        )
        self.assertEqual(
            config["eval_dataloader"]["batch_sampler_fn"]["on_error"],
            "warn_and_no_balance",
        )


if __name__ == "__main__":
    unittest.main()
