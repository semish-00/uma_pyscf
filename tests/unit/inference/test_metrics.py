"""Base UMA metric aggregation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.inference.cli import load_evaluation_config
from uma_pyscf.inference.metrics import PredictionRecord, summarize_predictions

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs" / "evaluation" / "engineering_50_base_uma_s_1p2_v1.yaml"


def prediction(record_id: str, energy_error: float, force_error: float) -> PredictionRecord:
    return PredictionRecord(
        partition="holdout",
        record_id=record_id,
        atomic_numbers=(1, 1),
        charge=0,
        multiplicity=1,
        reference_energy_ev=-10.0,
        predicted_energy_ev=-10.0 + energy_error,
        reference_forces_ev_per_angstrom=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        predicted_forces_ev_per_angstrom=(
            (force_error, force_error, force_error),
            (force_error, force_error, force_error),
        ),
    )


class UmaEvaluationMetricTests(unittest.TestCase):
    def test_metrics_use_total_energy_and_force_components(self) -> None:
        metrics = summarize_predictions(
            (prediction("first", 2.0, 1.0), prediction("second", -1.0, -2.0))
        )
        self.assertEqual(metrics["records"], 2)
        self.assertEqual(metrics["atoms"], 4)
        self.assertEqual(metrics["compositions"], 1)
        self.assertAlmostEqual(float(metrics["energy_mean_error_ev"]), 0.5)
        self.assertAlmostEqual(float(metrics["energy_mae_ev"]), 1.5)
        self.assertAlmostEqual(float(metrics["energy_mae_ev_per_atom"]), 0.75)
        self.assertAlmostEqual(
            float(metrics["energy_same_composition_centered_mae_ev"]), 1.5
        )
        self.assertAlmostEqual(float(metrics["force_component_mae_ev_per_angstrom"]), 1.5)
        self.assertEqual(prediction("one", 2.0, 0.0).to_dict()["energy_error_ev"], 2.0)

    def test_empty_partition_and_wrong_force_shape_fail_closed(self) -> None:
        with self.assertRaises(ValidationError):
            summarize_predictions(())
        with self.assertRaises(ValidationError):
            PredictionRecord(
                partition="train",
                record_id="bad",
                atomic_numbers=(1, 1),
                charge=0,
                multiplicity=1,
                reference_energy_ev=0.0,
                predicted_energy_ev=0.0,
                reference_forces_ev_per_angstrom=((0.0, 0.0, 0.0),),
                predicted_forces_ev_per_angstrom=((0.0, 0.0, 0.0),),
            )

    def test_committed_config_is_valid_and_unknown_key_is_rejected(self) -> None:
        config = load_evaluation_config(CONFIG)
        self.assertEqual(config["model_name"], "uma-s-1p2")
        self.assertEqual(config["partitions"], ["train", "holdout"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(CONFIG.read_text(encoding="utf-8") + "shuffle: true\n")
            with self.assertRaises(ValidationError):
                load_evaluation_config(path)

        fine_tuned = load_evaluation_config(
            REPO_ROOT / "configs/evaluation/engineering_50_finetuned_200step_v1.yaml"
        )
        self.assertEqual(fine_tuned["model_name"], "engineering-50-overfit-200step-v1")


if __name__ == "__main__":
    unittest.main()
