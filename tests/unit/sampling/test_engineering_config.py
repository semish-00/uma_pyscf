"""Regression checks for the first 50-candidate engineering set."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from uma_pyscf.calculators.config import load_dft_config, resource_for_candidate
from uma_pyscf.sampling.generate import generate_candidates

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLING_CONFIG = REPO_ROOT / "configs" / "sampling" / "engineering_50_v1.yaml"
DFT_CONFIG = REPO_ROOT / "configs" / "dft" / "omol_wb97mv_tzvpd_v1.yaml"


class EngineeringConfigTests(unittest.TestCase):
    def test_generates_exactly_fifty_accepted_unique_candidates(self) -> None:
        manifest, report = generate_candidates(SAMPLING_CONFIG)
        self.assertEqual(report.counts, {"total": 50, "accepted": 50, "rejected": 0})
        self.assertEqual(len({record.record_id for record in manifest.records}), 50)
        self.assertEqual(
            Counter(record.structure.atom_count for record in manifest.records),
            {5: 40, 8: 10},
        )

    def test_exercises_both_resource_tiers(self) -> None:
        manifest, _ = generate_candidates(SAMPLING_CONFIG)
        config = load_dft_config(DFT_CONFIG)
        resources = Counter(
            (
                resource_for_candidate(record, config)["ncpus"],
                resource_for_candidate(record, config)["max_memory_mb"],
            )
            for record in manifest.records
        )
        self.assertEqual(resources, {(8, 24000): 40, (16, 48000): 10})


if __name__ == "__main__":
    unittest.main()
