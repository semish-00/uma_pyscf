"""Deterministic score selection with parent quotas."""

from __future__ import annotations

from pathlib import Path
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.sampling.selection import load_selection_config, select_candidates
from uma_pyscf.schemas.acquisition import AcquisitionScoreManifest, AcquisitionScoreRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/sampling/mf_pfp_screening_engineering_50_dry_run_v1.yaml"
TRAJECTORY_CONFIG_PATH = REPO_ROOT / "configs/sampling/mf_neb_arrhenius_selection_dry_run_v1.yaml"


def manifest() -> AcquisitionScoreManifest:
    records = tuple(
        AcquisitionScoreRecord(
            record_id=f"r{parent}{index}",
            parent_id=f"p{parent}",
            scores={
                "pfp_uma_force_rms": float(parent * 10 + index),
                "pfp_uma_combined_rank": float(index * 10 + parent),
            },
        )
        for parent in range(5)
        for index in range(3)
    )
    return AcquisitionScoreManifest(
        score_id="scores_v1",
        source={"id": "pool_v1", "sha256": "a" * 64},
        records=records,
    )


class SelectionTests(unittest.TestCase):
    def test_committed_config_is_valid_and_selection_is_deterministic(self) -> None:
        config = load_selection_config(CONFIG_PATH)
        first = select_candidates(manifest(), config, score_file_sha256="b" * 64)
        second = select_candidates(manifest(), config, score_file_sha256="b" * 64)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertLessEqual(len(first.union_record_ids), 30)
        for selected in first.policy_selections.values():
            self.assertEqual(len(selected), 10)
            parent_counts: dict[str, int] = {}
            by_id = {record.record_id: record for record in manifest().records}
            for record_id in selected:
                parent = by_id[record_id].parent_id
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
            self.assertTrue(all(count <= 2 for count in parent_counts.values()))

    def test_missing_score_and_union_overflow_fail_closed(self) -> None:
        config = load_selection_config(CONFIG_PATH)
        missing = AcquisitionScoreManifest(
            score_id="missing",
            source={"id": "pool", "sha256": "a" * 64},
            records=tuple(
                AcquisitionScoreRecord(record_id=f"r{i}", parent_id=f"p{i}", scores={"other": 1.0})
                for i in range(10)
            ),
        )
        with self.assertRaises(ValidationError):
            select_candidates(missing, config, score_file_sha256="b" * 64)
        overflow = dict(config)
        overflow["max_union_records"] = 1
        with self.assertRaises(ValidationError):
            select_candidates(manifest(), overflow, score_file_sha256="b" * 64)

    def test_trajectory_quota_is_enforced(self) -> None:
        config = load_selection_config(TRAJECTORY_CONFIG_PATH)
        records = tuple(
            AcquisitionScoreRecord(
                record_id=f"r{parent}{trajectory}{index}",
                parent_id=f"p{parent}",
                trajectory_id=f"t{parent}{trajectory}",
                frame_index=index,
                scores={
                    "pfp_uma_force_rms": float(100 - index),
                    "pfp_uma_combined_rank": float(100 - index),
                },
            )
            for parent in range(2)
            for trajectory in range(2)
            for index in range(5)
        )
        scores = AcquisitionScoreManifest(
            score_id="trajectory_scores",
            source={"id": "pool", "sha256": "a" * 64},
            records=records,
        )

        selection = select_candidates(scores, config, score_file_sha256="b" * 64)

        by_id = {record.record_id: record for record in records}
        for selected in selection.policy_selections.values():
            trajectory_counts: dict[str, int] = {}
            for record_id in selected:
                trajectory_id = str(by_id[record_id].trajectory_id)
                trajectory_counts[trajectory_id] = trajectory_counts.get(trajectory_id, 0) + 1
            self.assertTrue(all(count <= 3 for count in trajectory_counts.values()))

    def test_trajectory_quota_requires_trajectory_ids(self) -> None:
        config = load_selection_config(TRAJECTORY_CONFIG_PATH)
        with self.assertRaisesRegex(ValidationError, "trajectory_id"):
            select_candidates(manifest(), config, score_file_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
