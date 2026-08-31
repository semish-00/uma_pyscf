"""Acquisition-score and selection-manifest schema tests."""

from __future__ import annotations

from copy import deepcopy
import unittest

from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.schemas.acquisition import (
    AcquisitionScoreManifest,
    AcquisitionScoreRecord,
    SelectionManifest,
)


def scores() -> AcquisitionScoreManifest:
    return AcquisitionScoreManifest(
        score_id="scores_v1",
        source={"id": "pool_v1", "sha256": "a" * 64},
        records=(
            AcquisitionScoreRecord(record_id="r1", parent_id="p1", scores={"force_rms": 1.0}),
            AcquisitionScoreRecord(record_id="r2", parent_id="p2", scores={"force_rms": 2.0}),
        ),
    )


class AcquisitionSchemaTests(unittest.TestCase):
    def test_score_manifest_round_trip(self) -> None:
        original = scores()
        self.assertEqual(
            AcquisitionScoreManifest.from_dict(original.to_dict()).to_dict(),
            original.to_dict(),
        )

    def test_duplicate_record_and_nonfinite_score_fail(self) -> None:
        record = scores().records[0]
        with self.assertRaises(ValidationError):
            AcquisitionScoreManifest(
                score_id="bad", source={"id": "pool", "sha256": "b" * 64}, records=(record, record)
            )
        with self.assertRaises(ValidationError):
            AcquisitionScoreRecord(record_id="bad", parent_id="p", scores={"x": float("nan")})

    def test_selection_union_and_config_hash_are_rederived(self) -> None:
        config = {"selection_id": "sel_v1"}
        original = SelectionManifest(
            selection_id="sel_v1",
            source={"id": "scores_v1", "sha256": "c" * 64},
            config_sha256=canonical_json_fingerprint(config),
            config=config,
            policy_selections={"high": ("r2",), "random": ("r1",)},
            union_record_ids=("r1", "r2"),
        )
        self.assertEqual(
            SelectionManifest.from_dict(original.to_dict()).to_dict(), original.to_dict()
        )
        tampered = deepcopy(original.to_dict())
        tampered["union_record_ids"] = ["r1"]
        with self.assertRaises(ValidationError):
            SelectionManifest.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
