"""Deterministic execution sharding for multi-GPU label batches."""

from __future__ import annotations

import unittest

from uma_pyscf.calculators.sharding import shard_candidate_manifest
from uma_pyscf.core.errors import ValidationError
from uma_pyscf.core.ids import canonical_json_fingerprint
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord
from uma_pyscf.schemas.label_record import ElectronicState, Structure


def candidate(record_id: str, numbers: tuple[int, ...]) -> CandidateRecord:
    return CandidateRecord(
        record_id=record_id,
        structure=Structure(
            atomic_numbers=numbers,
            positions_angstrom=tuple(
                (float(index), 0.0, 0.0) for index in range(len(numbers))
            ),
            parent_structure_id=f"{record_id}_parent",
            sampling_method="unit_fixture",
        ),
        state=ElectronicState(
            charge=0,
            multiplicity=1 if sum(numbers) % 2 == 0 else 2,
            spin_2s=0 if sum(numbers) % 2 == 0 else 1,
            state_provenance="unit_fixture",
        ),
    )


def manifest(records: tuple[CandidateRecord, ...]) -> CandidateManifest:
    config = {"fixture": "execution_sharding"}
    return CandidateManifest(
        sampling_id="sharding_fixture_v1",
        config_sha256=canonical_json_fingerprint(config),
        config=config,
        records=records,
    )


class ShardingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = manifest(
            (
                candidate("h2", (1, 1)),
                candidate("sih4", (14, 1, 1, 1, 1)),
                candidate("geh4", (32, 1, 1, 1, 1)),
                candidate("sicl4", (14, 17, 17, 17, 17)),
                candidate("gecl4", (32, 17, 17, 17, 17)),
                candidate("sigeh6", (14, 32, 1, 1, 1, 1, 1, 1)),
            )
        )

    def test_shards_cover_every_record_once_and_are_reproducible(self) -> None:
        first = [
            shard_candidate_manifest(self.manifest, shard_index=index, shard_count=3)
            for index in range(3)
        ]
        second = [
            shard_candidate_manifest(self.manifest, shard_index=index, shard_count=3)
            for index in range(3)
        ]
        first_ids = [record.record_id for shard in first for record in shard.records]
        expected_ids = sorted(record.record_id for record in self.manifest.records)
        self.assertEqual(sorted(first_ids), expected_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertEqual(
            [shard.to_dict() for shard in first],
            [shard.to_dict() for shard in second],
        )
        self.assertTrue(all(shard.records for shard in first))

    def test_assignment_is_independent_of_manifest_record_order(self) -> None:
        reversed_manifest = manifest(tuple(reversed(self.manifest.records)))
        expected = shard_candidate_manifest(self.manifest, shard_index=1, shard_count=3)
        actual = shard_candidate_manifest(reversed_manifest, shard_index=1, shard_count=3)
        self.assertEqual(
            [record.record_id for record in expected.records],
            [record.record_id for record in actual.records],
        )

    def test_invalid_or_empty_shards_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "exceeds"):
            shard_candidate_manifest(self.manifest, shard_index=0, shard_count=7)
        with self.assertRaisesRegex(ValidationError, "shard_index"):
            shard_candidate_manifest(self.manifest, shard_index=3, shard_count=3)


if __name__ == "__main__":
    unittest.main()
