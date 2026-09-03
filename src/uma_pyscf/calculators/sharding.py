"""Deterministically balance one candidate manifest across GPU workers.

Execution shards are not scientific dataset splits.  They only decide which
GPU labels a candidate; parent/reaction grouping is still enforced later by
the dataset split manifest.
"""

from __future__ import annotations

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint
from ..schemas.candidate import CandidateManifest, CandidateRecord

__all__ = ["shard_candidate_manifest"]


def _estimated_cost(record: CandidateRecord) -> int:
    """Return a stable coarse GPU4PySCF cost proxy for load balancing."""
    atom_count = record.structure.atom_count
    electron_count = record.electron_count
    return atom_count * electron_count * electron_count


def shard_candidate_manifest(
    manifest: CandidateManifest, *, shard_index: int, shard_count: int
) -> CandidateManifest:
    """Return one deterministic, cost-balanced execution shard.

    Candidates are sorted by descending coarse cost and assigned to the worker
    with the smallest accumulated cost.  Worker index breaks load ties.  This
    keeps every record on exactly one worker while remaining independent of
    input order and wall-clock state.
    """
    if not isinstance(manifest, CandidateManifest):
        raise ValidationError(
            f"manifest must be CandidateManifest; got {type(manifest).__name__}."
        )
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValidationError("shard_count must be a positive integer.")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise ValidationError("shard_index must be an integer.")
    if not 0 <= shard_index < shard_count:
        raise ValidationError(
            f"shard_index must be in [0, {shard_count}); got {shard_index}."
        )
    if shard_count > len(manifest.records):
        raise ValidationError(
            f"shard_count={shard_count} exceeds the {len(manifest.records)} candidates; "
            "empty GPU workers are not allowed."
        )

    assignments: list[list[CandidateRecord]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    ordered = sorted(
        manifest.records,
        key=lambda record: (-_estimated_cost(record), record.record_id),
    )
    for record in ordered:
        worker = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[worker].append(record)
        loads[worker] += _estimated_cost(record)

    source_sha256 = canonical_json_fingerprint(manifest.to_dict())
    shard_config = {
        "kind": "execution_shard",
        "source_manifest": {
            "sampling_id": manifest.sampling_id,
            "sha256": source_sha256,
        },
        "assignment": "descending_atom_electron2_greedy_v1",
        "shard_count": shard_count,
        "shard_index": shard_index,
        "estimated_loads": loads,
    }
    return CandidateManifest(
        sampling_id=f"{manifest.sampling_id}_shard{shard_index:03d}of{shard_count:03d}",
        config_sha256=canonical_json_fingerprint(shard_config),
        config=shard_config,
        records=tuple(sorted(assignments[shard_index], key=lambda record: record.record_id)),
    )
