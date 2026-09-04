#!/usr/bin/env python3
"""Merge immutable candidate manifests by record ID and retain arm membership."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from uma_pyscf.core.ids import canonical_json_fingerprint, sha256_of_file
from uma_pyscf.core.io import read_json, write_json_atomic
from uma_pyscf.schemas.candidate import CandidateManifest, CandidateRecord


def merge(
    paths: list[Path], sampling_id: str
) -> tuple[CandidateManifest, dict[str, Any]]:
    inputs = []
    by_record_id: dict[str, CandidateRecord] = {}
    memberships: dict[str, list[str]] = {}
    for path in paths:
        manifest = CandidateManifest.from_dict(read_json(path))
        inputs.append(
            {
                "sampling_id": manifest.sampling_id,
                "path": str(path),
                "sha256": sha256_of_file(path),
                "record_count": len(manifest.records),
            }
        )
        for record in manifest.records:
            previous = by_record_id.get(record.record_id)
            if previous is not None and previous.to_dict() != record.to_dict():
                raise ValueError(
                    f"record {record.record_id!r} differs between candidate manifests"
                )
            by_record_id[record.record_id] = record
            memberships.setdefault(record.record_id, []).append(manifest.sampling_id)
    config = {
        "operation": "candidate_manifest_union_by_record_id",
        "inputs": inputs,
    }
    records = tuple(by_record_id[record_id] for record_id in sorted(by_record_id))
    manifest = CandidateManifest(
        sampling_id=sampling_id,
        config_sha256=canonical_json_fingerprint(config),
        config=config,
        records=records,
    )
    report = {
        "schema": "uma-pyscf-candidate-manifest-union-report-v1",
        "sampling_id": sampling_id,
        "inputs": inputs,
        "input_record_count": sum(item["record_count"] for item in inputs),
        "unique_record_count": len(records),
        "overlap_record_count": sum(len(arms) > 1 for arms in memberships.values()),
        "arm_membership": dict(sorted(memberships.items())),
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--sampling-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest, report = merge(args.manifest, args.sampling_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.sampling_id}_candidates.json"
    report_path = args.output_dir / f"{args.sampling_id}_union_report.json"
    write_json_atomic(manifest_path, manifest.to_dict())
    write_json_atomic(report_path, report)
    print(
        f"inputs={report['input_record_count']} unique={report['unique_record_count']} "
        f"overlap={report['overlap_record_count']}"
    )
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
