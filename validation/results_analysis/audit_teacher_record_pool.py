#!/usr/bin/env python3
"""Audit a multi-source label pool and emit a deterministic eligibility ledger.

QC acceptance is necessary but not sufficient: explicit scientific quarantine
entries win over QC, and state-qualified duplicate geometries are represented
once.  Explicit rejected-record overrides are recorded as non-global,
record-specific classifications (for example ``valid_high_energy``).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _key_value(raw: str, flag: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"{flag} requires NAME=VALUE; got {raw!r}")
    key, value = raw.split("=", 1)
    if not key or not value:
        raise ValueError(f"{flag} requires non-empty NAME=VALUE; got {raw!r}")
    return key, value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_fingerprint(record: dict[str, Any], decimals: int) -> str:
    numbers = [int(value) for value in record["structure"]["atomic_numbers"]]
    positions = record["structure"]["positions_angstrom"]
    pairs: list[tuple[int, int, float]] = []
    for left in range(len(numbers)):
        for right in range(left + 1, len(numbers)):
            distance = math.dist(positions[left], positions[right])
            pairs.append(
                (
                    min(numbers[left], numbers[right]),
                    max(numbers[left], numbers[right]),
                    round(distance, decimals),
                )
            )
    state = record["state"]
    canonical = json.dumps(
        {
            "composition": sorted(Counter(numbers).items()),
            "pairs": sorted(pairs),
            "charge": state["charge"],
            "multiplicity": state["multiplicity"],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def audit(
    sources: list[tuple[str, str]],
    quarantine: dict[str, str],
    include_rejected: dict[str, str],
    fingerprint_decimals: int,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    by_record_id: dict[str, list[int]] = defaultdict(list)
    source_matches: dict[str, int] = {}
    for source_order, (source, pattern) in enumerate(sources):
        paths = [Path(name) for name in sorted(glob.glob(pattern, recursive=True))]
        source_matches[source] = len(paths)
        for path in paths:
            raw = path.read_bytes()
            record = json.loads(raw)
            record_id = str(record["record_id"])
            qc_status = str(record.get("qc", {}).get("status", "missing"))
            entry = {
                "record_id": record_id,
                "source": source,
                "source_order": source_order,
                "path": str(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_label_sha256": record.get("raw", {}).get("checksum_sha256"),
                "qc_status": qc_status,
                "parent_structure_id": record["structure"]["parent_structure_id"],
                "sampling_method": record["structure"]["sampling_method"],
                "charge": record["state"]["charge"],
                "multiplicity": record["state"]["multiplicity"],
                "geometry_fingerprint_sha256": _geometry_fingerprint(
                    record, fingerprint_decimals
                ),
                "classification": None,
                "eligible": False,
                "reason": None,
            }
            entries.append(entry)
            by_record_id[record_id].append(len(entries) - 1)

    for entry in entries:
        record_id = entry["record_id"]
        if record_id in quarantine:
            entry["classification"] = "quarantined"
            entry["reason"] = quarantine[record_id]
        elif entry["qc_status"] == "accepted":
            entry["classification"] = "qc_accepted"
            entry["eligible"] = True
            entry["reason"] = "accepted_by_record_qc"
        elif record_id in include_rejected:
            entry["classification"] = include_rejected[record_id]
            entry["eligible"] = True
            entry["reason"] = "explicit_record_specific_override"
        else:
            entry["classification"] = "qc_not_accepted"
            entry["reason"] = f"qc_status={entry['qc_status']}"

    duplicate_record_ids: list[dict[str, Any]] = []
    for record_id, indices in sorted(by_record_id.items()):
        if len(indices) < 2:
            continue
        members = [entries[index] for index in indices]
        duplicate_record_ids.append(
            {
                "record_id": record_id,
                "sources": [member["source"] for member in members],
                "paths": [member["path"] for member in members],
                "identical_files": len({member["file_sha256"] for member in members}) == 1,
            }
        )
        for duplicate in members[1:]:
            duplicate["eligible"] = False
            duplicate["classification"] = "duplicate_record_id"
            duplicate["reason"] = f"represented_by={members[0]['path']}"

    by_geometry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry["eligible"]:
            by_geometry[entry["geometry_fingerprint_sha256"]].append(entry)
    duplicate_geometries: list[dict[str, Any]] = []
    for fingerprint, members in sorted(by_geometry.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda item: (item["source_order"], item["record_id"]))
        representative = ordered[0]
        duplicate_geometries.append(
            {
                "geometry_fingerprint_sha256": fingerprint,
                "representative_record_id": representative["record_id"],
                "record_ids": [member["record_id"] for member in ordered],
                "sources": [member["source"] for member in ordered],
            }
        )
        for duplicate in ordered[1:]:
            duplicate["eligible"] = False
            duplicate["classification"] = "duplicate_geometry"
            duplicate["reason"] = f"represented_by={representative['record_id']}"

    eligible = sorted(
        (entry for entry in entries if entry["eligible"]), key=lambda item: item["record_id"]
    )
    parent_sources: dict[str, set[str]] = defaultdict(set)
    for entry in eligible:
        parent_sources[entry["parent_structure_id"]].add(entry["source"])
    parent_overlaps = [
        {"parent_structure_id": parent, "sources": sorted(source_names)}
        for parent, source_names in sorted(parent_sources.items())
        if len(source_names) > 1
    ]
    counts_by_source: dict[str, dict[str, int]] = {}
    for source, _ in sources:
        selected = [entry for entry in entries if entry["source"] == source]
        counts_by_source[source] = {
            "matched": len(selected),
            "qc_accepted": sum(entry["qc_status"] == "accepted" for entry in selected),
            "eligible": sum(entry["eligible"] for entry in selected),
            "quarantined": sum(entry["classification"] == "quarantined" for entry in selected),
        }
    return {
        "schema": "uma-pyscf-teacher-pool-audit-v1",
        "fingerprint_distance_decimals": fingerprint_decimals,
        "source_patterns": dict(sources),
        "source_matches": source_matches,
        "quarantine": dict(sorted(quarantine.items())),
        "explicit_rejected_record_overrides": dict(sorted(include_rejected.items())),
        "counts": {
            "files": len(entries),
            "qc_accepted": sum(entry["qc_status"] == "accepted" for entry in entries),
            "eligible_unique": len(eligible),
            "quarantined": sum(
                entry["classification"] == "quarantined" for entry in entries
            ),
            "duplicate_record_id_groups": len(duplicate_record_ids),
            "duplicate_geometry_groups": len(duplicate_geometries),
            "parent_source_overlap_groups": len(parent_overlaps),
        },
        "counts_by_source": counts_by_source,
        "duplicate_record_ids": duplicate_record_ids,
        "duplicate_geometries": duplicate_geometries,
        "parent_source_overlaps": parent_overlaps,
        "eligible_records": [
            {
                key: entry[key]
                for key in (
                    "record_id",
                    "source",
                    "path",
                    "file_sha256",
                    "raw_label_sha256",
                    "parent_structure_id",
                    "sampling_method",
                    "charge",
                    "multiplicity",
                    "classification",
                )
            }
            for entry in eligible
        ],
        "entries": sorted(entries, key=lambda item: (item["record_id"], item["source_order"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=GLOB")
    parser.add_argument("--quarantine", action="append", default=[], metavar="ID=REASON")
    parser.add_argument(
        "--include-rejected", action="append", default=[], metavar="ID=CLASSIFICATION"
    )
    parser.add_argument("--fingerprint-decimals", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = [_key_value(raw, "--source") for raw in args.source]
    if len({name for name, _ in sources}) != len(sources):
        raise ValueError("--source names must be unique")
    payload = audit(
        sources,
        dict(_key_value(raw, "--quarantine") for raw in args.quarantine),
        dict(_key_value(raw, "--include-rejected") for raw in args.include_rejected),
        args.fingerprint_decimals,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], sort_keys=True))
    print(json.dumps(payload["counts_by_source"], sort_keys=True))
    print(f"sha256={_file_sha256(args.output)}")


if __name__ == "__main__":
    main()
