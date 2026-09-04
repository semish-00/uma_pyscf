#!/usr/bin/env python3
"""Compare two deterministic sets of label records by record ID.

This is intended for protocol sentinels such as density-fit versus direct SCF.
It does not change QC state or decide whether an electronic state is suitable
for production training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
GRADIENT_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def _load_records(pattern: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    records: dict[str, dict[str, Any]] = {}
    checksums: dict[str, str] = {}
    for name in sorted(glob.glob(pattern, recursive=True)):
        path = Path(name)
        raw = path.read_bytes()
        payload = json.loads(raw)
        record_id = payload["record_id"]
        if record_id in records:
            raise ValueError(f"duplicate record_id {record_id!r} in {pattern!r}")
        records[record_id] = payload
        checksums[record_id] = hashlib.sha256(raw).hexdigest()
    if not records:
        raise ValueError(f"no JSON records matched {pattern!r}")
    return records, checksums


def _flatten_gradient(record: dict[str, Any]) -> list[float]:
    return [
        float(component)
        for vector in record["results"]["gradient_hartree_per_bohr"]
        for component in vector
    ]


def _record_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["structure"] != candidate["structure"]:
        raise ValueError(f"structure mismatch for {reference['record_id']!r}")
    if reference["state"] != candidate["state"]:
        raise ValueError(f"state mismatch for {reference['record_id']!r}")
    reference_gradient = _flatten_gradient(reference)
    candidate_gradient = _flatten_gradient(candidate)
    if len(reference_gradient) != len(candidate_gradient):
        raise ValueError(f"gradient shape mismatch for {reference['record_id']!r}")
    gradient_differences = [
        candidate_value - reference_value
        for reference_value, candidate_value in zip(
            reference_gradient, candidate_gradient, strict=True
        )
    ]
    gradient_rmse = math.sqrt(
        math.fsum(value * value for value in gradient_differences)
        / len(gradient_differences)
    )
    gradient_max = max(abs(value) for value in gradient_differences)
    energy_difference = (
        float(candidate["results"]["energy_hartree"])
        - float(reference["results"]["energy_hartree"])
    )
    reference_s2 = reference["results"].get("s2")
    candidate_s2 = candidate["results"].get("s2")
    return {
        "record_id": reference["record_id"],
        "parent_structure_id": reference["structure"]["parent_structure_id"],
        "charge": reference["state"]["charge"],
        "multiplicity": reference["state"]["multiplicity"],
        "reference_converged": reference["results"]["converged"],
        "candidate_converged": candidate["results"]["converged"],
        "energy_difference_hartree": energy_difference,
        "absolute_energy_difference_mev": abs(energy_difference) * HARTREE_TO_EV * 1000,
        "gradient_rmse_hartree_per_bohr": gradient_rmse,
        "gradient_rmse_ev_per_angstrom": gradient_rmse * GRADIENT_TO_EV_PER_ANGSTROM,
        "gradient_max_component_hartree_per_bohr": gradient_max,
        "gradient_max_component_ev_per_angstrom": (
            gradient_max * GRADIENT_TO_EV_PER_ANGSTROM
        ),
        "s2_difference": (
            None
            if reference_s2 is None or candidate_s2 is None
            else float(candidate_s2) - float(reference_s2)
        ),
    }


def compare(reference_pattern: str, candidate_pattern: str) -> dict[str, Any]:
    reference, reference_checksums = _load_records(reference_pattern)
    candidate, candidate_checksums = _load_records(candidate_pattern)
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    shared = sorted(set(reference) & set(candidate))
    rows = [_record_metrics(reference[record_id], candidate[record_id]) for record_id in shared]

    state_groups: dict[str, list[str]] = defaultdict(list)
    for record_id in shared:
        state_groups[reference[record_id]["structure"]["parent_structure_id"]].append(record_id)
    orderings: list[dict[str, Any]] = []
    for parent_id, record_ids in sorted(state_groups.items()):
        by_multiplicity = {
            reference[record_id]["state"]["multiplicity"]: record_id for record_id in record_ids
        }
        if set(by_multiplicity) != {2, 4}:
            continue
        doublet_id = by_multiplicity[2]
        quartet_id = by_multiplicity[4]
        reference_gap = (
            reference[quartet_id]["results"]["energy_hartree"]
            - reference[doublet_id]["results"]["energy_hartree"]
        )
        candidate_gap = (
            candidate[quartet_id]["results"]["energy_hartree"]
            - candidate[doublet_id]["results"]["energy_hartree"]
        )
        orderings.append(
            {
                "parent_structure_id": parent_id,
                "quartet_minus_doublet_reference_hartree": reference_gap,
                "quartet_minus_doublet_candidate_hartree": candidate_gap,
                "gap_difference_mev": (
                    (candidate_gap - reference_gap) * HARTREE_TO_EV * 1000
                ),
                "ordering_preserved": (reference_gap == 0 and candidate_gap == 0)
                or (reference_gap * candidate_gap > 0),
            }
        )

    def maximum(field: str) -> float | None:
        return max((float(row[field]) for row in rows), default=None)

    s2_values = [abs(float(row["s2_difference"])) for row in rows if row["s2_difference"]]
    return {
        "schema": "uma-pyscf-label-set-comparison-v1",
        "reference_pattern": reference_pattern,
        "candidate_pattern": candidate_pattern,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "shared_count": len(shared),
        "missing_candidate_record_ids": missing,
        "unexpected_candidate_record_ids": unexpected,
        "all_shared_converged": all(
            row["reference_converged"] and row["candidate_converged"] for row in rows
        ),
        "summary": {
            "max_absolute_energy_difference_mev": maximum(
                "absolute_energy_difference_mev"
            ),
            "max_gradient_rmse_ev_per_angstrom": maximum(
                "gradient_rmse_ev_per_angstrom"
            ),
            "max_gradient_component_ev_per_angstrom": maximum(
                "gradient_max_component_ev_per_angstrom"
            ),
            "max_absolute_s2_difference": max(s2_values, default=0.0),
            "state_pair_count": len(orderings),
            "all_state_orderings_preserved": all(
                row["ordering_preserved"] for row in orderings
            ),
        },
        "record_file_sha256": {
            "reference": reference_checksums,
            "candidate": candidate_checksums,
        },
        "state_orderings": orderings,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Reference JSON glob")
    parser.add_argument("--candidate", required=True, help="Candidate JSON glob")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
