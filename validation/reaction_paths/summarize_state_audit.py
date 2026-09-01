#!/usr/bin/env python
"""Summarize paired doublet/quartet GPU4PySCF labels without approving a state."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from uma_pyscf.core.io import read_json, write_json_atomic
from uma_pyscf.schemas.label_record import LabelRecord

HARTREE_TO_EV = 27.211386245988
REVIEW_S2_MAX_ABS_DEVIATION = 0.05


def run(records_dir: Path, output: Path) -> None:
    grouped: dict[str, dict[int, LabelRecord]] = defaultdict(dict)
    for path in sorted(records_dir.glob("*.json")):
        record = LabelRecord.from_dict(read_json(path))
        parent = record.structure.parent_structure_id
        if parent is None:
            raise ValueError(f"{record.record_id} has no parent_structure_id")
        if record.state.multiplicity not in (2, 4):
            raise ValueError(f"{record.record_id} is not a doublet/quartet audit record")
        if record.state.multiplicity in grouped[parent]:
            raise ValueError(f"{parent} repeats multiplicity {record.state.multiplicity}")
        grouped[parent][record.state.multiplicity] = record
    if not grouped:
        raise ValueError(f"{records_dir} contains no label records")

    rows: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parent, states in sorted(grouped.items()):
        if set(states) != {2, 4}:
            raise ValueError(f"{parent} does not have exactly one doublet and one quartet")
        doublet = states[2]
        quartet = states[4]
        doublet_s2 = doublet.results.s2
        quartet_s2 = quartet.results.s2
        if doublet_s2 is None or quartet_s2 is None:
            raise ValueError(f"{parent} is missing open-shell S2")
        gap_ev = (quartet.results.energy_hartree - doublet.results.energy_hartree) * HARTREE_TO_EV
        source_id = parent.split("_mm_x", maxsplit=1)[0]
        row = {
            "parent_structure_id": parent,
            "source_id": source_id,
            "quartet_minus_doublet_ev": gap_ev,
            "doublet_lower": gap_ev > 0.0,
            "doublet": {
                "record_id": doublet.record_id,
                "energy_hartree": doublet.results.energy_hartree,
                "s2": doublet_s2,
                "s2_target": doublet.results.s2_target,
                "s2_deviation": doublet.results.s2_deviation,
                "converged": doublet.results.converged,
            },
            "quartet": {
                "record_id": quartet.record_id,
                "energy_hartree": quartet.results.energy_hartree,
                "s2": quartet_s2,
                "s2_target": quartet.results.s2_target,
                "s2_deviation": quartet.results.s2_deviation,
                "converged": quartet.results.converged,
            },
        }
        row["review_ready"] = bool(
            row["doublet_lower"]
            and doublet.results.converged
            and quartet.results.converged
            and doublet.results.s2_deviation is not None
            and quartet.results.s2_deviation is not None
            and abs(doublet.results.s2_deviation) <= REVIEW_S2_MAX_ABS_DEVIATION
            and abs(quartet.results.s2_deviation) <= REVIEW_S2_MAX_ABS_DEVIATION
        )
        rows.append(row)
        by_source[source_id].append(row)

    source_summary = {
        source_id: {
            "geometry_count": len(source_rows),
            "all_geometries_doublet_lower": all(row["doublet_lower"] for row in source_rows),
            "all_geometries_review_ready": all(row["review_ready"] for row in source_rows),
            "minimum_quartet_minus_doublet_ev": min(
                row["quartet_minus_doublet_ev"] for row in source_rows
            ),
        }
        for source_id, source_rows in sorted(by_source.items())
    }
    summary = {
        "schema": "uma-pyscf-state-audit-summary-v1",
        "decision": "review_only_not_state_approval",
        "s2_review_threshold": REVIEW_S2_MAX_ABS_DEVIATION,
        "record_count": 2 * len(rows),
        "geometry_count": len(rows),
        "sources": source_summary,
        "geometries": rows,
    }
    write_json_atomic(output, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.records_dir, arguments.output)
