#!/usr/bin/env python
"""Build PFP-versus-UMA acquisition scores without reading HF result fields."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import fsum, sqrt
from pathlib import Path
import sys
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _composition(numbers: list[int]) -> str:
    return ";".join(f"{number}:{count}" for number, count in sorted(Counter(numbers).items()))


def _vector_norm(values: list[float]) -> float:
    return sqrt(fsum(float(value) ** 2 for value in values))


def _force_scores(
    uma_forces: list[list[float]], pfp_forces: list[list[float]]
) -> tuple[float, float, float]:
    if len(uma_forces) != len(pfp_forces) or not uma_forces:
        raise ValueError("UMA/PFP force row counts differ or are empty")
    component_differences: list[float] = []
    atom_differences: list[float] = []
    angular_differences: list[float] = []
    for uma_row, pfp_row in zip(uma_forces, pfp_forces, strict=True):
        if len(uma_row) != 3 or len(pfp_row) != 3:
            raise ValueError("force rows must contain three components")
        delta = [float(pfp) - float(uma) for uma, pfp in zip(uma_row, pfp_row, strict=True)]
        component_differences.extend(delta)
        atom_differences.append(_vector_norm(delta))
        uma_norm = _vector_norm(uma_row)
        pfp_norm = _vector_norm(pfp_row)
        if uma_norm > 1.0e-12 and pfp_norm > 1.0e-12:
            cosine = fsum(float(a) * float(b) for a, b in zip(uma_row, pfp_row, strict=True))
            cosine = max(-1.0, min(1.0, cosine / (uma_norm * pfp_norm)))
            angular_differences.append(1.0 - cosine)
    rms = sqrt(fsum(value * value for value in component_differences) / len(component_differences))
    maximum = max(atom_differences)
    angular = fsum(angular_differences) / len(angular_differences) if angular_differences else 0.0
    return rms, maximum, angular


def _rank_fraction(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda record_id: (values[record_id], record_id))
    denominator = max(1, len(ordered) - 1)
    return {record_id: rank / denominator for rank, record_id in enumerate(ordered)}


def build_score_manifest(
    candidates: dict[str, Any], uma_evaluation: dict[str, Any], pfp_records: dict[str, Any]
) -> dict[str, Any]:
    if candidates.get("schema") != "uma-pyscf-candidate-manifest-v1":
        raise ValueError("candidate input has the wrong schema")
    uma_schema = uma_evaluation.get("schema")
    if uma_schema == "uma-pyscf-uma-evaluation-v1":
        uma_prediction_id = str(uma_evaluation["evaluation_id"])
        uma_model = str(uma_evaluation["model"]["name"])
        uma_by_id = {
            str(record["record_id"]): {
                "energy_ev": record["predicted_energy_ev"],
                "forces_ev_per_angstrom": record["predicted_forces_ev_per_angstrom"],
            }
            for record in uma_evaluation.get("predictions", [])
        }
    elif uma_schema == "uma-pyscf-model-predictions-v1":
        uma_prediction_id = str(uma_evaluation["prediction_id"])
        uma_model = str(uma_evaluation["model"]["name"])
        uma_by_id = {
            str(record["record_id"]): {
                "energy_ev": record["results"]["energy_ev"],
                "forces_ev_per_angstrom": record["results"]["forces_ev_per_angstrom"],
            }
            for record in uma_evaluation.get("records", [])
        }
    else:
        raise ValueError("UMA input has neither a supported evaluation nor prediction schema")
    candidate_by_id = {
        str(record["record_id"]): record for record in candidates.get("records", [])
    }
    if not candidate_by_id or set(candidate_by_id) != set(uma_by_id):
        raise ValueError("candidate and UMA record ids do not match exactly")
    if set(candidate_by_id) != set(pfp_records):
        raise ValueError("candidate and PFP record ids do not match exactly")

    raw: dict[str, dict[str, Any]] = {}
    energy_delta_by_composition: dict[str, list[float]] = defaultdict(list)
    for record_id in sorted(candidate_by_id):
        candidate = candidate_by_id[record_id]
        uma = uma_by_id[record_id]
        pfp = pfp_records[record_id]
        if pfp.get("schema") != "uma-pyscf-pfp-single-point-v1":
            raise ValueError(f"PFP record {record_id!r} has the wrong schema")
        structure = candidate["structure"]
        state = candidate["state"]
        pfp_input = pfp["input"]
        if int(state["charge"]) != 0 or int(state["multiplicity"]) != 1:
            raise ValueError(f"{record_id!r} is outside the neutral-singlet dry-run scope")
        if list(structure["atomic_numbers"]) != list(pfp_input["atomic_numbers"]):
            raise ValueError(f"atomic numbers differ for {record_id!r}")
        if structure["positions_angstrom"] != pfp_input["positions_angstrom"]:
            maximum_delta = max(
                abs(float(left) - float(right))
                for left_row, right_row in zip(
                    structure["positions_angstrom"],
                    pfp_input["positions_angstrom"],
                    strict=True,
                )
                for left, right in zip(left_row, right_row, strict=True)
            )
            if maximum_delta > 1.0e-12:
                raise ValueError(f"positions differ for {record_id!r}")
        force_rms, force_max_atom, force_angular = _force_scores(
            uma["forces_ev_per_angstrom"],
            pfp["results"]["forces_ev_per_angstrom"],
        )
        energy_delta = float(pfp["results"]["energy_ev"]) - float(uma["energy_ev"])
        composition = _composition(list(structure["atomic_numbers"]))
        energy_delta_by_composition[composition].append(energy_delta)
        raw[record_id] = {
            "parent_id": str(structure["parent_structure_id"]),
            "trajectory_id": candidate.get("generation_parameters", {}).get("trajectory_id"),
            "frame_index": candidate.get("generation_parameters", {}).get("frame_index"),
            "composition": composition,
            "energy_delta": energy_delta,
            "pfp_uma_force_rms": force_rms,
            "pfp_uma_force_max_atom": force_max_atom,
            "pfp_uma_force_angular": force_angular,
            "pfp_model": pfp["model"],
            "pfp_runtime": pfp["provenance"]["runtime_versions"],
        }

    energy_means = {
        composition: fsum(values) / len(values)
        for composition, values in energy_delta_by_composition.items()
    }
    metrics = (
        "pfp_uma_force_rms",
        "pfp_uma_force_max_atom",
        "pfp_uma_force_angular",
        "pfp_uma_energy_centered_abs",
    )
    for values in raw.values():
        values["pfp_uma_energy_centered_abs"] = abs(
            float(values["energy_delta"]) - energy_means[str(values["composition"])]
        )
    rank_by_metric = {
        metric: _rank_fraction(
            {record_id: float(values[metric]) for record_id, values in raw.items()}
        )
        for metric in metrics
    }

    records: list[dict[str, Any]] = []
    for record_id, values in raw.items():
        scores = {metric: float(values[metric]) for metric in metrics}
        scores["pfp_uma_combined_rank"] = fsum(
            rank_by_metric[metric][record_id] for metric in metrics
        ) / len(metrics)
        records.append(
            {
                "record_id": record_id,
                "parent_id": values["parent_id"],
                "trajectory_id": values["trajectory_id"],
                "frame_index": values["frame_index"],
                "scores": scores,
                "provenance": {
                    "uma_prediction_id": uma_prediction_id,
                    "uma_model": uma_model,
                    "pfp_model": values["pfp_model"],
                    "pfp_runtime": values["pfp_runtime"],
                    "reference_fields_used": False,
                },
            }
        )
    source_sha256 = hashlib.sha256(
        json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "uma-pyscf-acquisition-scores-v1",
        "score_id": f"{candidates['sampling_id']}_pfp_uma_scores_v1",
        "source": {"id": candidates["sampling_id"], "sha256": source_sha256},
        "records": records,
    }


def _load_pfp_records(path: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for source in sorted(path.glob("*.json")):
        record = _read_json(source)
        record_id = str(record["record_id"])
        if record_id in records:
            raise ValueError(f"duplicate PFP record id {record_id!r}")
        records[record_id] = record
    if not records:
        raise ValueError(f"{path} contains no PFP records")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--uma-predictions",
        "--uma-evaluation",
        dest="uma_predictions",
        type=Path,
        required=True,
        help="Unlabeled model-prediction manifest or legacy engineering evaluation.",
    )
    parser.add_argument("--pfp-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build_score_manifest(
            _read_json(args.candidates),
            _read_json(args.uma_predictions),
            _load_pfp_records(args.pfp_records),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"records={len(manifest['records'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
