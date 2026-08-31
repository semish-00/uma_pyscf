#!/usr/bin/env python
"""Evaluate UMA-PySCF candidate geometries with a pinned PFP model.

This script is intentionally self-contained and lives outside ``src`` because
``pfp_api_client`` is available only inside Matlantis.  Its records are
multi-fidelity comparison artifacts, not canonical GPU4PySCF teacher labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

PROTOCOL_SCHEMA = "uma-pyscf-pfp-protocol-v1"
RECORD_SCHEMA = "uma-pyscf-pfp-single-point-v1"
RUN_SCHEMA = "uma-pyscf-pfp-run-v1"
SUMMARY_SCHEMA = "uma-pyscf-pfp-summary-v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        scratch.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"{path} is not a {PROTOCOL_SCHEMA} protocol")
    required = {
        "schema",
        "model_version",
        "calc_mode",
        "max_retries",
        "required_charge",
        "required_multiplicity",
    }
    unknown = set(protocol) - required
    missing = required - set(protocol)
    if unknown or missing:
        raise ValueError(
            f"protocol fields differ: missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    version = str(protocol["model_version"])
    if not version or version.lower() == "latest":
        raise ValueError("model_version must be pinned and cannot be 'latest'")
    if int(protocol["max_retries"]) < 1:
        raise ValueError("max_retries must be positive")
    return protocol


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "uma-pyscf-candidate-manifest-v1"
    ):
        raise ValueError(f"{path} is not an uma-pyscf-candidate-manifest-v1 manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("candidate manifest has no records")
    return manifest


def _selected_records(manifest: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    records = list(manifest["records"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        records = records[:limit]
    return records


def _validate_candidate(candidate: dict[str, Any], protocol: dict[str, Any]) -> None:
    state = candidate["state"]
    charge = int(state["charge"])
    multiplicity = int(state["multiplicity"])
    if charge != int(protocol["required_charge"]):
        raise ValueError(f"charge {charge} is outside this PFP comparison protocol")
    if multiplicity != int(protocol["required_multiplicity"]):
        raise ValueError(
            f"multiplicity {multiplicity} is outside this state-controlled comparison protocol"
        )
    structure = candidate["structure"]
    numbers = structure["atomic_numbers"]
    positions = structure["positions_angstrom"]
    if not numbers or len(numbers) != len(positions):
        raise ValueError("atomic_numbers and positions_angstrom do not match")


def _run_identity(
    manifest: dict[str, Any], protocol: dict[str, Any], selected: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "sampling_id": manifest["sampling_id"],
        "candidate_manifest_sha256": _fingerprint(manifest),
        "protocol": protocol,
        "protocol_sha256": _fingerprint(protocol),
        "record_ids": [candidate["record_id"] for candidate in selected],
    }


def _prepare_output(output_dir: Path, identity: dict[str, Any]) -> None:
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        existing = _read_json(identity_path)
        if existing != identity:
            raise ValueError(
                f"{output_dir} belongs to different inputs; use a new output directory"
            )
    else:
        _write_json_atomic(identity_path, identity)


def _runtime_versions() -> dict[str, Any]:
    import ase
    import pfp_api_client

    return {
        "ase": getattr(ase, "__version__", None),
        "pfp_api_client": getattr(pfp_api_client, "__version__", None),
        "python": platform.python_version(),
    }


def _evaluate(
    candidate: dict[str, Any], protocol: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    from ase import Atoms
    import numpy as np
    from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
    from pfp_api_client.pfp.estimator import Estimator

    _validate_candidate(candidate, protocol)
    structure = candidate["structure"]
    atoms = Atoms(
        numbers=structure["atomic_numbers"],
        positions=structure["positions_angstrom"],
        pbc=False,
    )
    estimator = Estimator(
        model_version=protocol["model_version"],
        calc_mode=protocol["calc_mode"],
        max_retries=int(protocol["max_retries"]),
    )
    atoms.calc = ASECalculator(estimator)
    started = time.perf_counter()
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    elapsed = time.perf_counter() - started
    calc_stats = atoms.calc.results.get("calc_stats")
    return {
        "schema": RECORD_SCHEMA,
        "record_id": candidate["record_id"],
        "input": {
            "atomic_numbers": list(structure["atomic_numbers"]),
            "positions_angstrom": [list(row) for row in structure["positions_angstrom"]],
            "charge": int(candidate["state"]["charge"]),
            "multiplicity": int(candidate["state"]["multiplicity"]),
        },
        "model": {
            "name": "PFP",
            "model_version": protocol["model_version"],
            "calc_mode": protocol["calc_mode"],
        },
        "results": {
            "energy_ev": energy,
            "forces_ev_per_angstrom": forces.tolist(),
            "max_force_ev_per_angstrom": float(np.linalg.norm(forces, axis=1).max()),
            "wall_time_seconds": elapsed,
            "calc_stats": _jsonable(calc_stats),
        },
        "provenance": {
            "candidate_manifest_sha256": identity["candidate_manifest_sha256"],
            "protocol_sha256": identity["protocol_sha256"],
            "runtime_versions": _runtime_versions(),
            "hostname": socket.gethostname(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def run(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    protocol = _load_protocol(args.config)
    selected = _selected_records(manifest, args.limit)
    for candidate in selected:
        _validate_candidate(candidate, protocol)
    identity = _run_identity(manifest, protocol, selected)
    _prepare_output(args.output_dir, identity)
    if args.dry_run:
        print(
            f"dry-run records={len(selected)} model={protocol['model_version']} "
            f"calc_mode={protocol['calc_mode']} output={args.output_dir}"
        )
        return 0

    records_dir = args.output_dir / "records"
    completed = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    for index, candidate in enumerate(selected, start=1):
        record_id = candidate["record_id"]
        destination = records_dir / f"{record_id}.json"
        if destination.exists():
            existing = _read_json(destination)
            if (
                existing.get("schema") != RECORD_SCHEMA
                or existing.get("record_id") != record_id
                or existing.get("provenance", {}).get("protocol_sha256")
                != identity["protocol_sha256"]
            ):
                raise ValueError(f"resume record is incompatible: {destination}")
            skipped += 1
            print(f"[{index}/{len(selected)}] skip {record_id}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] evaluate {record_id}", flush=True)
        try:
            record = _evaluate(candidate, protocol, identity)
            _write_json_atomic(destination, record)
            completed += 1
        except Exception as exc:  # preserve partial progress and report the exact failed id
            failed.append(
                {"record_id": record_id, "error_type": type(exc).__name__, "error": str(exc)}
            )
            print(
                f"[{index}/{len(selected)}] FAILED {record_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if not args.keep_going:
                break

    summary = {
        "schema": SUMMARY_SCHEMA,
        "run_identity": identity,
        "counts": {
            "selected": len(selected),
            "completed_this_pass": completed,
            "skipped_this_pass": skipped,
            "failed_this_pass": len(failed),
            "records_present": (
                len(list(records_dir.glob("*.json"))) if records_dir.exists() else 0
            ),
        },
        "failures": failed,
    }
    _write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["counts"], sort_keys=True), flush=True)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
