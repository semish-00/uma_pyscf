"""Resumable manifest labeling with atomic records and an attempt ledger."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, sha256_of_file
from ..core.io import read_json, write_json_atomic
from ..schemas._fields import require_bool, require_mapping, require_sequence, require_str
from ..schemas.candidate import CandidateManifest, CandidateRecord
from ..schemas.label_record import (
    ElectronicState,
    Engine,
    LabelRecord,
    QcState,
    RawArtifact,
)
from ..schemas.state_registry import StateRegistry
from ..states.registry import registry_identity
from .config import method_from_config, resource_for_candidate, scope_violations
from .model import CalculationFailure, CalculationOutput, CalculatorAdapter

__all__ = [
    "LABEL_EVENT",
    "LABEL_LEDGER_SCHEMA",
    "build_label_plan",
    "run_label_batch",
]

LABEL_LEDGER_SCHEMA = "uma-pyscf-label-ledger-v1"
LABEL_EVENT = "labeled_with_dft_protocol_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _protocol_id(config: Mapping[str, Any]) -> str:
    return require_str(config.get("protocol_id"), "config.protocol_id")


def _attempts(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    retry = require_mapping(config.get("retry"), "config.retry")
    values = require_sequence(retry.get("attempts"), "config.retry.attempts")
    return tuple(
        require_mapping(value, f"config.retry.attempts[{index}]")
        for index, value in enumerate(values)
    )


def _initial_guess(config: Mapping[str, Any]) -> tuple[str, str]:
    density = require_mapping(config.get("initial_density"), "config.initial_density")
    return (
        require_str(density.get("guess"), "config.initial_density.guess"),
        require_str(density.get("generated_on"), "config.initial_density.generated_on"),
    )


def _candidate_with_protocol_state(
    candidate: CandidateRecord, config: Mapping[str, Any]
) -> ElectronicState:
    guess, _ = _initial_guess(config)
    return ElectronicState(
        charge=candidate.state.charge,
        multiplicity=candidate.state.multiplicity,
        spin_2s=candidate.state.spin_2s,
        initial_guess=guess,
        state_provenance=candidate.state.state_provenance,
    )


def _validate_runtime(output: CalculationOutput, config: Mapping[str, Any]) -> None:
    engine = require_mapping(config.get("engine"), "config.engine")
    expected_name = require_str(engine.get("name"), "config.engine.name")
    if output.engine_name != expected_name:
        raise ValidationError(
            f"Calculator returned engine {output.engine_name!r}; protocol requires "
            f"{expected_name!r}."
        )
    expected_versions = require_mapping(
        engine.get("required_versions"), "config.engine.required_versions"
    )
    for name, expected in expected_versions.items():
        actual = output.engine_versions.get(name)
        if actual != expected:
            raise ValidationError(
                f"Runtime {name} version is {actual!r}; protocol requires {expected!r}."
            )
    required_gpu = require_str(engine.get("required_gpu_name"), "config.engine.required_gpu_name")
    if output.engine_versions.get("cuda_device_name") != required_gpu:
        raise ValidationError(
            "Runtime GPU is "
            f"{output.engine_versions.get('cuda_device_name')!r}; protocol requires "
            f"{required_gpu!r}."
        )
    provenance = require_mapping(config.get("provenance"), "config.provenance")
    required_keys = require_sequence(
        provenance.get("required_runtime_keys"), "config.provenance.required_runtime_keys"
    )
    missing = [
        require_str(key, "config.provenance.required_runtime_keys[]")
        for key in required_keys
        if require_str(key, "config.provenance.required_runtime_keys[]")
        not in output.engine_versions
    ]
    if missing:
        raise ValidationError(f"Runtime provenance is missing required keys {missing!r}.")


def _new_ledger(
    manifest: CandidateManifest,
    config: Mapping[str, Any],
    protocol_sha256: str,
    state_registry: StateRegistry | None,
) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema": LABEL_LEDGER_SCHEMA,
        "run_id": f"{manifest.sampling_id}__{_protocol_id(config)}",
        "sampling_id": manifest.sampling_id,
        "manifest_sha256": canonical_json_fingerprint(manifest.to_dict()),
        "protocol_id": _protocol_id(config),
        "protocol_sha256": protocol_sha256,
        "records": {},
    }
    identity = registry_identity(state_registry)
    if identity is not None:
        ledger["state_registry"] = identity
    return ledger


def _load_or_create_ledger(
    path: Path,
    manifest: CandidateManifest,
    config: Mapping[str, Any],
    protocol_sha256: str,
    state_registry: StateRegistry | None,
) -> dict[str, Any]:
    expected = _new_ledger(manifest, config, protocol_sha256, state_registry)
    if not path.exists():
        return expected
    ledger = require_mapping(read_json(path), "ledger")
    for key in (
        "schema",
        "run_id",
        "sampling_id",
        "manifest_sha256",
        "protocol_id",
        "protocol_sha256",
        "state_registry",
        "records",
    ):
        if ledger.get(key) != expected.get(key) and key != "records":
            raise ValidationError(
                f"Existing ledger {path} has {key}={ledger.get(key)!r}; expected "
                f"{expected.get(key)!r}. Use a new output directory for a different input."
            )
    require_mapping(ledger.get("records"), "ledger.records")
    return ledger


def _record_matches_protocol(path: Path, protocol_sha256: str) -> LabelRecord:
    record = LabelRecord.from_dict(read_json(path))
    matching = [
        entry
        for entry in record.qc.history
        if entry.get("event") == LABEL_EVENT
        and entry.get("protocol_sha256") == protocol_sha256
    ]
    if not matching:
        raise ValidationError(
            f"Resume record {path} does not carry {LABEL_EVENT} for protocol "
            f"{protocol_sha256}; refusing to skip it."
        )
    return record


def build_label_plan(
    manifest: CandidateManifest,
    config: Mapping[str, Any],
    *,
    state_registry: StateRegistry | None = None,
) -> dict[str, Any]:
    """Return a deterministic dry-run plan without importing PySCF."""
    protocol_sha256 = canonical_json_fingerprint(config)
    records: list[dict[str, Any]] = []
    for candidate in manifest.records:
        violations = scope_violations(candidate, config, state_registry)
        records.append(
            {
                "record_id": candidate.record_id,
                "status": "blocked" if violations else "ready",
                "scope_violations": list(violations),
                "resource": resource_for_candidate(candidate, config),
                "attempts": [
                    {
                        "id": require_str(attempt.get("id"), "attempt.id"),
                        "density_fit": require_bool(
                            attempt.get("density_fit"), "attempt.density_fit"
                        ),
                    }
                    for attempt in _attempts(config)
                ],
            }
        )
    plan = {
        "schema": "uma-pyscf-label-plan-v1",
        "sampling_id": manifest.sampling_id,
        "manifest_sha256": canonical_json_fingerprint(manifest.to_dict()),
        "protocol_id": _protocol_id(config),
        "protocol_sha256": protocol_sha256,
        "counts": {
            "total": len(records),
            "ready": sum(record["status"] == "ready" for record in records),
            "blocked": sum(record["status"] == "blocked" for record in records),
        },
        "records": records,
    }
    identity = registry_identity(state_registry)
    if identity is not None:
        plan["state_registry"] = identity
    return plan


def _completed_record(
    candidate: CandidateRecord,
    output: CalculationOutput,
    config: Mapping[str, Any],
    *,
    method_density_fit: bool,
    attempt_id: str,
    protocol_sha256: str,
    candidate_sha256: str,
    raw_location: str,
    raw_sha256: str,
    utc: str,
    state_registry: StateRegistry | None,
) -> LabelRecord:
    guess, generated_on = _initial_guess(config)
    versions: dict[str, str | None] = dict(output.engine_versions)
    versions.update(
        {
            "protocol_id": _protocol_id(config),
            "protocol_sha256": protocol_sha256,
            "input_fingerprint_sha256": candidate_sha256,
            "initial_density": guess,
            "initial_density_generated_on": generated_on,
            "attempt_id": attempt_id,
            "density_fit": str(method_density_fit).lower(),
        }
    )
    identity = registry_identity(state_registry)
    if identity is not None and (
        candidate.state.charge != 0 or candidate.state.multiplicity != 1
    ):
        versions.update(identity)
    return LabelRecord(
        record_id=candidate.record_id,
        structure=candidate.structure,
        state=_candidate_with_protocol_state(candidate, config),
        method=method_from_config(config, density_fit=method_density_fit),
        engine=Engine(name=output.engine_name, versions=versions),
        results=output.results,
        raw=RawArtifact(logical_location=raw_location, checksum_sha256=raw_sha256),
        qc=QcState(
            status="pending",
            history=(
                {
                    "utc": utc,
                    "event": LABEL_EVENT,
                    "protocol_id": _protocol_id(config),
                    "protocol_sha256": protocol_sha256,
                    "input_fingerprint_sha256": candidate_sha256,
                    "attempt_id": attempt_id,
                    "density_fit": method_density_fit,
                    "initial_density": guess,
                    "initial_density_generated_on": generated_on,
                },
            ),
        ),
    )


def run_label_batch(
    manifest: CandidateManifest,
    config: Mapping[str, Any],
    output_dir: str | Path,
    adapter: CalculatorAdapter,
    *,
    now: Callable[[], str] = _utc_now,
    retry_failed: bool = False,
    state_registry: StateRegistry | None = None,
) -> dict[str, Any]:
    """Label a manifest, publishing progress after every attempt for safe resume."""
    if not isinstance(manifest, CandidateManifest):
        raise ValidationError(
            f"manifest must be CandidateManifest; got {type(manifest).__name__}."
        )
    root = Path(output_dir)
    ledger_path = root / "attempt_ledger.json"
    summary_path = root / "summary.json"
    protocol_sha256 = canonical_json_fingerprint(config)
    ledger = _load_or_create_ledger(
        ledger_path, manifest, config, protocol_sha256, state_registry
    )
    ledger_records = require_mapping(ledger["records"], "ledger.records")
    ledger["records"] = ledger_records
    counts = {"completed": 0, "skipped": 0, "failed": 0, "blocked": 0}
    stop_on_failure = require_bool(
        require_mapping(config.get("retry"), "config.retry").get("stop_on_first_failure"),
        "config.retry.stop_on_first_failure",
    )

    for candidate in manifest.records:
        existing_value = ledger_records.get(candidate.record_id)
        existing = (
            require_mapping(existing_value, f"ledger.records.{candidate.record_id}")
            if existing_value is not None
            else {"status": "pending", "attempts": []}
        )
        ledger_records[candidate.record_id] = existing
        attempts_log = list(
            require_sequence(
                existing.get("attempts", []),
                f"ledger.records.{candidate.record_id}.attempts",
            )
        )
        existing["attempts"] = attempts_log

        if existing.get("status") == "completed":
            record_path = root / require_str(
                existing.get("record_path"), f"ledger.records.{candidate.record_id}.record_path"
            )
            _record_matches_protocol(record_path, protocol_sha256)
            counts["skipped"] += 1
            continue
        if existing.get("status") == "failed" and not retry_failed:
            counts["failed"] += 1
            continue

        violations = scope_violations(candidate, config, state_registry)
        if violations:
            existing.update(
                {
                    "status": "blocked",
                    "scope_violations": list(violations),
                    "finished_utc": now(),
                }
            )
            write_json_atomic(ledger_path, ledger)
            counts["blocked"] += 1
            if stop_on_failure:
                break
            continue

        resource = resource_for_candidate(candidate, config)
        candidate_sha256 = canonical_json_fingerprint(candidate.to_dict())
        previous_failure: str | None = None
        completed = False
        for plan_index, attempt in enumerate(_attempts(config)):
            retry_on = {
                require_str(value, "attempt.retry_on[]")
                for value in require_sequence(attempt.get("retry_on"), "attempt.retry_on")
            }
            if plan_index > 0 and previous_failure not in retry_on:
                break
            attempt_id = require_str(attempt.get("id"), "attempt.id")
            density_fit = require_bool(attempt.get("density_fit"), "attempt.density_fit")
            attempt_number = len(attempts_log) + 1
            started = now()
            attempt_entry: dict[str, Any] = {
                "attempt_number": attempt_number,
                "attempt_id": attempt_id,
                "density_fit": density_fit,
                "status": "running",
                "started_utc": started,
                "resource": dict(resource),
            }
            attempts_log.append(attempt_entry)
            existing.update({"status": "running", "scope_violations": []})
            write_json_atomic(ledger_path, ledger)
            try:
                output = adapter.calculate(
                    candidate,
                    method_from_config(config, density_fit=density_fit),
                    config,
                    attempt_id=attempt_id,
                    resource=resource,
                )
                _validate_runtime(output, config)
                raw_relative = (
                    Path("raw")
                    / candidate.record_id
                    / f"{attempt_number:02d}_{attempt_id}.json"
                )
                raw_path = root / raw_relative
                raw_document = {
                    "schema": "uma-pyscf-raw-label-attempt-v1",
                    "record_id": candidate.record_id,
                    "input_fingerprint_sha256": candidate_sha256,
                    "protocol_id": _protocol_id(config),
                    "protocol_sha256": protocol_sha256,
                    "attempt_number": attempt_number,
                    "attempt_id": attempt_id,
                    "method": method_from_config(config, density_fit=density_fit).to_dict(),
                    "resource": dict(resource),
                    "engine_name": output.engine_name,
                    "engine_versions": dict(output.engine_versions),
                    "results": output.results.to_dict(),
                    "engine_payload": output.raw_payload,
                }
                identity = registry_identity(state_registry)
                if identity is not None and (
                    candidate.state.charge != 0 or candidate.state.multiplicity != 1
                ):
                    raw_document["state_registry"] = identity
                write_json_atomic(raw_path, raw_document)
                raw_sha256 = sha256_of_file(raw_path)
                record = _completed_record(
                    candidate,
                    output,
                    config,
                    method_density_fit=density_fit,
                    attempt_id=attempt_id,
                    protocol_sha256=protocol_sha256,
                    candidate_sha256=candidate_sha256,
                    raw_location=raw_relative.as_posix(),
                    raw_sha256=raw_sha256,
                    utc=now(),
                    state_registry=state_registry,
                )
                record_relative = Path("records") / f"{candidate.record_id}.json"
                record_path = root / record_relative
                write_json_atomic(record_path, record.to_dict())
                attempt_entry.update(
                    {
                        "status": "completed",
                        "finished_utc": now(),
                        "raw_path": raw_relative.as_posix(),
                        "raw_sha256": raw_sha256,
                        "record_path": record_relative.as_posix(),
                        "record_sha256": sha256_of_file(record_path),
                    }
                )
                existing.update(
                    {
                        "status": "completed",
                        "finished_utc": now(),
                        "record_path": record_relative.as_posix(),
                        "record_sha256": sha256_of_file(record_path),
                    }
                )
                counts["completed"] += 1
                completed = True
                write_json_atomic(ledger_path, ledger)
                break
            except CalculationFailure as exc:
                previous_failure = exc.category
                attempt_entry.update(
                    {
                        "status": "failed",
                        "finished_utc": now(),
                        "failure_category": exc.category,
                        "message": str(exc),
                    }
                )
                existing.update(
                    {
                        "status": "failed",
                        "finished_utc": now(),
                        "failure_category": exc.category,
                        "message": str(exc),
                    }
                )
                write_json_atomic(ledger_path, ledger)
            except (ValidationError, OSError, ValueError) as exc:
                previous_failure = "pipeline_error"
                attempt_entry.update(
                    {
                        "status": "failed",
                        "finished_utc": now(),
                        "failure_category": "pipeline_error",
                        "message": str(exc),
                    }
                )
                existing.update(
                    {
                        "status": "failed",
                        "finished_utc": now(),
                        "failure_category": "pipeline_error",
                        "message": str(exc),
                    }
                )
                write_json_atomic(ledger_path, ledger)
                break
        if not completed:
            counts["failed"] += 1
            if stop_on_failure:
                break

    summary = {
        "schema": "uma-pyscf-label-summary-v1",
        "run_id": ledger["run_id"],
        "sampling_id": manifest.sampling_id,
        "protocol_id": _protocol_id(config),
        "protocol_sha256": protocol_sha256,
        "counts": counts,
        "ledger": ledger_path.name,
        "release_allowed": False,
        "release_blockers": [
            "scientific_thresholds_pending_freeze",
            "composition_baseline_required",
            "state_registry_required_for_non_default_states",
        ],
    }
    identity = registry_identity(state_registry)
    if identity is not None:
        summary["state_registry"] = identity
    write_json_atomic(summary_path, summary)
    return summary
