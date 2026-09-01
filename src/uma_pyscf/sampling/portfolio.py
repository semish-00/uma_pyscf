"""Assemble score-independent calibration portfolios from candidate manifests."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import random
from typing import Any

import yaml

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, sha256_of_file, validate_record_id
from ..core.io import read_json, write_json_atomic
from ..schemas._fields import (
    reject_unknown_keys,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
    validated_json_object,
)
from ..schemas.candidate import CandidateManifest, CandidateRecord
from .filters import pair_distance_fingerprint

__all__ = [
    "PORTFOLIO_CONFIG_SCHEMA_VERSION",
    "assemble_portfolio",
    "load_portfolio_config",
    "write_portfolio_outputs",
]

PORTFOLIO_CONFIG_SCHEMA_VERSION = 1
PORTFOLIO_REPORT_SCHEMA = "uma-pyscf-candidate-portfolio-report-v1"
PORTFOLIO_SKIP_REASONS = (
    "duplicate_geometry_state",
    "parent_limit",
    "trajectory_limit",
    "quota_reached",
)

_CONFIG_KEYS = (
    "schema_version",
    "portfolio_id",
    "created",
    "description",
    "seed",
    "strategy",
    "duplicate_decimals",
    "max_per_parent",
    "max_per_trajectory",
    "sources",
)
_REQUIRED_CONFIG_KEYS = tuple(
    key for key in _CONFIG_KEYS if key not in {"created", "description", "max_per_trajectory"}
)
_SOURCE_KEYS = ("category", "manifest", "quota")
_RESERVED_PROVENANCE_KEYS = (
    "portfolio_source_category",
    "portfolio_source_manifest_id",
    "portfolio_source_manifest_sha256",
)


def load_portfolio_config(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate one score-independent portfolio config."""
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"Portfolio config {source} cannot be read: {exc}.") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"Portfolio config {source} is not valid YAML: {exc}.") from exc
    config = validated_json_object(loaded, "portfolio_config")
    reject_unknown_keys(config, _CONFIG_KEYS, "portfolio_config")
    for key in _REQUIRED_CONFIG_KEYS:
        require_key(config, key, "portfolio_config")
    version = require_int(config["schema_version"], "portfolio_config.schema_version")
    if version != PORTFOLIO_CONFIG_SCHEMA_VERSION:
        raise ValidationError(
            "portfolio_config.schema_version must be "
            f"{PORTFOLIO_CONFIG_SCHEMA_VERSION}; got {version}."
        )
    validate_record_id(require_str(config["portfolio_id"], "portfolio_config.portfolio_id"))
    if "created" in config:
        require_str(config["created"], "portfolio_config.created")
    if "description" in config:
        require_str(config["description"], "portfolio_config.description")
    require_int(config["seed"], "portfolio_config.seed")
    strategy = require_str(config["strategy"], "portfolio_config.strategy")
    if strategy != "parent_round_robin":
        raise ValidationError("portfolio_config.strategy must be 'parent_round_robin'.")
    decimals = require_int(config["duplicate_decimals"], "portfolio_config.duplicate_decimals")
    if decimals < 0:
        raise ValidationError("portfolio_config.duplicate_decimals must not be negative.")
    parent_limit = require_int(config["max_per_parent"], "portfolio_config.max_per_parent")
    if parent_limit < 1:
        raise ValidationError("portfolio_config.max_per_parent must be positive.")
    if "max_per_trajectory" in config:
        trajectory_limit = require_int(
            config["max_per_trajectory"], "portfolio_config.max_per_trajectory"
        )
        if trajectory_limit < 1:
            raise ValidationError("portfolio_config.max_per_trajectory must be positive.")
    sources = require_sequence(config["sources"], "portfolio_config.sources")
    if not sources:
        raise ValidationError("portfolio_config.sources must not be empty.")
    categories: set[str] = set()
    for index, value in enumerate(sources):
        item_path = f"portfolio_config.sources[{index}]"
        item = require_mapping(value, item_path)
        reject_unknown_keys(item, _SOURCE_KEYS, item_path)
        category = validate_record_id(
            require_str(require_key(item, "category", item_path), f"{item_path}.category")
        )
        if category in categories:
            raise ValidationError(f"{item_path}.category repeats {category!r}.")
        categories.add(category)
        manifest = Path(
            require_str(require_key(item, "manifest", item_path), f"{item_path}.manifest")
        )
        if manifest.is_absolute() or ".." in manifest.parts:
            raise ValidationError(
                f"{item_path}.manifest must be a relative path without '..'; got {manifest}."
            )
        quota = require_int(require_key(item, "quota", item_path), f"{item_path}.quota")
        if quota < 1:
            raise ValidationError(f"{item_path}.quota must be positive.")
    return config


def _parent_id(record: CandidateRecord) -> str:
    parent = record.structure.parent_structure_id
    if parent is None:
        raise ValidationError(
            f"Candidate {record.record_id!r} has no parent_structure_id; blind portfolio "
            "assembly requires parent provenance."
        )
    return validate_record_id(parent)


def _trajectory_id(record: CandidateRecord) -> str | None:
    value = record.generation_parameters.get("trajectory_id")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(
            f"Candidate {record.record_id!r} generation_parameters.trajectory_id "
            "must be a string."
        )
    return validate_record_id(value)


def _geometry_state_key(record: CandidateRecord, decimals: int) -> tuple[Any, ...]:
    return (
        tuple(sorted(record.structure.atomic_numbers)),
        pair_distance_fingerprint(record.structure, decimals),
        record.state.charge,
        record.state.multiplicity,
    )


def _parent_round_robin(
    records: tuple[CandidateRecord, ...], *, seed: int, category: str
) -> list[CandidateRecord]:
    groups: dict[str, list[CandidateRecord]] = defaultdict(list)
    for record in records:
        groups[_parent_id(record)].append(record)
    rng = random.Random(f"{seed}:{category}")
    parents = sorted(groups)
    rng.shuffle(parents)
    for parent in parents:
        groups[parent].sort(key=lambda item: item.record_id)
        rng.shuffle(groups[parent])
    ordered: list[CandidateRecord] = []
    while any(groups[parent] for parent in parents):
        for parent in parents:
            if groups[parent]:
                ordered.append(groups[parent].pop())
    return ordered


def _enrich_record(
    record: CandidateRecord, *, category: str, source_id: str, source_sha256: str
) -> CandidateRecord:
    parameters = dict(record.generation_parameters)
    conflicts = sorted(set(parameters).intersection(_RESERVED_PROVENANCE_KEYS))
    if conflicts:
        raise ValidationError(
            f"Candidate {record.record_id!r} already defines reserved portfolio provenance "
            f"keys {conflicts!r}."
        )
    parameters.update(
        {
            "portfolio_source_category": category,
            "portfolio_source_manifest_id": source_id,
            "portfolio_source_manifest_sha256": source_sha256,
        }
    )
    return CandidateRecord(
        record_id=record.record_id,
        structure=record.structure,
        state=record.state,
        generation_parameters=parameters,
    )


def assemble_portfolio(
    config_path: str | Path, source_root: str | Path
) -> tuple[CandidateManifest, dict[str, Any]]:
    """Assemble a deterministic, model-score-blind candidate portfolio."""
    config = load_portfolio_config(config_path)
    root = Path(source_root)
    loaded_sources: list[tuple[dict[str, Any], CandidateManifest, str]] = []
    all_record_ids: dict[str, str] = {}
    resolved = deepcopy(config)
    resolved_sources: list[dict[str, Any]] = []
    for raw_source in config["sources"]:
        source = dict(raw_source)
        relative_path = Path(str(source["manifest"]))
        manifest_path = root / relative_path
        manifest = CandidateManifest.from_dict(read_json(manifest_path))
        manifest_sha256 = sha256_of_file(manifest_path)
        for record in manifest.records:
            previous = all_record_ids.get(record.record_id)
            if previous is not None:
                raise ValidationError(
                    f"Candidate record id {record.record_id!r} appears in both {previous!r} "
                    f"and {manifest.sampling_id!r}."
                )
            all_record_ids[record.record_id] = manifest.sampling_id
        loaded_sources.append((source, manifest, manifest_sha256))
        resolved_sources.append(
            {
                "category": source["category"],
                "manifest": relative_path.as_posix(),
                "source_id": manifest.sampling_id,
                "source_sha256": manifest_sha256,
                "quota": source["quota"],
            }
        )
    resolved["resolved_sources"] = resolved_sources

    decimals = int(config["duplicate_decimals"])
    parent_limit = int(config["max_per_parent"])
    trajectory_limit = (
        int(config["max_per_trajectory"]) if "max_per_trajectory" in config else None
    )
    seed = int(config["seed"])
    selected_records: list[CandidateRecord] = []
    selected_fingerprints: set[tuple[Any, ...]] = set()
    parent_counts: dict[str, int] = defaultdict(int)
    trajectory_counts: dict[str, int] = defaultdict(int)
    source_summaries: list[dict[str, Any]] = []

    for source, manifest, manifest_sha256 in loaded_sources:
        category = str(source["category"])
        quota = int(source["quota"])
        ordered = _parent_round_robin(manifest.records, seed=seed, category=category)
        selected: list[CandidateRecord] = []
        skipped = {reason: 0 for reason in PORTFOLIO_SKIP_REASONS}
        for index, record in enumerate(ordered):
            if len(selected) == quota:
                skipped["quota_reached"] += len(ordered) - index
                break
            key = _geometry_state_key(record, decimals)
            if key in selected_fingerprints:
                skipped["duplicate_geometry_state"] += 1
                continue
            parent = _parent_id(record)
            if parent_counts[parent] >= parent_limit:
                skipped["parent_limit"] += 1
                continue
            trajectory = _trajectory_id(record)
            if (
                trajectory_limit is not None
                and trajectory is not None
                and trajectory_counts[trajectory] >= trajectory_limit
            ):
                skipped["trajectory_limit"] += 1
                continue
            enriched = _enrich_record(
                record,
                category=category,
                source_id=manifest.sampling_id,
                source_sha256=manifest_sha256,
            )
            selected.append(enriched)
            selected_records.append(enriched)
            selected_fingerprints.add(key)
            parent_counts[parent] += 1
            if trajectory is not None:
                trajectory_counts[trajectory] += 1
        if len(selected) != quota:
            raise ValidationError(
                f"Portfolio source {category!r} could select only {len(selected)} of quota "
                f"{quota}; available={len(ordered)}, skipped={skipped}."
            )
        source_summaries.append(
            {
                "category": category,
                "source_id": manifest.sampling_id,
                "source_sha256": manifest_sha256,
                "quota": quota,
                "available_count": len(manifest.records),
                "selected_record_ids": [record.record_id for record in selected],
                "skipped_counts": skipped,
            }
        )

    config_sha256 = canonical_json_fingerprint(resolved)
    output_manifest = CandidateManifest(
        sampling_id=str(config["portfolio_id"]),
        config_sha256=config_sha256,
        config=resolved,
        records=tuple(selected_records),
    )
    report: dict[str, Any] = {
        "schema": PORTFOLIO_REPORT_SCHEMA,
        "portfolio_id": str(config["portfolio_id"]),
        "config_sha256": config_sha256,
        "config": resolved,
        "counts": {
            "source_manifests": len(source_summaries),
            "available": sum(int(item["available_count"]) for item in source_summaries),
            "selected": len(selected_records),
        },
        "sources": source_summaries,
    }
    report_ids = tuple(
        record_id
        for source in source_summaries
        for record_id in source["selected_record_ids"]
    )
    if tuple(record.record_id for record in output_manifest.records) != report_ids:
        raise ValidationError("Portfolio manifest order does not match its report.")
    return output_manifest, report


def write_portfolio_outputs(
    manifest: CandidateManifest, report: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Atomically publish the selected candidate manifest and audit report."""
    root = Path(output_dir)
    manifest_path = root / f"{manifest.sampling_id}_candidates.json"
    report_path = root / f"{report['portfolio_id']}_portfolio_report.json"
    write_json_atomic(manifest_path, manifest.to_dict())
    write_json_atomic(report_path, report)
    return manifest_path, report_path
