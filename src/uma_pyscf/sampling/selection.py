"""Deterministic, parent-capped acquisition selection."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import yaml

from ..core.errors import ValidationError
from ..core.ids import canonical_json_fingerprint, sha256_of_file, validate_record_id
from ..core.io import read_json, write_json_atomic
from ..schemas._fields import (
    reject_unknown_keys,
    require_bool,
    require_int,
    require_key,
    require_mapping,
    require_sequence,
    require_str,
)
from ..schemas.acquisition import (
    AcquisitionScoreManifest,
    AcquisitionScoreRecord,
    SelectionManifest,
)

__all__ = [
    "load_selection_config",
    "run_selection",
    "select_candidates",
]

_CONFIG_KEYS = (
    "schema_version",
    "selection_id",
    "created",
    "description",
    "dry_run",
    "seed",
    "max_union_records",
    "max_per_parent",
    "max_per_trajectory",
    "policies",
)
_POLICY_KEYS = ("name", "kind", "score", "direction", "budget")


def load_selection_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"{source} is not valid YAML: {exc}.") from exc
    config = require_mapping(raw, "selection_config")
    reject_unknown_keys(config, _CONFIG_KEYS, "selection_config")
    if (
        require_int(
            require_key(config, "schema_version", "selection_config"),
            "selection_config.schema_version",
        )
        != 1
    ):
        raise ValidationError("selection_config.schema_version must be 1.")
    validate_record_id(
        require_str(
            require_key(config, "selection_id", "selection_config"),
            "selection_config.selection_id",
        )
    )
    require_str(require_key(config, "created", "selection_config"), "selection_config.created")
    require_str(
        require_key(config, "description", "selection_config"),
        "selection_config.description",
    )
    require_bool(require_key(config, "dry_run", "selection_config"), "selection_config.dry_run")
    require_int(require_key(config, "seed", "selection_config"), "selection_config.seed")
    for key in ("max_union_records", "max_per_parent"):
        value = require_int(
            require_key(config, key, "selection_config"), f"selection_config.{key}"
        )
        if value < 1:
            raise ValidationError(f"selection_config.{key} must be positive.")
    if "max_per_trajectory" in config:
        trajectory_limit = require_int(
            config["max_per_trajectory"], "selection_config.max_per_trajectory"
        )
        if trajectory_limit < 1:
            raise ValidationError("selection_config.max_per_trajectory must be positive.")
    policies = require_sequence(
        require_key(config, "policies", "selection_config"), "selection_config.policies"
    )
    if not policies:
        raise ValidationError("selection_config.policies must not be empty.")
    names: set[str] = set()
    for index, raw_policy in enumerate(policies):
        path_name = f"selection_config.policies[{index}]"
        policy = require_mapping(raw_policy, path_name)
        reject_unknown_keys(policy, _POLICY_KEYS, path_name)
        name = validate_record_id(
            require_str(require_key(policy, "name", path_name), f"{path_name}.name")
        )
        if name in names:
            raise ValidationError(f"{path_name}.name repeats {name!r}.")
        names.add(name)
        kind = require_str(require_key(policy, "kind", path_name), f"{path_name}.kind")
        if kind not in ("random", "score"):
            raise ValidationError(f"{path_name}.kind must be 'random' or 'score'.")
        budget = require_int(require_key(policy, "budget", path_name), f"{path_name}.budget")
        if budget < 1:
            raise ValidationError(f"{path_name}.budget must be positive.")
        if kind == "score":
            validate_record_id(
                require_str(require_key(policy, "score", path_name), f"{path_name}.score")
            )
            direction = require_str(
                require_key(policy, "direction", path_name), f"{path_name}.direction"
            )
            if direction not in ("max", "min"):
                raise ValidationError(f"{path_name}.direction must be 'max' or 'min'.")
        elif "score" in policy or "direction" in policy:
            raise ValidationError(f"{path_name} is random and must not define score or direction.")
    return config


def _ordered_records(
    records: tuple[AcquisitionScoreRecord, ...], policy: dict[str, Any], seed: int
) -> list[AcquisitionScoreRecord]:
    ordered = sorted(records, key=lambda item: item.record_id)
    if policy["kind"] == "random":
        random.Random(f"{seed}:{policy['name']}").shuffle(ordered)
        return ordered
    score_name = str(policy["score"])
    missing = [record.record_id for record in ordered if score_name not in record.scores]
    if missing:
        raise ValidationError(
            f"Policy {policy['name']!r} requires score {score_name!r}, missing from "
            f"{len(missing)} record(s), including {missing[0]!r}."
        )
    reverse = policy["direction"] == "max"
    if reverse:
        return sorted(ordered, key=lambda item: (-item.scores[score_name], item.record_id))
    return sorted(ordered, key=lambda item: (item.scores[score_name], item.record_id))


def select_candidates(
    scores: AcquisitionScoreManifest,
    config: dict[str, Any],
    *,
    score_file_sha256: str,
) -> SelectionManifest:
    """Apply every policy independently, enforcing a parent quota and union cap."""
    max_per_parent = int(config["max_per_parent"])
    max_per_trajectory = (
        int(config["max_per_trajectory"]) if "max_per_trajectory" in config else None
    )
    if max_per_trajectory is not None:
        missing_trajectory = [
            record.record_id for record in scores.records if record.trajectory_id is None
        ]
        if missing_trajectory:
            raise ValidationError(
                "selection_config.max_per_trajectory requires trajectory_id on every score "
                f"record; missing on {missing_trajectory[0]!r}."
            )
    selected_by_policy: dict[str, tuple[str, ...]] = {}
    for raw_policy in config["policies"]:
        policy = dict(raw_policy)
        parent_counts: dict[str, int] = {}
        trajectory_counts: dict[str, int] = {}
        selected: list[str] = []
        for record in _ordered_records(scores.records, policy, int(config["seed"])):
            if parent_counts.get(record.parent_id, 0) >= max_per_parent:
                continue
            if (
                max_per_trajectory is not None
                and trajectory_counts.get(str(record.trajectory_id), 0) >= max_per_trajectory
            ):
                continue
            selected.append(record.record_id)
            parent_counts[record.parent_id] = parent_counts.get(record.parent_id, 0) + 1
            if record.trajectory_id is not None:
                trajectory_counts[record.trajectory_id] = (
                    trajectory_counts.get(record.trajectory_id, 0) + 1
                )
            if len(selected) == int(policy["budget"]):
                break
        if len(selected) != int(policy["budget"]):
            raise ValidationError(
                f"Policy {policy['name']!r} selected {len(selected)} records, expected "
                f"{policy['budget']}; pool size or max_per_parent is too small."
            )
        selected_by_policy[str(policy["name"])] = tuple(selected)
    union_ids = tuple(sorted({item for ids in selected_by_policy.values() for item in ids}))
    if len(union_ids) > int(config["max_union_records"]):
        raise ValidationError(
            f"Policy union has {len(union_ids)} records, exceeding max_union_records="
            f"{config['max_union_records']}. No records were written."
        )
    return SelectionManifest(
        selection_id=str(config["selection_id"]),
        source={"id": scores.score_id, "sha256": score_file_sha256},
        config_sha256=canonical_json_fingerprint(config),
        config=config,
        policy_selections=selected_by_policy,
        union_record_ids=union_ids,
    )


def run_selection(
    score_path: str | Path, config_path: str | Path, output_path: str | Path
) -> SelectionManifest:
    score_file = Path(score_path)
    scores = AcquisitionScoreManifest.from_dict(read_json(score_file))
    config = load_selection_config(config_path)
    manifest = select_candidates(
        scores,
        config,
        score_file_sha256=sha256_of_file(score_file),
    )
    write_json_atomic(output_path, manifest.to_dict())
    return manifest
