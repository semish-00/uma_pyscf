"""The ``uma-pyscf evaluate-uma`` command."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

from ..core.errors import UmaPyscfError, ValidationError
from ..core.ids import sha256_of_file, validate_record_id
from ..core.io import read_json
from ..schemas._fields import (
    reject_unknown_keys,
    require_int,
    require_key,
    require_sequence,
    require_str,
    validated_json_object,
)
from ..schemas.dataset_manifest import AseDatasetManifest
from .uma import evaluate_ase_lmdb

__all__ = ["configure_evaluate_uma", "load_evaluation_config", "run_evaluate_uma"]

_CONFIG_KEYS = (
    "schema_version",
    "evaluation_id",
    "created",
    "description",
    "dataset_id",
    "model_name",
    "model_source",
    "model_license",
    "task",
    "device",
    "inference_settings",
    "seed",
    "fairchem_core_version",
    "partitions",
)
_REQUIRED_KEYS = tuple(key for key in _CONFIG_KEYS if key not in {"created", "description"})


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"Evaluation config {source} cannot be read: {exc}.") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"Evaluation config {source} is not valid YAML: {exc}.") from exc
    config = validated_json_object(loaded, "config")
    reject_unknown_keys(config, _CONFIG_KEYS, "config")
    for key in _REQUIRED_KEYS:
        require_key(config, key, "config")
    if require_int(config["schema_version"], "config.schema_version") != 1:
        raise ValidationError("config.schema_version must be 1.")
    for key in ("evaluation_id", "dataset_id"):
        validate_record_id(require_str(config[key], f"config.{key}"))
    if require_str(config["task"], "config.task") != "omol":
        raise ValidationError("config.task must be 'omol'.")
    if require_str(config["device"], "config.device") != "cuda":
        raise ValidationError("config.device must be 'cuda' for the GPU baseline.")
    settings = require_str(config["inference_settings"], "config.inference_settings")
    if settings not in {"default", "turbo", "batch"}:
        raise ValidationError("config.inference_settings must be default, turbo, or batch.")
    seed = require_int(config["seed"], "config.seed")
    if seed < 0:
        raise ValidationError("config.seed must not be negative.")
    partitions = tuple(
        validate_record_id(require_str(value, f"config.partitions[{index}]"))
        for index, value in enumerate(require_sequence(config["partitions"], "config.partitions"))
    )
    if not partitions or len(set(partitions)) != len(partitions):
        raise ValidationError("config.partitions must be non-empty and contain no duplicates.")
    require_str(config["model_name"], "config.model_name")
    require_str(config["model_source"], "config.model_source")
    require_str(config["model_license"], "config.model_license")
    require_str(config["fairchem_core_version"], "config.fairchem_core_version")
    return config


def configure_evaluate_uma(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, metavar="<config>")
    parser.add_argument("--dataset-dir", required=True, metavar="<dir>")
    parser.add_argument("--output", required=True, metavar="<json>")
    parser.add_argument("--repository", required=True, metavar="<dir>")
    parser.add_argument("--model-cache-dir", required=True, metavar="<dir>")
    parser.add_argument("--container-sha256-file", required=True, metavar="<file>")


def run_evaluate_uma(args: argparse.Namespace) -> int:
    try:
        config = load_evaluation_config(args.config)
        dataset_dir = Path(args.dataset_dir)
        manifest_path = dataset_dir / "dataset_manifest.json"
        manifest = AseDatasetManifest.from_dict(read_json(manifest_path))
        if manifest.dataset_id != config["dataset_id"]:
            raise ValidationError(
                f"Dataset is {manifest.dataset_id!r}, expected {config['dataset_id']!r}."
            )
        container_line = Path(args.container_sha256_file).read_text(encoding="utf-8").split()
        if len(container_line) < 1 or len(container_line[0]) != 64:
            raise ValidationError("Container SHA-256 file does not begin with a SHA-256 digest.")
        artifact = evaluate_ase_lmdb(
            manifest,
            manifest_sha256=sha256_of_file(manifest_path),
            dataset_dir=dataset_dir,
            evaluation_id=str(config["evaluation_id"]),
            model_name=str(config["model_name"]),
            model_source=str(config["model_source"]),
            model_license=str(config["model_license"]),
            model_cache_dir=Path(args.model_cache_dir),
            task=str(config["task"]),
            device=str(config["device"]),
            inference_settings=str(config["inference_settings"]),
            seed=int(config["seed"]),
            fairchem_core_version=str(config["fairchem_core_version"]),
            partitions=tuple(str(value) for value in config["partitions"]),
            output_path=Path(args.output),
            repository=Path(args.repository),
            container_sha256=container_line[0].lower(),
        )
    except (UmaPyscfError, OSError, ValueError) as exc:
        print(f"{args.config}: ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"evaluation={artifact['evaluation_id']} model={artifact['model']['name']} "
        f"dataset={artifact['dataset']['id']} output={args.output}"
    )
    for partition, metrics in artifact["metrics_by_partition"].items():
        print(
            f"partition={partition} records={metrics['records']} "
            f"energy_mae_ev={metrics['energy_mae_ev']:.8g} "
            f"force_mae_ev_per_angstrom={metrics['force_component_mae_ev_per_angstrom']:.8g}"
        )
    return 0
